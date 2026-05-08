"""Run a chosen (template, shot-count) configuration on the missing open
LLMs by replaying the EXACT prompts from the legacy DCS-MT closed-API
dump against vLLM-served open weights.

Why this script exists
----------------------
The closed-API audit (2026-05-08) settled on a universal ICL config:

    template5 with 10 shots (random-retrieval exemplars), T=0

picked because it wins or ties on Gemini-1.5-Flash, GPT-4o, Llama3-8B,
and Gemma2-9B. The remaining open LLMs in our headline tables --
AfriqueLlama-8B, AfriqueQwen-8B, Tiny-Aya-Earth, Tiny-Aya-Global,
TranslateGemma-12B -- still only have vanilla `icl_5shot` data, so
Table 2 currently mixes prompt families across rows.

This script fills that gap. For each (model, src, tgt) tuple it:

  1. Reads `<legacy>/<src>-to-<tgt>-template5-shots10.tsv` (T=0 layout)
     or `template5-shot-10.tsv` (T=0.6 layout) from the closed-API
     dump. The `prompt` column is in LangChain debug format
     ("System: <sys>\\nHuman: <user>").
  2. Parses each row into (system, user). When the system text is
     constant across the TSV (templates 1/2/3) we forward it as a
     proper system message; when it varies per row (templates 4/5
     glossary) we flatten "system\\n\\nuser" into a single user
     message so prompt content stays byte-identical to what closed
     APIs received.
  3. Runs vLLM batched inference with greedy decoding, applying the
     model's chat template (if any) so each request becomes
     <chat-template>(system?, user) -> hyp.
  4. Scores ChrF + SSA-COMET inline and writes canonical
     predictions.json / results.json.

Source of prompts (preferred -> fallback):

    /ext_data/idris/DCS-MT-Updated/old_experiments/LLMS
    /home/idris/ext_data/DCS-MT-Old/LLMS/translations
    /Users/MacbookPro/Documents/GitHub/DCS-MT/LLMS/translations

By default we pull from `gemini-1.5-flash-002_temp_0.0/` because that
dump has bidirectional template5_10shot coverage; the GPT-4o T=0 dump
only has xxx->eng, T=0.6 dump only has eng->xxx. Override with
`--legacy-dir` if you want a different prompt source.

Outputs land at:

    experiments/llm/open/icl/<short_name>/template{N}_{K}shot/<pair>/

so the next `python -m afriscience_mt aggregate` picks them up.

Usage
-----

    # Default: template5_10shot on the 5 missing open LLMs
    python scripts/rerun_open_llm_template.py

    # Subset of models / pairs / shots
    python scripts/rerun_open_llm_template.py \\
        --models afriquellama_8b tiny_aya_global \\
        --pairs eng-amh eng-yor \\
        --template 5 --shots 10

    # If you have a different chosen config
    python scripts/rerun_open_llm_template.py --template 5 --shots 5

    # Override the legacy dump location (Mac local checkout)
    python scripts/rerun_open_llm_template.py \\
        --legacy-dir /Users/MacbookPro/Documents/GitHub/DCS-MT/LLMS/translations
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "data" / "processed_files" / "test"
EXP_ROOT = ROOT / "experiments" / "llm" / "open" / "icl"

# Same fallback chain as rerun_gemini_newer.py.
LEGACY_DUMP_CANDIDATES = [
    Path("/ext_data/idris/DCS-MT-Updated/old_experiments/LLMS"),
    Path("/home/idris/ext_data/DCS-MT-Old/LLMS/translations"),
    Path("/Users/MacbookPro/Documents/GitHub/DCS-MT/LLMS/translations"),
]
DEFAULT_PROMPT_SOURCE_MODEL = "gemini-1.5-flash-002"  # bidirectional T=0 templated coverage
DEFAULT_PROMPT_SOURCE_TEMP = 0.0

# Open-LLM HF ids — confirmed from existing zero-shot results.json files.
DEFAULT_MODELS: dict[str, str] = {
    # missing from the templated sweep (this script's headline targets)
    "afriquellama_8b":   "McGill-NLP/AfriqueLlama-8B",
    "afriqueqwen_8b":    "McGill-NLP/AfriqueQwen-8B",
    "tiny_aya_earth":    "CohereLabs/tiny-aya-earth",
    "tiny_aya_global":   "CohereLabs/tiny-aya-global",
    "translategemma_12b": "Infomaniak-AI/vllm-translategemma-12b-it",
}

LANGS = ["amh", "hau", "lug", "nso", "yor", "zul"]
PAIRS = [f"eng-{l}" for l in LANGS] + [f"{l}-eng" for l in LANGS]
EXPECTED_N = 970


# ---------------- legacy dump discovery ----------------

def find_legacy_dump(temp: float, source_model: str,
                     override: Path | None = None) -> Path:
    sub = f"{source_model}_temp_{temp:.1f}"
    candidates = [override] if override else LEGACY_DUMP_CANDIDATES
    tried = []
    for root in candidates:
        if root is None:
            continue
        for d in (root / sub, root):
            if d.is_dir() and any(d.glob("*-template*.tsv")):
                return d
            tried.append(str(d))
    raise SystemExit(
        f"no legacy dump found for {sub}. Tried: {tried}"
    )


def find_tsv(legacy_dir: Path, src: str, tgt: str,
             template: int, shots: int) -> Path | None:
    """Match the two filename conventions seen across DCS-MT dumps."""
    candidates = [
        legacy_dir / f"{src}-to-{tgt}-template{template}-shots{shots}.tsv",
        legacy_dir / f"{src}-to-{tgt}-template{template}-shot-{shots}.tsv",
    ]
    if shots == 0:
        candidates.insert(0, legacy_dir / f"{src}-to-{tgt}-template{template}.tsv")
    for c in candidates:
        if c.is_file():
            return c
    return None


# ---------------- prompt parsing ----------------

def parse_prompt(saved: str) -> tuple[str, str]:
    """LangChain debug format: 'System: <sys>\\nHuman: <user>' -> (sys, user).
    If no Human: marker, return ('', saved) to send as a single user message."""
    if "\nHuman: " not in saved:
        return "", saved
    sys_part, user_part = saved.split("\nHuman: ", 1)
    if sys_part.startswith("System: "):
        sys_part = sys_part[len("System: "):]
    return sys_part, user_part


def read_prompts(tsv: Path) -> list[tuple[str, str]]:
    out = []
    with open(tsv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            p = r.get("prompt") or ""
            out.append(parse_prompt(p))
    return out


def system_is_constant(prompts: list[tuple[str, str]]) -> bool:
    if not prompts:
        return True
    first = prompts[0][0]
    return all(p[0] == first for p in prompts)


# ---------------- test-set helpers ----------------

def load_test_lang(lang: str) -> list[str]:
    with open(TEST_DIR / f"{lang}.tsv") as f:
        return [r["text"] for r in csv.DictReader(f, delimiter="\t")]


# ---------------- canonical IO ----------------

def write_canonical(*, out_dir: Path, model_name: str, short_name: str,
                    src: str, tgt: str, template: int, shots: int,
                    temp: float, hyps: list[str], system_constant: bool,
                    prompt_source_model: str, prompt_source_temp: float,
                    metrics: dict | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pair = f"{src}-{tgt}"
    strategy = f"template{template}_{shots}shot"
    exp_id = f"llm_icl__{short_name}__{strategy}__{pair}"
    refs = load_test_lang(tgt)
    srcs = load_test_lang(src)
    n = min(len(hyps), EXPECTED_N)
    samples = [
        {"src": srcs[i] if i < len(srcs) else "",
         "ref": refs[i] if i < len(refs) else "",
         "hyp": hyps[i]}
        for i in range(n)
    ]
    (out_dir / "predictions.json").write_text(json.dumps({
        "schema_version": "1.0",
        "experiment_id": exp_id,
        "source_lang": src, "target_lang": tgt,
        "samples": samples,
    }, indent=2, ensure_ascii=False))
    res_metrics = metrics or {
        "bleu": None, "chrf": None, "ssa_comet": None, "num_samples": n,
    }
    (out_dir / "results.json").write_text(json.dumps({
        "schema_version": "1.0",
        "experiment_id": exp_id,
        "experiment_type": "llm_icl",
        "timestamp": datetime.utcnow().isoformat(),
        "source_lang": src, "target_lang": tgt, "lang_pair": pair,
        "split": "test", "dataset": "dcs",
        "model": {"name": model_name, "type": "llm", "short_name": short_name},
        "config": {
            "prompt_strategy": strategy, "n_shots": shots,
            "temperature": temp, "template": template,
            "selection_method": "random",
        },
        "metrics": res_metrics,
        "metadata": {
            "source": "rerun_open_llm_template",
            "is_ablation": False,
            "prompt_strategy": strategy,
            "n_shots": shots,
            "prompt_provenance": (
                f"replayed_from_{prompt_source_model}_temp_{prompt_source_temp:.1f}"
            ),
            "system_constant_per_tsv": system_constant,
            "comet_metric": "ssa-comet",
            "needs_eval": metrics is None,
        },
    }, indent=2, ensure_ascii=False))


# ---------------- driver ----------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None,
                    help="model_short keys to run (default: 5 missing open LLMs).")
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="lang pairs (default: all 12).")
    ap.add_argument("--template", type=int, default=5,
                    help="prompt template number (default: 5 — universal best).")
    ap.add_argument("--shots", type=int, default=10,
                    help="shot count (default: 10 — best for 3/4 of {Gemini, GPT-4o, Llama3, Gemma2}).")
    ap.add_argument("--legacy-dir", default=None,
                    help="override legacy dump root.")
    ap.add_argument("--prompt-source-model", default=DEFAULT_PROMPT_SOURCE_MODEL,
                    help="legacy model whose dump we read prompts from "
                         "(default: gemini-1.5-flash-002, bidirectional T=0).")
    ap.add_argument("--prompt-source-temp", type=float,
                    default=DEFAULT_PROMPT_SOURCE_TEMP,
                    help="legacy dump temperature to read FROM (default 0.0).")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="open-LLM inference temperature (default 0.0).")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    ap.add_argument("--quantization", default=None,
                    help="optional vLLM quantization (e.g. 'awq', 'gptq').")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip configs whose results.json already exists.")
    ap.add_argument("--skip-scoring", action="store_true",
                    help="write predictions only; defer ChrF + SSA-COMET.")
    args = ap.parse_args()

    if args.shots > 20:
        print("error: shot counts > 20 are forbidden (50-shot dropped 2026-05-08).",
              file=sys.stderr)
        return 2

    models = args.models or list(DEFAULT_MODELS.keys())
    bad_models = [m for m in models if m not in DEFAULT_MODELS]
    if bad_models:
        print(f"error: unknown model_short {bad_models}. "
              f"Known: {list(DEFAULT_MODELS.keys())}", file=sys.stderr)
        return 2

    pairs = args.pairs or PAIRS
    bad_pairs = [p for p in pairs if p not in PAIRS]
    if bad_pairs:
        print(f"error: unknown pairs {bad_pairs}", file=sys.stderr)
        return 2

    legacy_dir = find_legacy_dump(
        args.prompt_source_temp, args.prompt_source_model,
        override=Path(args.legacy_dir).expanduser() if args.legacy_dir else None,
    )
    print(f"prompt source: {legacy_dir}")
    print(f"config: template{args.template}_{args.shots}shot  T={args.temperature}")
    print(f"models: {models}")
    print(f"pairs: {pairs}")

    # Load all the prompts once per pair (same for every model).
    prompts_per_pair: dict[str, tuple[list[tuple[str, str]], bool, Path]] = {}
    for pair in pairs:
        src, tgt = pair.split("-")
        tsv = find_tsv(legacy_dir, src, tgt, args.template, args.shots)
        if tsv is None:
            print(f"  WARN: no TSV for {pair} template{args.template}_{args.shots}shot; skipping")
            continue
        ps = read_prompts(tsv)
        prompts_per_pair[pair] = (ps, system_is_constant(ps), tsv)
        print(f"  {pair}: {len(ps)} prompts from {tsv.name} (system_constant={system_is_constant(ps)})")

    if not prompts_per_pair:
        print("error: no prompts loaded for any pair.", file=sys.stderr)
        return 2

    # Lazy-import vLLM only now (so --help works without it).
    from afriscience_mt.inference.vllm_client import VLLMClient

    t0 = time.time()
    for short_name in models:
        model_name = DEFAULT_MODELS[short_name]
        print(f"\n=== loading {short_name} ({model_name}) ===")
        client = VLLMClient(
            model_name=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_mem_util,
            max_model_len=args.max_model_len,
            quantization=args.quantization,
            max_new_tokens=args.max_new_tokens,
            trust_remote_code=True,
        )

        for pair in pairs:
            if pair not in prompts_per_pair:
                continue
            src, tgt = pair.split("-")
            strategy = f"template{args.template}_{args.shots}shot"
            out_dir = EXP_ROOT / short_name / strategy / pair
            res_path = out_dir / "results.json"
            if args.skip_existing and res_path.exists():
                print(f"  skip {pair} (already exists)")
                continue

            ps, sys_const, _ = prompts_per_pair[pair]

            ts = time.time()
            print(f"  [{pair}] generating {len(ps)} prompts ...")

            # Send proper system+user when system is constant across the TSV;
            # otherwise flatten so prompt content stays byte-identical.
            if sys_const and ps[0][0]:
                user_only = [u for _, u in ps]
                hyps = client.generate_batch(
                    prompts=user_only,
                    system_prompt=ps[0][0],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
            else:
                flat = [(f"{s}\n\n{u}" if s else u) for s, u in ps]
                hyps = client.generate_batch(
                    prompts=flat,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )

            metrics = None
            if not args.skip_scoring:
                try:
                    from afriscience_mt.evaluation.metrics import compute_all_metrics
                    srcs = load_test_lang(src)
                    refs = load_test_lang(tgt)
                    n = min(len(hyps), len(srcs), len(refs))
                    m = compute_all_metrics(
                        srcs[:n], hyps[:n], refs[:n],
                        metrics=["chrf", "ssa_comet"],
                    )
                    metrics = {
                        "chrf": m.get("chrf"),
                        "ssa_comet": m.get("ssa_comet"),
                        "num_samples": n,
                    }
                    print(f"    ChrF={metrics['chrf']:.2f}  "
                          f"COMET={100*metrics['ssa_comet']:.1f}  "
                          f"({time.time()-ts:.1f}s)")
                except Exception as e:
                    print(f"    scoring failed ({type(e).__name__}: {e}); writing predictions only")

            write_canonical(
                out_dir=out_dir, model_name=model_name, short_name=short_name,
                src=src, tgt=tgt, template=args.template, shots=args.shots,
                temp=args.temperature, hyps=hyps, system_constant=sys_const,
                prompt_source_model=args.prompt_source_model,
                prompt_source_temp=args.prompt_source_temp,
                metrics=metrics,
            )

        # release GPU before next model
        del client

    print(f"\nTotal: {time.time() - t0:.1f}s")
    print("Aggregate to update summary.csv:")
    print("  python -m afriscience_mt aggregate --out experiments/summary.csv")
    print("Then rebuild paper tables:")
    print("  python scripts/build_paper_tables.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
