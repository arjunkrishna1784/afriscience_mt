"""Rerun GPT-4o (or any newer OpenAI model) using the EXACT prompts that
were sent to the legacy gpt-4o-temp-0.0 runs, read from the legacy TSV
dump's `prompt` column. No prompt rebuilding — just resend.

This guarantees byte-for-byte prompt parity with the legacy gpt-4o
predictions, so the new outputs support a clean "model version"
ablation analogous to the Gemini rerun.

The closed-API audit (2026-05-08) found one structural gap: GPT-4o
``eng->xxx`` at T=0 only has the semantic-similarity strategies; the
templated runs (templates 1..5) live exclusively at T=0.6. Re-running
with the *same prompts* against a newer model fills that gap (and
also gives us a prompt-identical comparison to the original GPT-4o).

Source of prompts (preferred -> fallback):
    /ext_data/idris/DCS-MT-Updated/old_experiments/LLMS/gpt-4o_temp_<T>/
    /home/idris/ext_data/DCS-MT-Old/LLMS/translations/gpt-4o_temp_<T>/
    /Users/MacbookPro/Documents/GitHub/DCS-MT/LLMS/translations/gpt-4o_temp_<T>/

TSV format:  src_text \\t prompt \\t translation
The `prompt` column is in LangChain debug format
("System: <sys>\\nHuman: <user>"); we parse that and send proper
system + user messages.

Sweep: every TSV in the legacy dump for the chosen temperature.
50-shot files are skipped per project policy (docs/PAPER_TODOS.md).

Outputs land at:
    experiments/llm/closed/<temp>/icl/<short_name>/template{N}_{K}shot/<pair>/
        predictions.json + results.json

Usage:
    export OPENAI_API_KEY=...
    python scripts/rerun_gpt4o_templated.py --model gpt-4o
    python scripts/rerun_gpt4o_templated.py --model gpt-5 --temperature 0.6
    python scripts/rerun_gpt4o_templated.py --model gpt-4o --use-batch
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "data" / "processed_files" / "test"
EXP_ROOT = ROOT / "experiments" / "llm" / "closed"

# Preferred source first; fall back to local checkout / DCS-MT-Old.
LEGACY_DUMP_CANDIDATES = [
    Path("/ext_data/idris/DCS-MT-Updated/old_experiments/LLMS"),
    Path("/home/idris/ext_data/DCS-MT-Old/LLMS/translations"),
    Path("/Users/MacbookPro/Documents/GitHub/DCS-MT/LLMS/translations"),
]

LANGS = ["amh", "hau", "lug", "nso", "yor", "zul"]
PAIRS = [f"eng-{l}" for l in LANGS] + [f"{l}-eng" for l in LANGS]
EXPECTED_N = 970


def find_legacy_dump(temp: float) -> Path:
    sub = f"gpt-4o_temp_{temp:.1f}"
    for root in LEGACY_DUMP_CANDIDATES:
        d = root / sub
        if d.is_dir():
            return d
    raise SystemExit(f"no legacy dump found for {sub} under {LEGACY_DUMP_CANDIDATES}")


def list_legacy_configs(legacy_dir: Path):
    """Yield (template, shots, src, tgt, tsv_path) from the dump.
    Skips 50-shot per project policy."""
    rx = re.compile(
        r"^(?P<src>[a-z]+)-to-(?P<tgt>[a-z]+)-template(?P<n>\d+)-shots(?P<k>\d+)(?P<old>(-old)?)\.tsv$"
    )
    out = []
    for tsv in sorted(legacy_dir.glob("*.tsv")):
        m = rx.match(tsv.name)
        if not m or m.group("old"):
            continue
        k = int(m.group("k"))
        if k > 20:
            continue
        out.append((int(m.group("n")), k, m.group("src"), m.group("tgt"), tsv))
    return out


def sanitise_short(name: str) -> str:
    return name.replace("-", "_").replace(".", "_").lower()


def temp_setting(temp: float) -> str:
    return "t0" if abs(temp) < 1e-9 else f"t{int(round(temp * 10))}".replace("0", "")


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


def load_test_refs(lang: str) -> list[str]:
    with open(TEST_DIR / f"{lang}.tsv") as f:
        return [r["text"] for r in csv.DictReader(f, delimiter="\t")]


def load_test_sources(lang: str) -> list[str]:
    return load_test_refs(lang)


def cache_path_for(out_dir: Path) -> Path:
    return out_dir / "_responses.jsonl"


def load_cache(cache_path: Path) -> dict[int, str]:
    if not cache_path.exists():
        return {}
    done = {}
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                done[int(d["i"])] = d.get("hyp") or ""
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return done


def append_cache(cache_path: Path, idx: int, hyp: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"i": idx, "hyp": hyp}, ensure_ascii=False) + "\n")


def system_is_constant(prompts: list[tuple[str, str]]) -> bool:
    """True iff every row's system part is identical (templates 1/2/3),
    False if it varies per row (templates 4/5 with row-specific glossary)."""
    if not prompts:
        return True
    first = prompts[0][0]
    return all(p[0] == first for p in prompts)


def call_one(api_client, system: str, user: str) -> str:
    """One synchronous API call routed through OpenAIClient._call_api."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return (api_client._call_api(messages) or "").strip()


def submit_batch(api_client, prompts: list[tuple[str, str]],
                 *, job_label: str, poll_seconds: int,
                 state_file: Path) -> list[str]:
    """Use the OpenAI Batch API. Each row sent as a flattened user prompt
    'system\\n\\nuser' since the existing OpenAIClient.submit_batch_job
    takes List[str], not List[messages]."""
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    flat = [(f"{s}\n\n{u}" if s else u) for s, u in prompts]

    job_id = state.get(job_label)
    if not job_id:
        print(f"    submitting batch ({len(flat)} prompts)...")
        job_id = api_client.submit_batch_job(prompts=flat, job_name=job_label)
        state[job_label] = job_id
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(state, indent=2))
        print(f"    submitted: {job_id}")
    else:
        print(f"    resuming batch: {job_id}")

    while True:
        out = api_client.retrieve_batch_results(job_id)
        if out is not None:
            print(f"    batch complete ({len(out)} responses)")
            state.pop(job_label, None)
            state_file.write_text(json.dumps(state, indent=2))
            return out
        print(f"    pending; sleeping {poll_seconds}s...")
        time.sleep(poll_seconds)


def write_canonical(*, out_dir: Path, model: str, short_name: str,
                    src: str, tgt: str, template: int, n_shots: int,
                    temp: float, hyps: list[str], system_constant: bool,
                    metrics: dict | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pair = f"{src}-{tgt}"
    strategy = f"template{template}_{n_shots}shot"
    exp_id = f"llm_icl__{short_name}__{strategy}__{pair}"

    refs = load_test_refs(tgt)
    srcs = load_test_sources(src)
    n = min(len(hyps), EXPECTED_N)
    samples = [
        {
            "src": srcs[i] if i < len(srcs) else "",
            "ref": refs[i] if i < len(refs) else "",
            "hyp": hyps[i],
        }
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
        "model": {"name": model, "type": "llm", "short_name": short_name},
        "config": {
            "prompt_strategy": strategy, "n_shots": n_shots,
            "temperature": temp, "temp_setting": temp_setting(temp),
        },
        "metrics": res_metrics,
        "metadata": {
            "source": "rerun_gpt4o_templated",
            "is_ablation": True,
            "prompt_strategy": strategy,
            "n_shots": n_shots,
            "temp_setting": temp_setting(temp),
            "prompt_provenance": "imported_from_legacy_gpt4o_dump",
            "system_constant_per_tsv": system_constant,
            "needs_eval": metrics is None,
        },
    }, indent=2, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o",
                    help="OpenAI model name (e.g. gpt-4o, gpt-5).")
    ap.add_argument("--short-name", default=None,
                    help="model_short for summary.csv. Defaults to sanitised --model.")
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="Lang pairs to run (default: every pair found in dump).")
    ap.add_argument("--templates", type=int, nargs="+", default=None,
                    help="Restrict to these templates (default: all in dump).")
    ap.add_argument("--shot-counts", type=int, nargs="+", default=None,
                    help="Restrict to these shot counts (default: all in dump, "
                         "but 50-shot is always skipped).")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Inference temperature for the new model (default 0.0).")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--api-key", default=None,
                    help="OpenAI API key (default: $OPENAI_API_KEY).")
    ap.add_argument("--use-batch", action="store_true",
                    help="Submit prompts via the OpenAI Batch API "
                         "(50% cheaper, ~24h SLA, no per-minute caps).")
    ap.add_argument("--batch-poll-seconds", type=int, default=120)
    ap.add_argument("--batch-state-file",
                    default="experiments/_openai_batch_state.json")
    ap.add_argument("--legacy-temp", type=float, default=0.0,
                    help="Legacy dump temperature to read prompts FROM "
                         "(default 0.0). The new model is run at "
                         "--temperature; these can differ if you want to "
                         "ablate temperature with identical prompts.")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("error: pass --api-key or set OPENAI_API_KEY", file=sys.stderr)
        return 2

    if args.shot_counts and any(k > 20 for k in args.shot_counts):
        print("error: shot counts > 20 are forbidden (50-shot dropped 2026-05-08).",
              file=sys.stderr)
        return 2

    short_name = args.short_name or sanitise_short(args.model)

    legacy_dir = find_legacy_dump(args.legacy_temp)
    print(f"Reading prompts from: {legacy_dir}")

    configs = list_legacy_configs(legacy_dir)
    if args.pairs:
        bad = [p for p in args.pairs if p not in PAIRS]
        if bad:
            print(f"unknown pairs: {bad}", file=sys.stderr)
            return 2
        configs = [c for c in configs if f"{c[2]}-{c[3]}" in args.pairs]
    if args.templates:
        configs = [c for c in configs if c[0] in args.templates]
    if args.shot_counts:
        configs = [c for c in configs if c[1] in args.shot_counts]

    if not configs:
        print("no matching configs in dump.", file=sys.stderr)
        return 2

    print(f"OpenAI rerun: model={args.model}  short={short_name}  "
          f"T={args.temperature}  legacy_T={args.legacy_temp}  "
          f"batch={args.use_batch}  configs={len(configs)}")

    from afriscience_mt.api.factory import create_api_client
    api_client = create_api_client(
        provider="openai",
        model_name=args.model,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    out_temp_dir = EXP_ROOT / temp_setting(args.temperature) / "icl" / short_name
    state_file = ROOT / args.batch_state_file
    t0 = time.time()

    for idx, (tmpl, k, src, tgt, tsv) in enumerate(configs, 1):
        pair = f"{src}-{tgt}"
        strategy = f"template{tmpl}_{k}shot"
        out_dir = out_temp_dir / strategy / pair

        prompts = read_prompts(tsv)
        sys_const = system_is_constant(prompts)
        cache_path = cache_path_for(out_dir)
        cached = load_cache(cache_path)

        print(f"\n[{idx}/{len(configs)}] {strategy} {pair}  "
              f"({len(prompts)} prompts, {len(cached)} cached)  "
              f"-> {out_dir.relative_to(ROOT)}")

        ts = time.time()
        if args.use_batch:
            hyps = submit_batch(
                api_client, prompts,
                job_label=f"{short_name}/{pair}/{strategy}",
                poll_seconds=args.batch_poll_seconds,
                state_file=state_file,
            )
        else:
            hyps = [None] * len(prompts)
            for i, (s, u) in enumerate(tqdm(prompts, desc=strategy, leave=False)):
                if i in cached:
                    hyps[i] = cached[i]
                    continue
                try:
                    text = call_one(api_client, s, u)
                except Exception as e:
                    print(f"  ! row {i} failed ({type(e).__name__}: {e})", file=sys.stderr)
                    text = ""
                hyps[i] = text
                append_cache(cache_path, i, text)

        # score inline (cheap; metric module loads once per process)
        metrics = None
        try:
            from afriscience_mt.evaluation.metrics import compute_all_metrics
            srcs = load_test_sources(src)
            refs = load_test_refs(tgt)
            n = min(len(hyps), len(srcs), len(refs))
            m = compute_all_metrics(
                srcs[:n], hyps[:n], refs[:n],
                metrics=["chrf", "ssa_comet"],
            )
            metrics = {
                "chrf": m.get("chrf"), "ssa_comet": m.get("ssa_comet"),
                "num_samples": n,
            }
            print(f"   ChrF={metrics['chrf']:.2f}  COMET={100*metrics['ssa_comet']:.1f}  "
                  f"({time.time()-ts:.1f}s)")
        except Exception as e:
            print(f"   scoring failed ({type(e).__name__}: {e})")

        write_canonical(
            out_dir=out_dir, model=args.model, short_name=short_name,
            src=src, tgt=tgt, template=tmpl, n_shots=k,
            temp=args.temperature, hyps=hyps, system_constant=sys_const,
            metrics=metrics,
        )
        # cache survived? leave it for forensic; remove only on full success.
        if metrics is not None and cache_path.exists():
            try:
                cache_path.unlink()
            except OSError:
                pass

    print(f"\nTotal: {time.time() - t0:.1f}s")
    print("Aggregate to update summary.csv:")
    print("  python -m afriscience_mt aggregate --out experiments/summary.csv")
    print("Then rebuild paper tables:")
    print("  python scripts/build_paper_tables.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
