"""Run GPT-4o (or any newer OpenAI model) on the missing eng->xxx
templated runs at T=0.

Audit (2026-05-08) showed the closed-API ICL coverage is asymmetric:
GPT-4o eng->xxx at T=0 only has the semantic-similarity strategies;
templates 1-5 were never run for that direction at T=0 (they exist at
T=0.6 only). Filling that single gap completes the closed_ablations
table on all three blocks: headline, retrieval ablation, and
temperature ablation.

This script targets exactly the missing combos. By default it runs
templates {1, 2, 3, 4, 5} at the per-template shot counts that match
DCS-MT (templates 1, 2, 4 are 0-shot only; template 3 has 5/10/20;
template 5 has 0/5/10/20). 50-shot is forbidden by project policy
(2026-05-08 cap).

Outputs land under
``experiments/llm/closed/t0/icl/<short_name>/template{N}_{K}shot/<pair>/``
in the canonical schema, so they appear in summary.csv after running

    python -m afriscience_mt aggregate --out experiments/summary.csv

Usage:

    export OPENAI_API_KEY=...

    # default: 6 eng->xxx pairs, all template/shot combos at T=0
    python scripts/rerun_gpt4o_templated.py --model gpt-4o

    # newer OpenAI model
    python scripts/rerun_gpt4o_templated.py --model gpt-5

    # batch API (50% cheaper, ~24h SLA, no per-minute caps)
    python scripts/rerun_gpt4o_templated.py --model gpt-4o --use-batch

    # subset for sanity check
    python scripts/rerun_gpt4o_templated.py --model gpt-4o \\
        --templates 5 --shot-counts 5 --pairs eng-amh

Notes:
  - --short-name controls the output dir and the model_short value in
    summary.csv. Defaults to a sanitised version of --model
    (e.g. gpt-5 -> gpt5; gpt-4o -> gpt4o).
  - Templates are read from templates/template{N}.txt at the repo
    root. Edit those files to change the prompt wording.
  - Few-shot exemplars are drawn from the training split with random
    selection (the same setup used for the Gemini t0 templated runs).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR  = ROOT / "data" / "processed_files" / "test"
TRAIN_DIR = ROOT / "data" / "processed_files" / "train"
TEMPLATE_DIR = ROOT / "templates"
EXP_ROOT  = ROOT / "experiments" / "llm" / "closed" / "t0" / "icl"

# Per-template shot counts that match what DCS-MT ran (cap 20 by project policy).
TEMPLATE_SHOT_COUNTS = {
    1: [0],
    2: [0],
    3: [0, 5, 10, 20],
    4: [0],
    5: [0, 5, 10, 20],
}

PAIRS_E2X = ["eng-amh", "eng-hau", "eng-lug", "eng-nso", "eng-yor", "eng-zul"]

LANG_NAMES = {
    "eng": "English",
    "amh": "Amharic", "hau": "Hausa", "lug": "Luganda",
    "nso": "Northern Sotho", "yor": "Yoruba", "zul": "isiZulu",
}

RNG_SEED = 42  # fixed-seed random selection for reproducible exemplars


def sanitise_short(name: str) -> str:
    return name.replace("-", "").replace(".", "").replace("_", "").lower()


def load_template(n: int) -> str:
    path = TEMPLATE_DIR / f"template{n}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Template not found: {path}. The repo expects templates/"
            f"template{{1..5}}.txt vendored from DCS-MT."
        )
    return path.read_text(encoding="utf-8")


def load_split(split_dir: Path, lang: str) -> list[dict]:
    with open(split_dir / f"{lang}.tsv") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def build_pair_data(split_dir: Path, src: str, tgt: str) -> list[dict]:
    s = {(r["paper"], r["segment"]): r["text"] for r in load_split(split_dir, src)}
    t = {(r["paper"], r["segment"]): r["text"] for r in load_split(split_dir, tgt)}
    return [{"src": s[k], "ref": t[k]} for k in s if k in t]


def render_examples(examples: list[dict], src_lang: str, tgt_lang: str) -> str:
    if not examples:
        return ""
    src_name = LANG_NAMES[src_lang]
    tgt_name = LANG_NAMES[tgt_lang]
    lines = [f"Examples:"]
    for ex in examples:
        lines.append(f"{src_name}: {ex['src']}")
        lines.append(f"{tgt_name}: {ex['ref']}")
    return "\n".join(lines) + "\n"


def render_prompt(template_text: str, *, src_lang: str, tgt_lang: str,
                  src_text: str, examples: list[dict],
                  glossary: str = "") -> str:
    return template_text.format(
        src_lang=LANG_NAMES[src_lang],
        tgt_lang=LANG_NAMES[tgt_lang],
        src_text=src_text,
        examples=render_examples(examples, src_lang, tgt_lang),
        glossary=glossary,
    )


def build_prompts(samples: list[dict], train_pool: list[dict],
                  template_text: str, src: str, tgt: str,
                  n_shots: int) -> list[str]:
    rng = random.Random(RNG_SEED)
    prompts: list[str] = []
    for s in samples:
        examples = []
        if n_shots > 0 and train_pool:
            examples = rng.sample(train_pool, k=min(n_shots, len(train_pool)))
        prompts.append(render_prompt(
            template_text,
            src_lang=src, tgt_lang=tgt,
            src_text=s["src"], examples=examples,
        ))
    return prompts


def write_pair(out_dir: Path, *, model: str, short_name: str,
               src: str, tgt: str, prompt_strategy: str,
               template: int, n_shots: int, temp: float,
               samples: list[dict], metrics: dict | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pair = f"{src}-{tgt}"
    pred_path = out_dir / "predictions.json"
    res_path  = out_dir / "results.json"

    pred_path.write_text(json.dumps({
        "experiment_id": f"icl_{short_name}_{pair}_{prompt_strategy}",
        "source_lang": src, "target_lang": tgt,
        "samples": [{"src": s["src"], "ref": s["ref"], "hyp": s["hyp"]} for s in samples],
        "schema_version": "1.0",
    }, indent=2, ensure_ascii=False))

    res = {
        "schema_version": "1.0",
        "experiment_id": f"icl_{short_name}_{pair}_{prompt_strategy}",
        "experiment_type": "llm_icl",
        "timestamp": datetime.utcnow().isoformat(),
        "source_lang": src, "target_lang": tgt, "lang_pair": pair,
        "split": "test", "dataset": "dcs",
        "model": {"name": model, "type": "llm", "short_name": short_name},
        "config": {
            "n_shots": n_shots,
            "template": template,
            "selection_method": "random",
            "temperature": temp,
            "max_tokens": 1024,
        },
        "metrics": metrics or {"num_samples": len(samples)},
        "metadata": {
            "source": "rerun_gpt4o_templated",
            "prompt_strategy": prompt_strategy,
            "is_ablation": True,  # templated ablation runs
            "comet_metric": "ssa-comet",
        },
    }
    res_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))


def run_online(api_client, prompts: list[str]) -> list[str]:
    return api_client.generate_batch(prompts=prompts, show_progress=True)


def run_batch(api_client, prompts: list[str], *, job_label: str,
              poll_seconds: int, state_file: Path) -> list[str]:
    state = {}
    if state_file.exists():
        state = json.loads(state_file.read_text())

    job_id = state.get(job_label)
    if not job_id:
        print(f"    submitting batch ({len(prompts)} prompts)...")
        job_id = api_client.submit_batch_job(prompts=prompts, job_name=job_label)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o",
                    help="OpenAI model name (e.g. gpt-4o, gpt-5).")
    ap.add_argument("--short-name", default=None,
                    help="model_short used in summary.csv. "
                         "Defaults to sanitised --model.")
    ap.add_argument("--pairs", nargs="+", default=PAIRS_E2X,
                    help="eng->xxx pairs only (default: all 6).")
    ap.add_argument("--templates", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                    help="Template numbers to run (default: 1 2 3 4 5).")
    ap.add_argument("--shot-counts", type=int, nargs="+", default=None,
                    help="Override per-template shot counts. Default: "
                         "templates 1,2,4=[0]; template 3=[0,5,10,20]; "
                         "template 5=[0,5,10,20]. 50-shot is forbidden.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--api-key", default=None,
                    help="OpenAI API key (default: $OPENAI_API_KEY).")
    ap.add_argument("--use-batch", action="store_true",
                    help="Submit prompts via the OpenAI Batch API "
                         "(50% cheaper, ~24h SLA, no per-minute caps).")
    ap.add_argument("--batch-poll-seconds", type=int, default=120)
    ap.add_argument("--batch-state-file",
                    default="experiments/_openai_batch_state.json")
    ap.add_argument("--skip-scoring", action="store_true")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("error: pass --api-key or set OPENAI_API_KEY", file=sys.stderr)
        return 2

    short_name = args.short_name or sanitise_short(args.model)

    bad_pairs = [p for p in args.pairs if p not in PAIRS_E2X]
    if bad_pairs:
        print(f"error: pairs must be eng->xxx (got {bad_pairs})", file=sys.stderr)
        return 2

    bad_templates = [t for t in args.templates if t not in (1, 2, 3, 4, 5)]
    if bad_templates:
        print(f"error: templates must be in 1..5 (got {bad_templates})", file=sys.stderr)
        return 2

    if args.shot_counts and any(k > 20 for k in args.shot_counts):
        print("error: shot counts > 20 are not allowed (50-shot dropped 2026-05-08).",
              file=sys.stderr)
        return 2

    from afriscience_mt.api.factory import create_api_client
    api_client = create_api_client(
        provider="openai",
        model_name=args.model,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print(f"Rerunning OpenAI: model={args.model}  short={short_name}  "
          f"templates={args.templates}  T={args.temperature}  "
          f"batch={args.use_batch}")

    state_file = ROOT / args.batch_state_file
    t0 = time.time()

    for pair in args.pairs:
        src, tgt = pair.split("-")
        print(f"\n=== {pair} ===")

        test_samples  = build_pair_data(TEST_DIR, src, tgt)
        train_pool    = build_pair_data(TRAIN_DIR, src, tgt)

        for template in args.templates:
            shot_counts = args.shot_counts or TEMPLATE_SHOT_COUNTS.get(template, [0])
            for n_shots in shot_counts:
                if n_shots > 20:
                    continue
                strategy = f"template{template}_{n_shots}shot"
                out_dir = EXP_ROOT / short_name / strategy / pair
                template_text = load_template(template)
                prompts = build_prompts(test_samples, train_pool,
                                         template_text, src, tgt, n_shots)

                ts = time.time()
                print(f"  [{strategy}] -> {out_dir.relative_to(ROOT)}  ({len(prompts)} prompts)")

                if args.use_batch:
                    hyps = run_batch(api_client, prompts,
                                      job_label=f"{short_name}/{pair}/{strategy}",
                                      poll_seconds=args.batch_poll_seconds,
                                      state_file=state_file)
                else:
                    hyps = run_online(api_client, prompts)

                samples = [
                    {"src": s["src"], "ref": s["ref"], "hyp": h}
                    for s, h in zip(test_samples, hyps)
                ]

                metrics = None
                if not args.skip_scoring:
                    try:
                        from afriscience_mt.evaluation.metrics import compute_all_metrics
                        m = compute_all_metrics(
                            [s["src"] for s in samples],
                            [s["hyp"] for s in samples],
                            [s["ref"] for s in samples],
                            metrics=["chrf", "ssa_comet"],
                        )
                        metrics = {
                            "chrf": m.get("chrf"),
                            "ssa_comet": m.get("ssa_comet"),
                            "num_samples": len(samples),
                        }
                        print(f"    ChrF={metrics['chrf']:.2f}  "
                              f"COMET={100*metrics['ssa_comet']:.1f}  "
                              f"({time.time()-ts:.1f}s)")
                    except Exception as e:
                        print(f"    scoring failed ({type(e).__name__}: {e}); "
                              "writing predictions only")

                write_pair(
                    out_dir, model=args.model, short_name=short_name,
                    src=src, tgt=tgt,
                    prompt_strategy=strategy,
                    template=template, n_shots=n_shots,
                    temp=args.temperature,
                    samples=samples, metrics=metrics,
                )

    print(f"\nTotal: {time.time() - t0:.1f}s")
    print("Aggregate to update summary.csv:")
    print("  python -m afriscience_mt aggregate --out experiments/summary.csv")
    print("Then rebuild paper tables:")
    print("  python scripts/build_paper_tables.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
