"""Rerun Gemini on a newer model (e.g. gemini-2.0-flash, gemini-2.5-pro) on
all 12 language pairs at zero-shot and 5-shot ICL.

This reuses the same prompt template ("basic") and the same retrieval
strategy that the original Gemini-1.5-Flash runs used, so the new
predictions slot directly alongside the existing ones for an
apples-to-apples model-version comparison.

Outputs are written under
``experiments/llm/closed/<temp>/<phase>/<short_name>/<pair>/``
in the canonical (predictions.json + results.json) schema, so they
land in ``summary.csv`` automatically when you run

    python -m afriscience_mt aggregate --out experiments/summary.csv

Usage (from repo root):

    export GOOGLE_API_KEY=...
    python scripts/rerun_gemini_newer.py --model gemini-2.0-flash
    python scripts/rerun_gemini_newer.py --model gemini-2.5-pro --pairs eng-amh eng-hau
    python scripts/rerun_gemini_newer.py --model gemini-2.5-flash --shot-counts 0 5 10

Notes:
  - ``--short-name`` controls the output directory name and the
    ``model_short`` value that lands in summary.csv. Defaults to a
    sanitised version of ``--model`` (e.g. ``gemini-2.5-pro`` ->
    ``gemini_2_5_pro``). Pick a stable name -- it is what the table
    builder will reference.
  - Random retrieval is the default; pass ``--retrieval semantic`` to
    use OpenAI-embedding semantic similarity (requires OPENAI_API_KEY).
  - We score the predictions on BLEU/ChrF/SSA-COMET as we go, so the
    written results.json is ready for table generation immediately.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR  = ROOT / "data" / "processed_files" / "test"
TRAIN_DIR = ROOT / "data" / "processed_files" / "train"
EXP_ROOT  = ROOT / "experiments" / "llm" / "closed"

PAIRS = [
    "eng-amh", "eng-hau", "eng-lug", "eng-nso", "eng-yor", "eng-zul",
    "amh-eng", "hau-eng", "lug-eng", "nso-eng", "yor-eng", "zul-eng",
]


def sanitise_short(name: str) -> str:
    return name.replace("-", "_").replace(".", "_").lower()


def temp_setting(temp: float) -> str:
    return "t0" if abs(temp) < 1e-9 else f"t{int(round(temp * 10))}".replace("0", "")  # t0 or t06


def write_pair(out_dir: Path, *, model: str, short_name: str,
               src: str, tgt: str, prompt_strategy: str,
               n_shots: int, temp: float, samples: list[dict],
               metrics: dict | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pair = f"{src}-{tgt}"
    pred_path = out_dir / "predictions.json"
    res_path  = out_dir / "results.json"

    pred_path.write_text(json.dumps({
        "experiment_id": f"icl_{short_name}_{pair}_{n_shots}shot",
        "source_lang": src, "target_lang": tgt,
        "samples": [{"src": s["src"], "ref": s["ref"], "hyp": s["hyp"]} for s in samples],
        "schema_version": "1.0",
    }, indent=2, ensure_ascii=False))

    etype = "llm_zero_shot" if n_shots == 0 else "llm_icl"
    res = {
        "schema_version": "1.0",
        "experiment_id": f"icl_{short_name}_{pair}_{n_shots}shot",
        "experiment_type": etype,
        "timestamp": datetime.utcnow().isoformat(),
        "source_lang": src, "target_lang": tgt, "lang_pair": pair,
        "split": "test", "dataset": "dcs",
        "model": {"name": model, "type": "llm", "short_name": short_name},
        "config": {
            "n_shots": n_shots,
            "temperature": temp,
            "max_output_tokens": 1024,
        },
        "metrics": metrics or {"num_samples": len(samples)},
        "metadata": {
            "source": "rerun_gemini_newer",
            "prompt_strategy": prompt_strategy,
            "is_ablation": False,
            "comet_metric": "ssa-comet",
        },
    }
    res_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))


def load_split(split_dir: Path, lang: str) -> list[dict]:
    with open(split_dir / f"{lang}.tsv") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def build_pair_data(split_dir: Path, src: str, tgt: str) -> list[dict]:
    s = {(r["paper"], r["segment"]): r["text"] for r in load_split(split_dir, src)}
    t = {(r["paper"], r["segment"]): r["text"] for r in load_split(split_dir, tgt)}
    return [{"src": s[k], "ref": t[k]} for k in s if k in t]


def build_prompts_for_pair(runner, sources: list[str], src: str, tgt: str,
                            n_shots: int, retrieval: str) -> list[str]:
    """Materialise the ICL prompts that ICLRunner would submit one-by-one,
    so we can hand them to the Batch API instead."""
    from afriscience_mt.inference.prompt_builder import build_translation_prompt
    selection = "random" if retrieval == "random" else "semantic"
    prompts: list[str] = []
    for i, src_text in enumerate(sources):
        examples = []
        if n_shots > 0:
            examples = runner.example_selector.select(
                source_text=src_text,
                source_lang=src, target_lang=tgt,
                n_examples=n_shots,
                method=selection,
            ) if hasattr(runner, "example_selector") and runner.example_selector else []
            # Fallback: directly call translate_batch's machinery via a helper
            if not examples and runner.train_dataset is not None:
                from afriscience_mt.inference.example_selector import (
                    create_example_selector,
                )
                sel = create_example_selector(runner.train_dataset, method=selection)
                examples = sel.select(source_text=src_text,
                                      source_lang=src, target_lang=tgt,
                                      n_examples=n_shots)
        prompts.append(build_translation_prompt(
            source_text=src_text, source_lang=src, target_lang=tgt,
            template="basic", examples=examples,
        ))
    return prompts


def run_batch_mode(api_client, runner, *, src: str, tgt: str,
                   n_shots: int, retrieval: str, sources: list[str],
                   short_name: str, model: str, temp_setting_str: str,
                   poll_seconds: int, state_file: Path) -> list[str]:
    """Submit one batch per (pair, shot-count) and poll until done.
    Returns the list of decoded responses in source order."""
    job_name_key = f"{short_name}/{temp_setting_str}/{src}-{tgt}/{n_shots}shot/{retrieval}"
    state = {}
    if state_file.exists():
        state = json.loads(state_file.read_text())

    prompts = build_prompts_for_pair(runner, sources, src, tgt, n_shots, retrieval)

    job_name = state.get(job_name_key)
    if not job_name:
        print(f"    submitting batch ({len(prompts)} prompts)...")
        job_name = api_client.submit_batch_job(
            prompts=prompts,
            job_name=f"afriscience-{short_name}-{src}{tgt}-{n_shots}shot",
        )
        state[job_name_key] = job_name
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(state, indent=2))
        print(f"    submitted: {job_name}")
    else:
        print(f"    resuming batch: {job_name}")

    while True:
        out = api_client.retrieve_batch_results(job_name)
        if out is not None:
            print(f"    batch complete ({len(out)} responses)")
            # remove from state file once consumed
            state.pop(job_name_key, None)
            state_file.write_text(json.dumps(state, indent=2))
            return out
        print(f"    pending; sleeping {poll_seconds}s...")
        time.sleep(poll_seconds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Gemini model name (e.g. gemini-2.0-flash, gemini-2.5-pro)")
    ap.add_argument("--short-name", default=None,
                    help="model_short used in summary.csv. Defaults to sanitised --model.")
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="Subset of language pairs (default: all 12).")
    ap.add_argument("--shot-counts", type=int, nargs="+", default=[0, 5],
                    help="Shot counts to run (default: 0 5).")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=1024)
    ap.add_argument("--retrieval", choices=["random", "semantic"], default="random",
                    help="Few-shot exemplar retrieval (default: random).")
    ap.add_argument("--api-key", default=None,
                    help="Google API key (default: $GOOGLE_API_KEY).")
    ap.add_argument("--skip-scoring", action="store_true",
                    help="Write predictions.json without metrics; rescore later "
                         "with `afriscience-mt evaluate --mode rescore`.")
    ap.add_argument("--use-batch", action="store_true",
                    help="Submit prompts to the Gemini Batch API (50% cheaper, "
                         "no per-minute rate limits, ~24h SLA). Recommended for "
                         "Pro-tier models with low RPM.")
    ap.add_argument("--batch-poll-seconds", type=int, default=120,
                    help="Polling interval for --use-batch (default 120s).")
    ap.add_argument("--batch-state-file", default="experiments/_gemini_batch_state.json",
                    help="JSON file used to record submitted batch job names "
                         "so a crash can resume. (Only used with --use-batch.)")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("error: pass --api-key or set GOOGLE_API_KEY", file=sys.stderr)
        return 2

    short_name = args.short_name or sanitise_short(args.model)
    pairs = args.pairs or PAIRS
    bad = [p for p in pairs if p not in PAIRS]
    if bad:
        print(f"unknown pairs: {bad}", file=sys.stderr)
        return 2

    # Lazy import — keeps the script importable for `--help` even without google-genai.
    from afriscience_mt.inference.icl_runner import ICLRunner
    from afriscience_mt.api.factory import create_api_client
    from afriscience_mt.data.dataset import AfriScienceDataset

    print(f"Reruning Gemini: model={args.model}  short={short_name}  "
          f"shots={args.shot_counts}  retrieval={args.retrieval}  T={args.temperature}")

    api_client = create_api_client(
        provider="google",
        model_name=args.model,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_output_tokens,
    )

    temp_dir = EXP_ROOT / temp_setting(args.temperature)

    for pair in pairs:
        src, tgt = pair.split("-")
        ts = time.time()
        print(f"\n=== {pair} ===")

        test_ds = AfriScienceDataset(
            data_path=TEST_DIR, source_lang=src, target_lang=tgt, split="test",
        )
        train_ds = AfriScienceDataset(
            data_path=TRAIN_DIR, source_lang=src, target_lang=tgt, split="train",
        )

        runner = ICLRunner(api_client=api_client, train_dataset=train_ds)

        for n_shots in args.shot_counts:
            phase = "zero_shot" if n_shots == 0 else "icl"
            strategy = (
                "" if n_shots == 0
                else (f"icl_{n_shots}shot" if args.retrieval == "random"
                      else f"icl_{n_shots}shot_semantic")
            )
            out_dir = (temp_dir / phase / short_name
                       / (strategy or "") / pair) if n_shots else \
                      (temp_dir / phase / short_name / pair)

            print(f"  [{n_shots}-shot] -> {out_dir.relative_to(ROOT)}")

            if args.use_batch:
                hyps = run_batch_mode(
                    api_client, runner,
                    src=src, tgt=tgt,
                    n_shots=n_shots, retrieval=args.retrieval,
                    sources=test_ds.get_sources(),
                    short_name=short_name, model=args.model,
                    temp_setting_str=temp_setting(args.temperature),
                    poll_seconds=args.batch_poll_seconds,
                    state_file=ROOT / args.batch_state_file,
                )
                samples = [
                    {"src": s, "ref": r, "hyp": h}
                    for s, r, h in zip(test_ds.get_sources(), test_ds.get_targets(), hyps)
                ]
            else:
                translations = runner.translate_batch(
                    texts=test_ds.get_sources(),
                    source_lang=src, target_lang=tgt,
                    n_shots=n_shots, template="basic",
                    use_glossary=False,
                    selection_method=("random" if args.retrieval == "random" else "semantic"),
                )
                samples = [
                    {"src": s, "ref": r, "hyp": t["translation"]}
                    for s, r, t in zip(test_ds.get_sources(), test_ds.get_targets(), translations)
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
                    print(f"    ChrF={metrics['chrf']:.2f}  COMET={100*metrics['ssa_comet']:.1f}")
                except Exception as e:
                    print(f"    scoring failed ({type(e).__name__}: {e}); writing predictions only")

            write_pair(
                out_dir, model=args.model, short_name=short_name,
                src=src, tgt=tgt,
                prompt_strategy=strategy or "",
                n_shots=n_shots, temp=args.temperature,
                samples=samples, metrics=metrics,
            )
        print(f"  pair done in {time.time() - ts:.1f}s")

    print("\nAll pairs done.")
    print("Aggregate to update summary.csv:")
    print("  python -m afriscience_mt aggregate --out experiments/summary.csv")
    print("Then rebuild paper tables:")
    print("  python scripts/build_paper_tables.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
