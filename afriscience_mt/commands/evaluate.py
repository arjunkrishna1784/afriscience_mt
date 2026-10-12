"""`afriscience-mt evaluate` — run / re-run evaluation.

Two distinct flows:

1) From-scratch evaluation: load model, translate test set, score, write result.
   This is what you'd use after a new training run if you didn't auto-evaluate.

2) Re-score existing predictions: useful when you change metric versions
   (e.g. AfriCOMET → SSA-COMET) without re-running inference.

Modes:
  --mode zero_shot     evaluate a base seq2seq model (NLLB, M2M100)
  --mode finetuned     evaluate a fine-tuned model (local checkpoint)
  --mode rescore       re-score predictions.json files in-place
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from afriscience_mt.core.schema import (
    PredictionFile,
    load_predictions,
    load_result,
    save_result,
)


_TEST_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed_files" / "test"


@lru_cache(maxsize=32)
def _test_text_column(lang: str) -> tuple[str, ...]:
    """Return the `text` column of data/processed_files/test/<lang>.tsv (cached)."""
    tsv = _TEST_DIR / f"{lang}.tsv"
    if not tsv.exists():
        return ()
    with open(tsv) as f:
        return tuple(r["text"] for r in csv.DictReader(f, delimiter="\t"))


def _metric_present(r, label: str) -> bool:
    """True iff results.json already has a value for this metric label."""
    if label in {"bleu", "chrf", "ssa_comet"}:
        return getattr(r.metrics, label, None) is not None
    return label in r.metrics.extra


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="afriscience-mt evaluate")
    p.add_argument("--mode", required=True,
                   choices=["zero_shot", "finetuned", "rescore"])
    args, rest = p.parse_known_args(argv)

    if args.mode == "rescore":
        return _rescore(rest)
    if args.mode == "zero_shot":
        return _eval_zero_shot(rest)
    if args.mode == "finetuned":
        return _eval_finetuned(rest)
    return 1


def _rescore(argv: list[str]) -> int:
    """Re-compute metrics for predictions.json files; update sibling results.json."""
    p = argparse.ArgumentParser(prog="afriscience-mt evaluate --mode rescore")
    p.add_argument("--src", required=True,
                   help="dir to walk for predictions.json files")
    p.add_argument("--metric", default="ssa_comet",
                   choices=["bleu", "chrf", "ssa_comet"])
    p.add_argument("--metric-name", default=None,
                   help="override metric label (default: same as --metric)")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip files whose results.json already has this metric set")
    args = p.parse_args(argv)
    label = args.metric_name or args.metric

    pred_paths = sorted(Path(args.src).rglob("predictions.json"))
    if not pred_paths:
        print(f"No predictions.json found under {args.src}", file=sys.stderr)
        return 1

    # Preflight the entire batch before loading a metric model or writing results.
    prepared = []
    n_skip = 0
    for pred_path in pred_paths:
        try:
            result_path = pred_path.parent / "results.json"
            result = load_result(result_path)
            predictions = load_predictions(pred_path)
            if (predictions.experiment_id, predictions.source_lang, predictions.target_lang) != (
                result.experiment_id, result.source_lang, result.target_lang
            ):
                raise ValueError("Prediction/result experiment or language mismatch")
            sources, hypotheses, references = _rescore_inputs(predictions, result)
            if args.skip_existing and _metric_present(result, label):
                n_skip += 1
                continue
            prepared.append((result_path, result, sources, hypotheses, references))
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            print(f"Invalid artifact {pred_path}: {exc}", file=sys.stderr)
            return 1

    if not prepared:
        print(f"rescored 0, skipped {n_skip}")
        return 0

    from afriscience_mt.evaluation.metrics import compute_bleu, compute_chrf, MetricsCalculator
    calc = MetricsCalculator(metrics=["ssa_comet"]) if args.metric == "ssa_comet" else None
    n_done = 0
    for result_path, result, sources, hypotheses, references in prepared:
        try:
            if args.metric == "bleu":
                value = compute_bleu(hypotheses, references)["bleu"]
            elif args.metric == "chrf":
                value = compute_chrf(hypotheses, references)["chrf"]
            else:
                value = calc.compute(sources, hypotheses, references)["ssa_comet"]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid metric value: {value!r}")
            if label in {"bleu", "chrf", "ssa_comet"}:
                setattr(result.metrics, label, value)
            else:
                result.metrics.extra[label] = value
            result.metrics.num_samples = len(hypotheses)
            result.metadata.setdefault("rescored", []).append({
                "metric": label, "value": value,
                "timestamp": datetime.utcnow().isoformat(),
            })
            save_result(result, result_path)
        except Exception as exc:
            print(f"Rescore failed for {result_path}: {exc}", file=sys.stderr)
            return 1
        n_done += 1
        print(f"{result_path.parent}: {args.metric}={value:.4f}")
    print(f"rescored {n_done}, skipped {n_skip}")
    return 0


def _rescore_inputs(predictions, result):
    """Resolve legacy positional inputs only against an exact-length DCS test set.

    Equal lengths cannot establish order/provenance for hypothesis-only files.
    Stable sample IDs require a separate schema migration.
    """
    from afriscience_mt.core.validation import validate_parallel_inputs

    samples = predictions.samples
    if not isinstance(samples, list) or not samples:
        raise ValueError("samples must be a nonempty list")
    if any(not isinstance(sample, dict) for sample in samples):
        raise ValueError("Each sample must be an object")
    needs_src = any("src" not in sample for sample in samples)
    needs_ref = any("ref" not in sample for sample in samples)
    ext_src = ext_ref = ()
    if needs_src or needs_ref:
        if result.dataset != "dcs" or result.split != "test":
            raise ValueError("Legacy positional fallback requires dataset=dcs and split=test")
        if needs_src:
            ext_src = _test_text_column(predictions.source_lang)
            if len(ext_src) != len(samples):
                raise ValueError("Source test split length does not match predictions")
        if needs_ref:
            ext_ref = _test_text_column(predictions.target_lang)
            if len(ext_ref) != len(samples):
                raise ValueError("Reference test split length does not match predictions")
    sources = [s["src"] if "src" in s else ext_src[i] for i, s in enumerate(samples)]
    references = [s["ref"] if "ref" in s else ext_ref[i] for i, s in enumerate(samples)]
    hypotheses = [s["hyp"] for s in samples]
    validate_parallel_inputs(sources=sources, hypotheses=hypotheses, references=references)
    if any(not ref.strip() for ref in references):
        raise ValueError("Reference text must not be empty")
    return sources, hypotheses, references


def _eval_zero_shot(argv: list[str]) -> int:
    print("afriscience-mt evaluate --mode zero_shot: not yet refactored", file=sys.stderr)
    print("for now: use afriscience-evaluate or scripts/eval_*", file=sys.stderr)
    return 2


def _eval_finetuned(argv: list[str]) -> int:
    print("afriscience-mt evaluate --mode finetuned: not yet refactored", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
