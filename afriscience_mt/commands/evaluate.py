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
    args = p.parse_args(argv)

    from afriscience_mt.evaluation.metrics import compute_bleu, compute_chrf, MetricsCalculator
    calc = (MetricsCalculator(metrics=["ssa_comet"])
            if args.metric == "ssa_comet" else None)

    n_done = n_skip = 0
    for pred_path in Path(args.src).rglob("predictions.json"):
        try:
            pf = load_predictions(pred_path)
        except Exception as e:
            print(f"  skip {pred_path}: {e}")
            n_skip += 1
            continue

        # Look up src/ref from the canonical test split when not stored in the
        # sample (the default — predictions.json normally carries only `hyp`).
        ext_src = _test_text_column(pf.source_lang)
        ext_ref = _test_text_column(pf.target_lang)

        srcs, refs, hyps = [], [], []
        for i, s in enumerate(pf.samples):
            hyp = s.get("hyp")
            if not hyp:
                continue
            ref = s.get("ref") or (ext_ref[i] if i < len(ext_ref) else None)
            src = s.get("src") or (ext_src[i] if i < len(ext_src) else "")
            if not ref:
                continue
            srcs.append(src)
            refs.append(ref)
            hyps.append(hyp)
        if not hyps:
            print(f"  skip {pred_path}: no usable samples (no ref in predictions and no test split for {pf.source_lang}/{pf.target_lang})")
            n_skip += 1
            continue

        if args.metric == "bleu":
            value = compute_bleu(hyps, refs)["bleu"]
        elif args.metric == "chrf":
            value = compute_chrf(hyps, refs)["chrf"]
        else:
            value = calc.compute(hypotheses=hyps, references=refs,
                                 sources=srcs).get("ssa_comet")

        # Update sibling results.json
        result_path = pred_path.parent / "results.json"
        if not result_path.exists():
            print(f"  warn no results.json at {result_path.parent}")
        else:
            r = load_result(result_path)
            label = args.metric_name or args.metric
            setattr(r.metrics, label if label in {"bleu", "chrf", "ssa_comet"} else "extra",
                    value)
            r.metadata.setdefault("rescored", []).append({
                "metric": label, "value": value,
                "timestamp": datetime.utcnow().isoformat(),
            })
            save_result(r, result_path)
        n_done += 1
        print(f"  {pred_path.parent}: {args.metric}={value:.4f}")

    print(f"\nrescored {n_done}, skipped {n_skip}")
    return 0


def _eval_zero_shot(argv: list[str]) -> int:
    print("afriscience-mt evaluate --mode zero_shot: not yet refactored", file=sys.stderr)
    print("for now: use afriscience-evaluate or scripts/eval_*", file=sys.stderr)
    return 2


def _eval_finetuned(argv: list[str]) -> int:
    print("afriscience-mt evaluate --mode finetuned: not yet refactored", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
