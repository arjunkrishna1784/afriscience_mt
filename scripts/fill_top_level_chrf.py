"""Compute top-level chrF (and BLEU) on closed-API predictions and patch
``metrics.chrf`` / ``metrics.bleu`` in each cell's results.json.

The closed-API rerun scripts (rerun_gpt4o_templated.py /
rerun_gemini_newer.py) only filled ``metrics.ssa_comet`` for new runs;
top-level chrF / BLEU were left as None. This script walks the closed
experiment tree and back-fills sentence-level chrF and BLEU using
sacrebleu, only updating cells where the metric is currently None.

Idempotent: skips cells that already have both metrics populated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "experiments" / "llm" / "closed"


def patch_one(results_path: Path, force: bool = False) -> str:
    preds_path = results_path.parent / "predictions.json"
    if not preds_path.exists():
        return "no_predictions"

    d = json.loads(results_path.read_text())
    metrics = d.setdefault("metrics", {})
    need_chrf = force or metrics.get("chrf") is None
    need_bleu = force or metrics.get("bleu") is None
    if not (need_chrf or need_bleu):
        return "skipped"

    samples = json.loads(preds_path.read_text()).get("samples", [])
    refs = [s.get("ref", "") or "" for s in samples]
    hyps = [s.get("hyp", "") or "" for s in samples]
    if not refs:
        return "no_samples"

    if need_chrf:
        metrics["chrf"] = float(sacrebleu.corpus_chrf(hyps, [refs]).score)
    if need_bleu:
        metrics["bleu"] = float(sacrebleu.corpus_bleu(hyps, [refs]).score)

    results_path.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    return "updated"


def walk(base: Path, force: bool) -> dict[str, int]:
    counts = {"updated": 0, "skipped": 0, "no_predictions": 0,
              "no_samples": 0}
    for results_path in sorted(base.rglob("results.json")):
        # Never touch backed-up / discarded predictions.
        if "discarded" in results_path.parts:
            continue
        status = patch_one(results_path, force=force)
        counts[status] += 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE,
                    help="Root dir to scan for results.json files.")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even when chrf/bleu are already set.")
    args = ap.parse_args()

    counts = walk(args.base, args.force)
    for k, v in counts.items():
        print(f"  {k:18s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
