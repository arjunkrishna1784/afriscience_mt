"""Walk experiments/, rescore SSA-COMET on results.json files missing it.

Loads the COMET model exactly once. Skips dirs that already have
metrics.ssa_comet so it's safe to re-run. Shows a tqdm progress bar.

Usage (from repo root, dcs env):
    python scripts/rescore_missing_ssa.py
    # or limit scope:
    python scripts/rescore_missing_ssa.py --src experiments/llm/closed
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from afriscience_mt.commands.evaluate import _test_text_column
from afriscience_mt.core.schema import load_predictions, load_result, save_result
from afriscience_mt.evaluation.metrics import MetricsCalculator


def needs_rescore(results_path: Path) -> bool:
    try:
        d = json.loads(results_path.read_text())
    except Exception:
        return False
    return (d.get("metrics") or {}).get("ssa_comet") is None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="experiments",
                    help="root dir to walk (default: experiments)")
    args = ap.parse_args()

    targets: list[Path] = []
    for r in Path(args.src).rglob("results.json"):
        pred = r.parent / "predictions.json"
        if pred.exists() and needs_rescore(r):
            targets.append(pred)

    print(f"need ssa_comet rescoring: {len(targets)} dirs")
    if not targets:
        return 0

    calc = MetricsCalculator(metrics=["ssa_comet"])

    n_done = n_skip = 0
    pbar = tqdm(targets, desc="rescore", unit="dir")
    for pred_path in pbar:
        try:
            pf = load_predictions(pred_path)
        except Exception as e:
            tqdm.write(f"  skip {pred_path}: {e}")
            n_skip += 1
            continue

        ext_src = _test_text_column(pf.source_lang)
        ext_ref = _test_text_column(pf.target_lang)

        srcs, refs, hyps = [], [], []
        for j, s in enumerate(pf.samples):
            hyp = s.get("hyp")
            if not hyp:
                continue
            ref = s.get("ref") or (ext_ref[j] if j < len(ext_ref) else None)
            src = s.get("src") or (ext_src[j] if j < len(ext_src) else "")
            if not ref:
                continue
            srcs.append(src)
            refs.append(ref)
            hyps.append(hyp)
        if not hyps:
            tqdm.write(f"  skip {pred_path.parent}: no usable samples")
            n_skip += 1
            continue

        value = calc.compute(hypotheses=hyps, references=refs, sources=srcs).get("ssa_comet")

        result_path = pred_path.parent / "results.json"
        r = load_result(result_path)
        r.metrics.ssa_comet = value
        r.metadata.setdefault("rescored", []).append({
            "metric": "ssa_comet",
            "value": value,
            "timestamp": datetime.utcnow().isoformat(),
        })
        save_result(r, result_path)
        n_done += 1
        pbar.set_postfix(done=n_done, skip=n_skip, last=f"{value:.4f}")

    print(f"\nrescored {n_done}, skipped {n_skip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
