"""Walk experiments/, fill in missing chrF + SSA-COMET on results.json.

Like rescore_missing_ssa.py but also fills chrF when null. Loads the
SSA-COMET model exactly once and skips dirs that already have both
metrics, so it's safe to re-run.

Usage (from repo root, dcs env):
    python scripts/rescore_missing_metrics.py --src experiments/llm/open
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


def missing_metrics(results_path: Path) -> list[str]:
    try:
        d = json.loads(results_path.read_text())
    except Exception:
        return []
    m = d.get("metrics") or {}
    needed = []
    if m.get("chrf") is None:
        needed.append("chrf")
    if m.get("ssa_comet") is None:
        needed.append("ssa_comet")
    return needed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="experiments",
                    help="root dir to walk (default: experiments)")
    args = ap.parse_args()

    targets: list[tuple[Path, list[str]]] = []
    for r in Path(args.src).rglob("results.json"):
        pred = r.parent / "predictions.json"
        if not pred.exists():
            continue
        need = missing_metrics(r)
        if need:
            targets.append((pred, need))

    print(f"need rescoring: {len(targets)} dirs")
    if not targets:
        return 0

    needs_ssa = any("ssa_comet" in n for _, n in targets)
    metrics = []
    if any("chrf" in n for _, n in targets):
        metrics.append("chrf")
    if needs_ssa:
        metrics.append("ssa_comet")
    print(f"computing: {metrics}")
    calc = MetricsCalculator(metrics=metrics)

    n_done = n_skip = 0
    pbar = tqdm(targets, desc="rescore", unit="dir")
    for pred_path, need in pbar:
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

        out = calc.compute(sources=srcs, hypotheses=hyps, references=refs)

        result_path = pred_path.parent / "results.json"
        r = load_result(result_path)
        rescored_log = []
        if "chrf" in need and out.get("chrf") is not None:
            r.metrics.chrf = out["chrf"]
            rescored_log.append(("chrf", out["chrf"]))
        if "ssa_comet" in need and out.get("ssa_comet") is not None:
            r.metrics.ssa_comet = out["ssa_comet"]
            rescored_log.append(("ssa_comet", out["ssa_comet"]))

        ts = datetime.utcnow().isoformat()
        for metric, value in rescored_log:
            r.metadata.setdefault("rescored", []).append({
                "metric": metric,
                "value": value,
                "timestamp": ts,
            })
        save_result(r, result_path)
        n_done += 1
        last_bits = " ".join(f"{m}={v:.3f}" for m, v in rescored_log)
        pbar.set_postfix(done=n_done, skip=n_skip, last=last_bits)

    print(f"\nrescored {n_done}, skipped {n_skip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
