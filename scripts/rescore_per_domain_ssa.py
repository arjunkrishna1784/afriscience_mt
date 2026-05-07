"""Rescore SSA-COMET *per-domain* on selected experiment dirs and write the
breakdown into ``results.json.metrics.per_domain`` (so the spider builders
can read SSA-COMET per domain without re-loading the COMET model).

Walks the directories you give it, looks up each test sentence's
``category`` from ``data/processed_files/test/<src>.tsv``, groups
hypotheses + references by category, and runs the existing
``MetricsCalculator`` once per group. Skips dirs that already have
``per_domain`` with ``ssa_comet`` populated.

Usage (from repo root, dcs env with torch + comet):
    python scripts/rescore_per_domain_ssa.py \\
        --src experiments/seq2seq/finetune/nllb_200_1_3b/dcs \\
        --src experiments/llm/open/finetune/gemma2_9b_it/r64 \\
        --src experiments/llm/open/finetune/tiny_aya_earth/r64 \\
        --src experiments/llm/closed/t06/icl/gpt4o/template3_5shot \\
        --src experiments/llm/closed/t0/icl/gpt4o/template3_5shot

After running, regenerate the spiders with the SSA-COMET branch enabled:
    python scripts/redraw_paper_plots.py --spider-metric ssa_comet
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "data" / "processed_files" / "test"


def load_test(lang: str) -> list[dict]:
    with open(TEST_DIR / f"{lang}.tsv") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def has_per_domain_ssa(results_path: Path) -> bool:
    try:
        d = json.loads(results_path.read_text())
    except Exception:
        return False
    pd = (d.get("metrics") or {}).get("per_domain") or {}
    if not pd:
        return False
    first = next(iter(pd.values()), {})
    return "ssa_comet" in first


def rescore_one(pred_path: Path, calc) -> dict[str, dict] | None:
    payload = json.loads(pred_path.read_text())
    src_lang = payload["source_lang"]
    tgt_lang = payload["target_lang"]
    samples = payload.get("samples") or []

    src_tsv = load_test(src_lang)
    tgt_tsv = load_test(tgt_lang)
    if len(samples) != len(src_tsv):
        return None

    by_cat = defaultdict(lambda: {"src": [], "ref": [], "hyp": []})
    for i, s in enumerate(samples):
        hyp = s.get("hyp") or ""
        if not hyp:
            continue
        cat = src_tsv[i]["category"]
        by_cat[cat]["src"].append(src_tsv[i]["text"])
        by_cat[cat]["ref"].append(tgt_tsv[i]["text"])
        by_cat[cat]["hyp"].append(hyp)

    out = {}
    for cat, bag in by_cat.items():
        v = calc.compute(hypotheses=bag["hyp"],
                         references=bag["ref"],
                         sources=bag["src"]).get("ssa_comet")
        out[cat] = {"n": len(bag["hyp"]), "ssa_comet": v}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True,
                    help="dir to walk for predictions.json (repeatable)")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if per_domain.ssa_comet already exists")
    args = ap.parse_args()

    from afriscience_mt.evaluation.metrics import MetricsCalculator
    calc = MetricsCalculator(metrics=["ssa_comet"])

    targets: list[Path] = []
    for src in args.src:
        for r in Path(src).rglob("results.json"):
            pred = r.parent / "predictions.json"
            if not pred.exists():
                continue
            if not args.force and has_per_domain_ssa(r):
                continue
            targets.append(pred)

    print(f"per-domain SSA-COMET rescore queued: {len(targets)} dirs")
    if not targets:
        return 0

    try:
        from tqdm import tqdm
        iterator = tqdm(targets, desc="per-domain", unit="dir")
    except ImportError:
        iterator = targets

    for pred_path in iterator:
        try:
            per_dom = rescore_one(pred_path, calc)
        except Exception as e:
            print(f"  skip {pred_path.parent}: {e}")
            continue
        if not per_dom:
            print(f"  skip {pred_path.parent}: hyp/test row count mismatch")
            continue
        results_path = pred_path.parent / "results.json"
        d = json.loads(results_path.read_text())
        d.setdefault("metrics", {}).setdefault("per_domain", {})
        for cat, stats in per_dom.items():
            d["metrics"]["per_domain"].setdefault(cat, {})
            d["metrics"]["per_domain"][cat].update(stats)
        d.setdefault("metadata", {}).setdefault("rescored", []).append({
            "metric": "ssa_comet_per_domain",
            "n_domains": len(per_dom),
            "timestamp": datetime.utcnow().isoformat(),
        })
        results_path.write_text(json.dumps(d, indent=2, ensure_ascii=False))

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
