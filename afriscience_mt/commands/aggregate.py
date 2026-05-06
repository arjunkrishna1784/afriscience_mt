"""Walk canonical results and emit one tidy CSV/JSON across all experiments.

The output is intended to be the single source of truth for downstream
``afriscience-mt table`` and ``afriscience-mt plot`` calls.

CSV columns:
    experiment_type, model_short, model_name, model_type,
    source_lang, target_lang, lang_pair, dataset, split,
    prompt_strategy, lora_rank, n_shots,
    bleu, chrf, ssa_comet, num_samples,
    val_bleu, val_chrf, val_ssa_comet,
    timestamp, source, original_path
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from afriscience_mt.core.schema import load_result


COLUMNS = [
    "experiment_type", "model_short", "model_name", "model_type",
    "source_lang", "target_lang", "lang_pair", "dataset", "split",
    "prompt_strategy", "lora_rank", "n_shots",
    "is_ablation", "temp_setting",
    "bleu", "chrf", "ssa_comet", "num_samples",
    "val_bleu", "val_chrf", "val_ssa_comet",
    "timestamp", "source", "original_path",
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="afriscience-mt aggregate")
    p.add_argument("--src", default="experiments",
                   help="root containing canonical results (default: experiments/)")
    p.add_argument("--out", default="experiments/summary.csv",
                   help="output file (.csv or .json)")
    p.add_argument("--filter-type", action="append",
                   help="only include experiments matching this type "
                        "(may be repeated)")
    p.add_argument("--filter-model", action="append",
                   help="only include experiments matching this model_short "
                        "(may be repeated)")
    args = p.parse_args(argv)

    src = Path(args.src)
    rows: list[dict] = []
    for path in src.rglob("results.json"):
        try:
            r = load_result(path)
        except Exception as e:
            print(f"  skip {path}: {e}")
            continue
        if args.filter_type and r.experiment_type not in args.filter_type:
            continue
        if args.filter_model and (r.model.short_name or "") not in args.filter_model:
            continue
        rows.append(_row(r))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".json":
        with open(out, "w") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
    else:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"wrote {len(rows)} rows to {out}")
    return 0


def _row(r) -> dict:
    md = r.metadata or {}
    val = r.val_metrics
    return {
        "experiment_type": r.experiment_type,
        "model_short": r.model.short_name or "",
        "model_name": r.model.name,
        "model_type": r.model.type,
        "source_lang": r.source_lang,
        "target_lang": r.target_lang,
        "lang_pair": r.lang_pair,
        "dataset": r.dataset,
        "split": r.split,
        "prompt_strategy": md.get("prompt_strategy", ""),
        "lora_rank": (r.config or {}).get("lora_rank", ""),
        "n_shots": (r.config or {}).get("n_shots", ""),
        "is_ablation": bool(md.get("is_ablation", False)),
        "temp_setting": md.get("temp_setting", ""),
        "bleu": r.metrics.bleu,
        "chrf": r.metrics.chrf,
        "ssa_comet": r.metrics.ssa_comet,
        "num_samples": r.metrics.num_samples,
        "val_bleu": val.bleu if val else "",
        "val_chrf": val.chrf if val else "",
        "val_ssa_comet": val.ssa_comet if val else "",
        "timestamp": r.timestamp,
        "source": md.get("source", ""),
        "original_path": md.get("original_path", ""),
    }


if __name__ == "__main__":
    raise SystemExit(main())
