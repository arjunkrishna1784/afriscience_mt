"""Render paper-ready figures from canonical results.

Reads the same aggregate CSV as ``afriscience-mt table``. Plots:

  --kind heatmap     model × lang_pair heatmap of one metric
  --kind bars        per-model bars (avg metric across pairs), grouped by experiment_type
  --kind radar       per-model radar across languages (one polygon per model)
  --kind dataset     side-by-side bars for two datasets per model (MAFAND vs MAFAND+DCS)
  --kind domain      bars or heatmap of per-domain scores (requires --domains-csv)
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="afriscience-mt plot")
    p.add_argument("--src", default="experiments/summary.csv")
    p.add_argument("--out", required=True, help="output figure path (.pdf/.png)")
    p.add_argument("--kind", required=True,
                   choices=["heatmap", "bars", "radar", "dataset", "domain"])
    p.add_argument("--metric", default="ssa_comet",
                   choices=["bleu", "chrf", "ssa_comet"])
    p.add_argument("--filter-type", action="append")
    p.add_argument("--filter-model", action="append")
    p.add_argument("--filter-prompt", action="append")
    p.add_argument("--title")
    p.add_argument("--figsize", default="10,6",
                   help="comma-separated width,height in inches")
    args = p.parse_args(argv)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for `afriscience-mt plot`. pip install matplotlib",
              file=sys.stderr)
        return 1

    rows = _read(args.src)
    rows = _filter(rows, args)
    if not rows:
        print("no rows after filtering", file=sys.stderr)
        return 1

    fig_w, fig_h = (float(x) for x in args.figsize.split(","))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if args.kind == "heatmap":
        _plot_heatmap(ax, rows, args.metric)
    elif args.kind == "bars":
        _plot_bars(ax, rows, args.metric)
    elif args.kind == "radar":
        plt.close(fig)
        fig = _plot_radar(rows, args.metric, fig_w, fig_h)
    elif args.kind == "dataset":
        _plot_dataset_compare(ax, rows, args.metric)
    elif args.kind == "domain":
        print("domain plot requires per-domain CSV, not yet implemented",
              file=sys.stderr)
        return 1

    if args.title:
        fig.suptitle(args.title)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


def _read(path: str) -> list[dict]:
    with open(path) as f:
        out = []
        for r in csv.DictReader(f):
            for k in ("bleu", "chrf", "ssa_comet"):
                v = r.get(k)
                r[k] = float(v) if v not in (None, "", "None") else None
            out.append(r)
        return out


def _filter(rows: list[dict], args) -> list[dict]:
    out = rows
    if args.filter_type:
        out = [r for r in out if r["experiment_type"] in args.filter_type]
    if args.filter_model:
        out = [r for r in out if r["model_short"] in args.filter_model]
    if args.filter_prompt:
        out = [r for r in out if r.get("prompt_strategy") in args.filter_prompt]
    return out


def _plot_heatmap(ax, rows, metric):
    import numpy as np
    models = sorted({r["model_short"] for r in rows})
    pairs = sorted({r["lang_pair"] for r in rows})
    grid = np.full((len(models), len(pairs)), np.nan)
    for r in rows:
        v = r.get(metric)
        if v is None:
            continue
        i, j = models.index(r["model_short"]), pairs.index(r["lang_pair"])
        grid[i, j] = v
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, rotation=90, fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label(metric)


def _plot_bars(ax, rows, metric):
    grouped = defaultdict(list)
    for r in rows:
        v = r.get(metric)
        if v is not None:
            grouped[r["model_short"]].append(v)
    items = sorted(grouped.items(), key=lambda x: -sum(x[1]) / len(x[1]))
    labels = [k for k, _ in items]
    means = [sum(v) / len(v) for _, v in items]
    ax.barh(labels, means, color="#3b6fa8")
    ax.set_xlabel(metric)
    ax.invert_yaxis()


def _plot_radar(rows, metric, fig_w, fig_h):
    import numpy as np
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111, projection="polar")
    by_model: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = r.get(metric)
        if v is None:
            continue
        non_eng = r["target_lang"] if r["source_lang"] == "eng" else r["source_lang"]
        by_model[r["model_short"]][non_eng].append(v)
    langs = sorted({l for m in by_model.values() for l in m})
    angles = np.linspace(0, 2 * np.pi, len(langs), endpoint=False).tolist()
    angles += angles[:1]
    for model, lang_vals in by_model.items():
        vals = [sum(lang_vals.get(l, [0])) / max(1, len(lang_vals.get(l, [0])))
                for l in langs]
        vals += vals[:1]
        ax.plot(angles, vals, label=model, linewidth=1.5)
        ax.fill(angles, vals, alpha=0.05)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(langs)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    return fig


def _plot_dataset_compare(ax, rows, metric):
    import numpy as np
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = r.get(metric)
        if v is None:
            continue
        by[r["model_short"]][r["dataset"]].append(v)
    models = sorted(by)
    datasets = sorted({ds for m in by.values() for ds in m})
    width = 0.8 / len(datasets)
    x = np.arange(len(models))
    for i, ds in enumerate(datasets):
        means = [sum(by[m].get(ds, [])) / max(1, len(by[m].get(ds, [])))
                 if by[m].get(ds) else 0 for m in models]
        ax.bar(x + i * width, means, width, label=ds)
    ax.set_xticks(x + width * (len(datasets) - 1) / 2)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.legend()


if __name__ == "__main__":
    raise SystemExit(main())
