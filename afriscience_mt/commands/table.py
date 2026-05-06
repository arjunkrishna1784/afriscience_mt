"""Render paper-ready tables (markdown / LaTeX / CSV) from canonical results.

Canonical input: an aggregate CSV produced by ``afriscience-mt aggregate`` (default:
``experiments/summary.csv``), or a directory of canonical results that we'll
aggregate on the fly.

Common table layouts:
  --layout model_x_pair       rows=model_short, cols=lang_pair
  --layout model_x_lang       rows=model_short, cols=target_lang (avg over directions)
  --layout pair_x_metric      rows=lang_pair, cols=metric
  --layout dataset_compare    rows=model_short × dataset, cols=lang_pair
                              (for MAFAND vs MAFAND+DCS comparison tables)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict
from typing import Iterable


METRIC_LABEL = {"bleu": "BLEU", "chrf": "ChrF", "ssa_comet": "SSA-COMET"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="afriscience-mt table")
    p.add_argument("--src", default="experiments/summary.csv",
                   help="aggregate CSV produced by `afriscience-mt aggregate` "
                        "(default: experiments/summary.csv)")
    p.add_argument("--out",
                   help="output file (.md / .tex / .csv). If omitted, prints to stdout")
    p.add_argument("--metric", default="ssa_comet",
                   choices=["bleu", "chrf", "ssa_comet"],
                   help="which metric to render (default: ssa_comet)")
    p.add_argument("--layout", default="model_x_pair",
                   choices=["model_x_pair", "model_x_lang", "pair_x_metric",
                            "dataset_compare"],
                   help="table shape")
    p.add_argument("--format", default=None,
                   help="output format: md | tex | csv (auto-detected from --out)")
    p.add_argument("--filter-type", action="append",
                   help="restrict to these experiment_type values")
    p.add_argument("--filter-model", action="append",
                   help="restrict to these model_short values")
    p.add_argument("--filter-prompt", action="append",
                   help="restrict to these prompt_strategy values")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--ablation", action="store_true",
                   help="show ONLY ablation rows (mafand_pp / closed-LLM t06)")
    g.add_argument("--include-both", action="store_true",
                   help="show ablation alongside actual (default: actual only)")
    p.add_argument("--bold-best", action="store_true",
                   help="bold the best score in each column (LaTeX/MD only)")
    p.add_argument("--decimals", type=int, default=2)
    args = p.parse_args(argv)

    rows = list(_read_csv(args.src))
    rows = _apply_filters(rows, args)
    rows = _apply_ablation_mode(rows, args)
    if not rows:
        print("no rows after filtering", file=sys.stderr)
        return 1

    fmt = args.format or _infer_format(args.out)
    if args.layout == "dataset_compare":
        table = _layout_dataset_compare(rows, args.metric)
    elif args.layout == "model_x_pair":
        table = _layout_model_x_pair(rows, args.metric)
    elif args.layout == "model_x_lang":
        table = _layout_model_x_lang(rows, args.metric)
    else:
        table = _layout_pair_x_metric(rows)

    rendered = _render(table, fmt=fmt, decimals=args.decimals,
                       bold_best=args.bold_best, metric=args.metric)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered)
        print(f"wrote {args.out}")
    else:
        print(rendered)
    return 0


# ---------- IO ----------

def _read_csv(path: str | Path) -> Iterable[dict]:
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in ("bleu", "chrf", "ssa_comet"):
                r[k] = _to_float(r.get(k))
            yield r


def _to_float(x):
    if x in (None, "", "None"):
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def _apply_filters(rows: list[dict], args) -> list[dict]:
    out = rows
    if args.filter_type:
        out = [r for r in out if r["experiment_type"] in args.filter_type]
    if args.filter_model:
        out = [r for r in out if r["model_short"] in args.filter_model]
    if args.filter_prompt:
        out = [r for r in out if r.get("prompt_strategy") in args.filter_prompt]
    return out


def _is_ablation(r: dict) -> bool:
    v = r.get("is_ablation", "")
    return str(v).strip().lower() in {"true", "1"}


def _apply_ablation_mode(rows: list[dict], args) -> list[dict]:
    """Default: drop ablation rows. --ablation: keep only ablation. --include-both: pass through."""
    if args.include_both:
        return rows
    if args.ablation:
        return [r for r in rows if _is_ablation(r)]
    return [r for r in rows if not _is_ablation(r)]


def _infer_format(out: str | None) -> str:
    if not out:
        return "md"
    suffix = Path(out).suffix.lower()
    return {".tex": "tex", ".md": "md", ".csv": "csv"}.get(suffix, "md")


# ---------- layouts ----------

def _layout_model_x_pair(rows: list[dict], metric: str) -> dict:
    """Rows = model_short, cols = lang_pair."""
    cols = sorted({r["lang_pair"] for r in rows})
    grouped = defaultdict(dict)
    for r in rows:
        v = r.get(metric)
        if v is None:
            continue
        grouped[r["model_short"]][r["lang_pair"]] = v
    return {
        "title": f"{METRIC_LABEL.get(metric, metric)} by model × lang-pair",
        "row_label": "Model",
        "col_label": "Pair",
        "cols": cols,
        "rows": [(m, [grouped[m].get(c) for c in cols])
                 for m in sorted(grouped)],
    }


def _layout_model_x_lang(rows: list[dict], metric: str) -> dict:
    """Rows = model_short, cols = target_lang (averaged across both directions)."""
    cols = sorted({r["target_lang"] for r in rows})
    bucket: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = r.get(metric)
        if v is None:
            continue
        # bucket each pair under the non-eng side
        non_eng = r["target_lang"] if r["source_lang"] == "eng" else r["source_lang"]
        bucket[r["model_short"]][non_eng].append(v)
    cols = sorted({lang for m in bucket.values() for lang in m})
    return {
        "title": f"{METRIC_LABEL.get(metric, metric)} by model × language (avg)",
        "row_label": "Model",
        "col_label": "Lang",
        "cols": cols,
        "rows": [(m, [_mean(bucket[m].get(c, [])) for c in cols])
                 for m in sorted(bucket)],
    }


def _layout_dataset_compare(rows: list[dict], metric: str) -> dict:
    """Rows = (model_short, dataset), cols = lang_pair. Used for MAFAND vs DCS+MAFAND."""
    grouped = defaultdict(dict)
    cols = sorted({r["lang_pair"] for r in rows})
    for r in rows:
        v = r.get(metric)
        if v is None:
            continue
        key = (r["model_short"], r["dataset"])
        grouped[key][r["lang_pair"]] = v
    rows_out = [(f"{m} ({ds})", [grouped[(m, ds)].get(c) for c in cols])
                for (m, ds) in sorted(grouped)]
    return {
        "title": f"{METRIC_LABEL.get(metric, metric)} by (model, dataset) × lang-pair",
        "row_label": "Model (dataset)",
        "col_label": "Pair",
        "cols": cols,
        "rows": rows_out,
    }


def _layout_pair_x_metric(rows: list[dict]) -> dict:
    """Rows = lang_pair, cols = (BLEU, ChrF, SSA-COMET) averaged across models."""
    pairs = sorted({r["lang_pair"] for r in rows})
    metrics = ["ssa_comet", "chrf", "bleu"]
    bucket: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m in metrics:
            v = r.get(m)
            if v is not None:
                bucket[r["lang_pair"]][m].append(v)
    return {
        "title": "Mean per-pair metrics across models",
        "row_label": "Pair",
        "col_label": "Metric",
        "cols": [METRIC_LABEL[m] for m in metrics],
        "rows": [(p, [_mean(bucket[p].get(m, [])) for m in metrics])
                 for p in pairs],
    }


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


# ---------- rendering ----------

def _render(table: dict, *, fmt: str, decimals: int, bold_best: bool,
            metric: str) -> str:
    if fmt == "csv":
        return _render_csv(table, decimals)
    if fmt == "tex":
        return _render_tex(table, decimals, bold_best, metric)
    return _render_md(table, decimals, bold_best)


def _render_csv(table: dict, decimals: int) -> str:
    out = [",".join([table["row_label"], *table["cols"]])]
    for label, vals in table["rows"]:
        out.append(",".join([label, *(_fmt(v, decimals) for v in vals)]))
    return "\n".join(out)


def _render_md(table: dict, decimals: int, bold_best: bool) -> str:
    best = _column_bests(table) if bold_best else None
    head = "| " + " | ".join([table["row_label"], *table["cols"]]) + " |"
    sep = "|" + "|".join(["---"] * (len(table["cols"]) + 1)) + "|"
    body = []
    for label, vals in table["rows"]:
        cells = [label]
        for ci, v in enumerate(vals):
            s = _fmt(v, decimals)
            if best and v is not None and v == best[ci]:
                s = f"**{s}**"
            cells.append(s)
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([f"### {table['title']}", "", head, sep, *body])


def _render_tex(table: dict, decimals: int, bold_best: bool, metric: str) -> str:
    best = _column_bests(table) if bold_best else None
    n_cols = len(table["cols"])
    spec = "l" + "r" * n_cols
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{table['title']}}}",
        f"\\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        " & ".join([table["row_label"], *table["cols"]]) + r" \\",
        r"\midrule",
    ]
    for label, vals in table["rows"]:
        cells = [_tex_escape(label)]
        for ci, v in enumerate(vals):
            s = _fmt(v, decimals)
            if best and v is not None and v == best[ci]:
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _column_bests(table: dict) -> list[float | None]:
    n = len(table["cols"])
    bests: list[float | None] = [None] * n
    for _, vals in table["rows"]:
        for i, v in enumerate(vals):
            if v is None:
                continue
            if bests[i] is None or v > bests[i]:
                bests[i] = v
    return bests


def _fmt(v, decimals: int) -> str:
    if v is None:
        return "--"
    return f"{v:.{decimals}f}"


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


if __name__ == "__main__":
    raise SystemExit(main())
