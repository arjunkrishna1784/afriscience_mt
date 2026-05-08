"""Figure-4-style ChrF curves for closed APIs (GPT-4o & Gemini-1.5-Flash):

For every language pair × API, plot ChrF as a function of #shots
(0, 5, 10, 20), one line per template (1..5), plus template-5
semantic-prompt variant on Eng→xxx. We cap the sweep at 20 shots:
50-shot adds <1 COMET on top of 10-shot for both closed APIs in our
data, so the extra column was dropped from figures and tables.

Layout matches page 22 of the existing draft:
    (a)  3 rows × 2 lang-cols, each cell is a (gemini | gpt-4o) pair-frame.
         6 Eng→xxx pairs; lines: templates 1-5 + Template 5 (sem.)
    (b)  same shape for the 6 xxx→Eng pairs (no semantic line).

Outputs (no titles; LaTeX caption carries the description):
    figures/new/fig4_eng_xxx.pdf
    figures/new/fig4_xxx_eng.pdf
    figures/new/fig4_combined.pdf       -- both halves stacked, optional
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
CLOSED = ROOT / "experiments" / "llm" / "closed"
FIG_DIR = (
    ROOT / "_ACL2026__Decolonizing_Science_in_Africa_through_Text_Translation_"
    / "figures" / "new"
)
FIG_DIR.mkdir(parents=True, exist_ok=True)

LANGS = ["amh", "hau", "lug", "nso", "yor", "zul"]
SHOTS = [0, 5, 10, 20]
TEMPLATES = [1, 2, 3, 4, 5]
MODELS = [("gemini_15_flash", "gemini"),
          ("gpt4o",            "gpt-4o")]

# Match the existing Figure 4 colour palette closely.
TPL_COLOR = {
    1: "#1f77b4",  # blue
    2: "#ff7f0e",  # orange
    3: "#2ca02c",  # green
    4: "#d62728",  # red
    5: "#9467bd",  # purple
}
TPL_MARKER = {1: "v", 2: "D", 3: "s", 4: "d", 5: "^"}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 6,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
    "legend.fontsize": 5.5,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.6,
    "grid.linewidth": 0.3,
})


def metric_at(model: str, temp_setting: str, prompt_strategy: str,
              pair: str, metric: str) -> float | None:
    p = (CLOSED / temp_setting / "icl" / model / prompt_strategy / pair
         / "results.json")
    if not p.exists():
        return None
    v = json.loads(p.read_text()).get("metrics", {}).get(metric)
    if v is None:
        return None
    return v * 100 if metric == "ssa_comet" else v


def metric_curve(model: str, temp_setting: str, template: int, pair: str,
                 metric: str) -> list[float | None]:
    return [metric_at(model, temp_setting, f"template{template}_{k}shot",
                      pair, metric) for k in SHOTS]


def semantic_curve(model: str, pair: str, metric: str) -> list[float | None]:
    """eng→xxx only (semantic ICL lives under t0/.../semantic_{K}shot/)."""
    out = [None]  # no 0-shot semantic by construction
    for k in [5, 10, 20]:
        out.append(metric_at(model, "t0", f"semantic_{k}shot", pair, metric))
    return out


def _setup_subplot(ax, *, ylim=None, model_label=None):
    ax.set_xticks(SHOTS)
    ax.set_xticklabels(["0", "5", "10", "20"])
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(pad=1)
    if model_label:
        ax.text(0.97, 0.04, model_label, transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=6, fontweight="bold", color="#444")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.5)


def _plot_pair_cell(axL, axR, *, eng_to_x: bool, lang: str, metric: str):
    """Two subplots side-by-side: gemini (left) | gpt-4o (right) for one pair."""
    pair = f"eng-{lang}" if eng_to_x else f"{lang}-eng"
    temp_setting = "t06" if eng_to_x else "t0"

    all_vals = []
    for (model, _disp) in MODELS:
        for tpl in TEMPLATES:
            ys = metric_curve(model, temp_setting, tpl, pair, metric)
            all_vals.extend(v for v in ys if v is not None)
        if eng_to_x:
            ys = semantic_curve(model, pair, metric)
            all_vals.extend(v for v in ys if v is not None)
    if not all_vals:
        return
    lo, hi = min(all_vals), max(all_vals)
    pad = max(1.5, (hi - lo) * 0.15)
    ylim = (lo - pad, hi + pad)

    for (ax, (model, model_disp)) in zip([axL, axR], MODELS):
        _setup_subplot(ax, ylim=ylim, model_label=model_disp)
        for tpl in TEMPLATES:
            ys = metric_curve(model, temp_setting, tpl, pair, metric)
            xs = [SHOTS[i] for i, v in enumerate(ys) if v is not None]
            ys_clean = [v for v in ys if v is not None]
            if not ys_clean:
                continue
            ax.plot(xs, ys_clean,
                    color=TPL_COLOR[tpl], marker=TPL_MARKER[tpl],
                    markersize=3, linewidth=0.8,
                    label=f"Template {tpl}")
        if eng_to_x:
            ys_sem = semantic_curve(model, pair, metric)
            xs = [SHOTS[i] for i, v in enumerate(ys_sem) if v is not None]
            ys_clean = [v for v in ys_sem if v is not None]
            if ys_clean:
                ax.plot(xs, ys_clean,
                        color="#7f3b00", marker="o",
                        markersize=2.5, linewidth=0.7,
                        linestyle="--",
                        label="Template 5 (sem. prompt)")


def _legend_handles(eng_to_x: bool):
    """Build a single shared legend for the figure."""
    handles, labels = [], []
    for tpl in TEMPLATES:
        h, = plt.plot([], [], color=TPL_COLOR[tpl], marker=TPL_MARKER[tpl],
                      markersize=3, linewidth=0.8)
        handles.append(h); labels.append(f"Template {tpl}")
    if eng_to_x:
        h, = plt.plot([], [], color="#7f3b00", marker="o",
                      markersize=2.5, linewidth=0.7, linestyle="--")
        handles.append(h); labels.append("Template 5 (sem. prompt)")
    return handles, labels


def build_one_direction(eng_to_x: bool, fname: str, metric: str = "ssa_comet"):
    """3 rows × 2 lang-cols, each cell = paired (gemini | gpt-4o) plot."""
    metric_label = "COMET" if metric == "ssa_comet" else "ChrF"
    fig = plt.figure(figsize=(7.5, 5.0))
    # outer grid: 3 rows × 2 columns; inner per cell: 1 row × 2 cols (paired plot)
    outer = fig.add_gridspec(3, 2, hspace=0.32, wspace=0.20,
                             left=0.06, right=0.99, top=0.96, bottom=0.10)

    pair_grid_order = [
        # (row, col, lang)
        (0, 0, "amh"), (0, 1, "hau"),
        (1, 0, "lug"), (1, 1, "nso"),
        (2, 0, "yor"), (2, 1, "zul"),
    ]
    for (r, c, lang) in pair_grid_order:
        inner = outer[r, c].subgridspec(1, 2, wspace=0.04)
        axL = fig.add_subplot(inner[0, 0])
        axR = fig.add_subplot(inner[0, 1])
        # Tiny pair label centred above the paired cell.
        bb = outer[r, c].get_position(fig)
        pair_label = f"eng→{lang}" if eng_to_x else f"{lang}→eng"
        fig.text(bb.x0 + bb.width / 2, bb.y0 + bb.height + 0.005,
                 pair_label, ha="center", va="bottom",
                 fontsize=7, family="monospace")
        _plot_pair_cell(axL, axR, eng_to_x=eng_to_x, lang=lang, metric=metric)
        # Hide y tick labels on the right subplot of each pair (shared scale).
        axR.set_yticklabels([])
        # X label only on the bottom row.
        if r == 2:
            axL.set_xlabel("Shots", fontsize=6)
            axR.set_xlabel("Shots", fontsize=6)
        # Y label only on the leftmost column's left subplot.
        if c == 0:
            axL.set_ylabel(metric_label, fontsize=6)
        # Frame around the paired cell (visual grouping)
        rect = FancyBboxPatch(
            (bb.x0, bb.y0), bb.width, bb.height,
            transform=fig.transFigure,
            boxstyle="round,pad=0.005,rounding_size=0.005",
            linewidth=0.4, edgecolor="#888", facecolor="none",
            zorder=0,
        )
        fig.add_artist(rect)

    handles, labels = _legend_handles(eng_to_x)
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.01),
               ncol=len(labels), frameon=False, fontsize=6)

    out = FIG_DIR / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.relative_to(ROOT)}")
    return out


def main():
    # Output layout (mirrored: same filenames in comet/ and chrf/):
    #   figures/new/fig4/comet/eng_xxx.pdf, xxx_eng.pdf
    #   figures/new/fig4/chrf/eng_xxx.pdf,  xxx_eng.pdf
    # Output layout (mirrored: same filenames in comet/ and chrf/):
    #   figures/new/gpt-vs-gemini/comet/{eng_xxx,xxx_eng}.pdf
    #   figures/new/gpt-vs-gemini/chrf/{eng_xxx,xxx_eng}.pdf
    print("GPT-vs-Gemini template/shot curves:")
    print("  [primary, SSA-COMET]")
    build_one_direction(True,  "gpt-vs-gemini/comet/eng_xxx.pdf", metric="ssa_comet")
    build_one_direction(False, "gpt-vs-gemini/comet/xxx_eng.pdf", metric="ssa_comet")
    print("  [secondary, ChrF]")
    build_one_direction(True,  "gpt-vs-gemini/chrf/eng_xxx.pdf",  metric="chrf")
    build_one_direction(False, "gpt-vs-gemini/chrf/xxx_eng.pdf",  metric="chrf")


if __name__ == "__main__":
    main()
