"""Redraw the data-driven paper figures in the same matplotlib style as the
new assets (no PGF/TikZ, no inkscape exports). Outputs land in
``_ACL2026.../figures/`` and overwrite-replace the legacy PDFs.

Figures produced (filenames intentionally match the ones referenced in
``acl_latex.tex`` / ``appendix_plots_and_tables.tex``):

  m2m100_heatmaps.pdf     -- M2M100 (418M, 1.2B) zero-shot vs. fine-tuned (chrF + COMET)
  nllb_heatmaps.pdf       -- NLLB (600M, 1.3B)   zero-shot vs. fine-tuned (chrF + COMET)
  gemma_m2m_heatmaps.pdf  -- best LLM LoRA (Gemma2-9B vs. LLaMA-3.1-8B at r=64)
  multilingual-nllb.pdf   -- multilingual NLLB-600M, training-update curves
  radar_chart_domain_chrf.pdf -- per-domain chrF radar across selected systems
  weighted_chrf_score.pdf -- per-domain chrF weighted by domain size

Per-domain chrF is computed live from each system's ``predictions.json``
plus the test TSV's ``category`` column. SSA-COMET is intentionally not
recomputed here (GPU + minutes); chrF is a fine secondary metric for the
domain-level story.

Run from the repo root:
  python3 scripts/redraw_paper_plots.py
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "experiments" / "summary.csv"
EXP_DIR = ROOT / "experiments"
TEST_DIR = ROOT / "data" / "processed_files" / "test"
FIG_DIR = (
    ROOT
    / "_ACL2026__Decolonizing_Science_in_Africa_through_Text_Translation_"
    / "figures"
    / "new"
)
FIG_DIR.mkdir(parents=True, exist_ok=True)

LANGS = ["amh", "hau", "lug", "nso", "yor", "zul"]
PAIRS = [f"eng-{l}" for l in LANGS] + [f"{l}-eng" for l in LANGS]
PAIR_LABEL = [f"eng-{l}" for l in LANGS] + [f"{l}-eng" for l in LANGS]

DOMAINS = [
    "agriculture", "biochemistry", "biology", "chemistry",
    "computer science", "engineering", "geography", "health",
    "Indigenous knowledge", "sociology", "statistics",
]
DOMAIN_DISP = {
    "agriculture": "Agric.", "biochemistry": "Biochem.", "biology": "Biology",
    "chemistry": "Chem.", "computer science": "CS", "engineering": "Eng.",
    "geography": "Geog.", "health": "Health",
    "Indigenous knowledge": "Indig.\nKnow.", "sociology": "Soc.",
    "statistics": "Stat.",
}

# Match style from build_paper_assets.py
COLORS = {
    "zs": "#9ecae1", "ft": "#3182bd",
    "llm_zs": "#fdae6b", "llm_ft": "#e6550d",
    "closed": "#74c476",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.5,
})


# ---------------- IO ----------------

def load_summary() -> list[dict]:
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            for k in ("bleu", "chrf", "ssa_comet"):
                v = r.get(k)
                r[k] = float(v) if v not in (None, "", "None") else None
            rows.append(r)
    return rows


def metric_val(r: dict, metric: str) -> float | None:
    v = r.get(metric)
    if v is None:
        return None
    return v * 100 if metric == "ssa_comet" else v


def lookup(rows, *, metric, **filters) -> dict[str, float]:
    out = {}
    for r in rows:
        ok = True
        for k, v in filters.items():
            if str(r.get(k, "")) != str(v):
                ok = False
                break
        if not ok:
            continue
        v = metric_val(r, metric)
        if v is not None:
            out[r["lang_pair"]] = v
    return out


# ---------------- per-domain chrF ----------------

def load_test(lang: str) -> list[dict]:
    with open(TEST_DIR / f"{lang}.tsv") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def per_domain_chrf(pred_path: Path, src_lang: str, tgt_lang: str) -> dict[str, float]:
    """Group by category and compute chrF per domain."""
    if not pred_path.exists():
        return {}
    preds = json.load(open(pred_path)).get("samples", [])
    src_test = load_test(src_lang)
    tgt_test = load_test(tgt_lang)
    if len(preds) != len(src_test):
        return {}
    by_cat = defaultdict(lambda: ([], []))
    for i, sample in enumerate(preds):
        cat = src_test[i]["category"]
        ref = tgt_test[i]["text"]
        hyp = sample.get("hyp") or ""
        by_cat[cat][0].append(hyp)
        by_cat[cat][1].append(ref)
    out = {}
    for cat, (hyps, refs) in by_cat.items():
        if not hyps:
            continue
        out[cat] = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    return out


def domain_chrf_for_system(experiment_dir: Path) -> dict[str, dict[str, float]]:
    """For each pair sub-dir under `experiment_dir`, compute per-domain chrF.

    Returns {lang_pair: {domain: chrF}}.
    """
    out = {}
    for pair_dir in experiment_dir.iterdir():
        if not pair_dir.is_dir():
            continue
        pair = pair_dir.name
        if pair not in PAIRS:
            continue
        src, tgt = pair.split("-")
        d = per_domain_chrf(pair_dir / "predictions.json", src, tgt)
        if d:
            out[pair] = d
    return out


# ---------------- helpers ----------------

def vec(d: dict, pairs=PAIRS):
    return np.array([d.get(p, np.nan) for p in pairs], dtype=float)


def heatmap_panel(ax, mat, *, row_labels, col_labels, title, vmin, vmax, cmap):
    """Single heatmap panel with cell annotations. ``title`` is accepted
    for back-compat but not rendered (titles are removed; LaTeX carries them)."""
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                txt = f"{v:.0f}"
                # Auto-contrast text color
                color = "white" if (v - vmin) / (vmax - vmin + 1e-9) > 0.55 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        color=color, fontsize=6.5)
    return im


# ---------------- figure builders ----------------

def fig_seq2seq_heatmaps(rows, family: str, models: list[str], out_name: str):
    """family in {'M2M100', 'NLLB'}; one row per model, columns = pairs.
    Panels: zero-shot (chrF), fine-tune (chrF), zero-shot (COMET), fine-tune (COMET).
    """
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.6))

    for ci, metric in enumerate(["chrf", "ssa_comet"]):
        zs_mat = np.array([
            vec(lookup(rows, metric=metric, model_short=m, experiment_type="zero_shot"))
            for m in models
        ])
        ft_mat = np.array([
            vec(lookup(rows, metric=metric, model_short=m,
                       experiment_type="seq2seq_finetune", dataset="dcs",
                       is_ablation="False"))
            for m in models
        ])
        # global vmin/vmax per metric for fair comparison
        all_vals = np.concatenate([zs_mat.flatten(), ft_mat.flatten()])
        vmin = float(np.nanmin(all_vals))
        vmax = float(np.nanmax(all_vals))
        rl = [m.replace("m2m100_", "M2M-").replace("_", "-").upper()
              .replace("NLLB-200-1-3B", "NLLB-1.3B")
              .replace("NLLB-200-DISTILLED-600M", "NLLB-600M")
              .replace("M2M-418M", "M2M-418M").replace("M2M-1-2B", "M2M-1.2B")
              for m in models]
        heatmap_panel(axes[0, ci], zs_mat,
                      row_labels=rl, col_labels=PAIR_LABEL,
                      title=f"Zero-shot ({'ChrF' if metric=='chrf' else 'SSA-COMET'})",
                      vmin=vmin, vmax=vmax, cmap="Blues")
        heatmap_panel(axes[1, ci], ft_mat,
                      row_labels=rl, col_labels=PAIR_LABEL,
                      title=f"Fine-tuned ({'ChrF' if metric=='chrf' else 'SSA-COMET'})",
                      vmin=vmin, vmax=vmax, cmap="Blues")
    fig.tight_layout()
    out = FIG_DIR / out_name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_lora_heatmap(rows):
    """Best LoRA models (Gemma2-9B and LLaMA-3.1-8B at r=64) — chrF + COMET."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.9))
    models = ["gemma2_9b_it", "llama3_8b"]
    rl = ["Gemma2-9B (r=64)", "LLaMA-3.1-8B (r=64)"]
    for i, metric in enumerate(["chrf", "ssa_comet"]):
        mat = np.array([
            vec(lookup(rows, metric=metric, model_short=m,
                       experiment_type="lora_finetune", lora_rank="64",
                       is_ablation="False"))
            for m in models
        ])
        vmin, vmax = float(np.nanmin(mat)), float(np.nanmax(mat))
        heatmap_panel(axes[i], mat,
                      row_labels=rl, col_labels=PAIR_LABEL,
                      title=f"LoRA r=64 ({'ChrF' if metric=='chrf' else 'SSA-COMET'})",
                      vmin=vmin, vmax=vmax, cmap="Oranges")
    fig.tight_layout()
    out = FIG_DIR / "gemma_m2m_heatmaps.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _checkpoint_from_path(p: str | None) -> str | None:
    if not p:
        return None
    m = re.search(r"multilingual/(\d+k)_updates", p)
    return m.group(1) if m else None


def fig_multilingual_nllb(rows):
    """NLLB-600M multilingual training-curve, line chart per direction × metric.

    X-axis = training updates (40k, 80k, 200k, 400k); one line per
    language. Four PDFs total under figures/new/multilingual_nllb/:
      comet/eng_xxx.pdf,  comet/xxx_eng.pdf
      chrf/eng_xxx.pdf,   chrf/xxx_eng.pdf
    """
    chk = ["40k", "80k", "200k", "400k"]
    chk_x = [40, 80, 200, 400]   # numeric for the x-axis

    # data[metric][direction][lang][checkpoint] = score
    data = {
        "chrf":      {0: {l: {c: None for c in chk} for l in LANGS},
                      1: {l: {c: None for c in chk} for l in LANGS}},
        "ssa_comet": {0: {l: {c: None for c in chk} for l in LANGS},
                      1: {l: {c: None for c in chk} for l in LANGS}},
    }
    for r in rows:
        if r["experiment_type"] != "multilingual_finetune":
            continue
        c = _checkpoint_from_path(r.get("original_path"))
        if c not in chk:
            continue
        non_eng = (r["target_lang"] if r["source_lang"] == "eng"
                   else r["source_lang"])
        if non_eng not in LANGS:
            continue
        direction = 0 if r["source_lang"] == "eng" else 1
        for metric in ("chrf", "ssa_comet"):
            v = metric_val(r, metric)
            if v is not None:
                data[metric][direction][non_eng][c] = v

    # Per-checkpoint bar colours: light→dark blues so the within-group
    # progression reads at a glance.
    cmap = matplotlib.colormaps["Blues"]
    bar_colors = [cmap(0.30 + 0.18 * i) for i in range(len(chk))]

    written = []
    for metric, metric_disp, sub in [("ssa_comet", "COMET", "comet"),
                                     ("chrf",      "ChrF",  "chrf")]:
        for di, dir_disp in [(0, "eng_xxx"), (1, "xxx_eng")]:
            fig, ax = plt.subplots(figsize=(4.4, 2.6))
            n_lang = len(LANGS)
            n_chk  = len(chk)
            group_width = 0.85
            bar_width = group_width / n_chk
            x_centres = np.arange(n_lang)

            for li, lang in enumerate(LANGS):
                # Within-group x positions for the 4 bars.
                bar_xs = (x_centres[li] - group_width / 2 + bar_width / 2
                          + bar_width * np.arange(n_chk))
                ys = [data[metric][di][lang][c] for c in chk]
                # Bars
                for ci, (bx, y, col) in enumerate(zip(bar_xs, ys, bar_colors)):
                    if y is None:
                        continue
                    ax.bar(bx, y, width=bar_width, color=col,
                           edgecolor="black", linewidth=0.3,
                           label=f"{chk[ci]} updates" if li == 0 else None,
                           zorder=2)
                # Trend line connecting the same group's bar tops.
                pairs = [(bx, y) for bx, y in zip(bar_xs, ys) if y is not None]
                if len(pairs) >= 2:
                    xs_l, ys_l = zip(*pairs)
                    ax.plot(xs_l, ys_l, color="#444", linewidth=0.8,
                            marker="o", markersize=2.2,
                            markerfacecolor="white",
                            markeredgewidth=0.5, zorder=3)

            ax.set_xticks(x_centres)
            ax.set_xticklabels(LANGS, fontsize=7)
            ax.set_ylabel(metric_disp, fontsize=7)
            # Tight y-range so each direction × metric panel fills its frame.
            all_y = [data[metric][di][l][c] for l in LANGS for c in chk
                     if data[metric][di][l][c] is not None]
            if all_y:
                lo = min(all_y); hi = max(all_y)
                pad = max(1.0, (hi - lo) * 0.10)
                ax.set_ylim(max(0, lo - pad), hi + pad)
            ax.tick_params(labelsize=6.5)
            ax.grid(axis="y", linestyle=":", alpha=0.5, linewidth=0.4)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_linewidth(0.5)
            # Legend in upper-left corner (training-update bins only,
            # populated from the first language's bars).
            ax.legend(loc="upper left", fontsize=6, frameon=False, ncol=4,
                      bbox_to_anchor=(0.0, 1.18))
            fig.tight_layout()

            out = FIG_DIR / f"multilingual_nllb/{sub}/{dir_disp}.pdf"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            written.append(out)
    print(f"  ✓ {len(written)} PDFs in "
          f"{(FIG_DIR / 'multilingual_nllb').relative_to(ROOT)}/")
    return FIG_DIR / "multilingual_nllb"


def _full_lang_grid(rows, *, model_short, exp_type, metric, dataset=None,
                    lora_rank=None, prompt_strategy=None, is_ablation="False"):
    """7x7 (eng + 6 African) source × target matrix of `metric` for the matching
    rows. Cells with src==tgt or no data are NaN."""
    LFULL = ["eng"] + LANGS                 # 7 langs
    n = len(LFULL)
    mat = np.full((n, n), np.nan)
    for r in rows:
        if r["experiment_type"] != exp_type:
            continue
        if r["model_short"] != model_short:
            continue
        if dataset is not None and r["dataset"] != dataset:
            continue
        if lora_rank is not None and str(r["lora_rank"]) != str(lora_rank):
            continue
        if prompt_strategy is not None and r["prompt_strategy"] != prompt_strategy:
            continue
        if is_ablation is not None and str(r["is_ablation"]) != is_ablation:
            continue
        if r["source_lang"] not in LFULL or r["target_lang"] not in LFULL:
            continue
        v = metric_val(r, metric)
        if v is None:
            continue
        i = LFULL.index(r["source_lang"])
        j = LFULL.index(r["target_lang"])
        mat[i, j] = v
    return mat, LFULL


def _draw_7x7(ax, mat, langs, title, vmin, vmax, *, show_xlabel, show_ylabel,
              fontsize=7, decimals=1):
    """``title`` accepted but not rendered — LaTeX caption carries it."""
    im = ax.imshow(mat, cmap="Blues", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(np.arange(len(langs)))
    ax.set_yticks(np.arange(len(langs)))
    ax.set_xticklabels(langs, fontsize=fontsize)
    ax.set_yticklabels(langs, fontsize=fontsize)
    if show_xlabel:
        ax.set_xlabel("target", fontsize=fontsize)
    if show_ylabel:
        ax.set_ylabel("source", fontsize=fontsize)
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center",
                        color="lightgray", fontsize=fontsize)
                continue
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        color="lightgray", fontsize=fontsize)
                continue
            color = ("white" if (v - vmin) / (vmax - vmin + 1e-9) > 0.55
                    else "black")
            ax.text(j, i, f"{v:.{decimals}f}", ha="center", va="center",
                    color=color, fontsize=fontsize)
    return im


def fig_seq2seq_7x7(rows, model_short, model_disp, metric_disp, metric, fname):
    """Figure 7 layout: 1×2 panels — Zero-Shot | Fine-Tuned, 7×7 each.
    Note: fine-tuned X↔X is empty in the current dataset.
    """
    zs_mat, langs = _full_lang_grid(rows, model_short=model_short,
                                    exp_type="zero_shot", metric=metric)
    ft_mat, _ = _full_lang_grid(rows, model_short=model_short,
                                exp_type="seq2seq_finetune", metric=metric,
                                dataset="dcs")
    all_vals = np.concatenate([zs_mat[~np.isnan(zs_mat)],
                               ft_mat[~np.isnan(ft_mat)]])
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6))
    _draw_7x7(axes[0], zs_mat, langs,
              f"Zero-Shot Models {metric_disp} Scores",
              vmin, vmax, show_xlabel=True, show_ylabel=True)
    im = _draw_7x7(axes[1], ft_mat, langs,
                   f"Fine-Tuned Models {metric_disp} Scores",
                   vmin, vmax, show_xlabel=True, show_ylabel=False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
    cbar.ax.tick_params(labelsize=6)
    out = FIG_DIR / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_lora_7x7(rows, model_short, model_disp, lora_rank, fname):
    """Figure 8 layout: COMET | ChrF, both 7×7 fine-tuned-only at the given LoRA rank."""
    comet_mat, langs = _full_lang_grid(
        rows, model_short=model_short, exp_type="lora_finetune",
        metric="ssa_comet", dataset="dcs", lora_rank=lora_rank)
    chrf_mat, _ = _full_lang_grid(
        rows, model_short=model_short, exp_type="lora_finetune",
        metric="chrf", dataset="dcs", lora_rank=lora_rank)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6))
    cv = comet_mat[~np.isnan(comet_mat)]
    chv = chrf_mat[~np.isnan(chrf_mat)]
    cmin, cmax = (float(cv.min()), float(cv.max())) if cv.size else (0., 100.)
    hmin, hmax = (float(chv.min()), float(chv.max())) if chv.size else (0., 100.)

    im_c = _draw_7x7(axes[0], comet_mat, langs, "COMET Scores",
                     cmin, cmax, show_xlabel=True, show_ylabel=True)
    im_h = _draw_7x7(axes[1], chrf_mat, langs, "ChrF Scores",
                     hmin, hmax, show_xlabel=True, show_ylabel=False)
    fig.colorbar(im_h, ax=axes, fraction=0.025, pad=0.03)
    out = FIG_DIR / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_n2n_heatmap(rows):
    """6x6 African-to-African zero-shot heatmaps for the two larger seq2seq models
    (NLLB-1.3B, M2M100-1.2B), chrF and SSA-COMET. Diagonal masked (src == tgt).
    """
    models = [
        ("NLLB-1.3B",   "nllb_200_1_3b"),
        ("M2M100-1.2B", "m2m100_1_2b"),
    ]
    metrics = [("ChrF", "chrf"), ("SSA-COMET", "ssa_comet")]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0))
    for mi, (model_disp, model_short) in enumerate(models):
        for ki, (metric_disp, metric) in enumerate(metrics):
            ax = axes[mi, ki]
            mat = np.full((len(LANGS), len(LANGS)), np.nan)
            for r in rows:
                if (r["experiment_type"] != "zero_shot"
                        or r["model_short"] != model_short
                        or r["source_lang"] not in LANGS
                        or r["target_lang"] not in LANGS):
                    continue
                v = metric_val(r, metric)
                if v is None:
                    continue
                i = LANGS.index(r["source_lang"])
                j = LANGS.index(r["target_lang"])
                mat[i, j] = v

            vmin = float(np.nanmin(mat))
            vmax = float(np.nanmax(mat))
            im = ax.imshow(mat, cmap="Blues", vmin=vmin, vmax=vmax, aspect="equal")
            ax.set_xticks(np.arange(len(LANGS)))
            ax.set_yticks(np.arange(len(LANGS)))
            ax.set_xticklabels(LANGS)
            ax.set_yticklabels(LANGS)
            if ki == 0:
                ax.set_ylabel(f"{model_disp}\nsource", fontsize=7)
            if mi == 0:
                ax.text(0.5, 1.05, metric_disp, transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=8)
            if mi == len(models) - 1:
                ax.set_xlabel("target")
            ax.grid(False)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    v = mat[i, j]
                    if np.isnan(v):
                        ax.text(j, i, "—", ha="center", va="center",
                                color="gray", fontsize=7)
                        continue
                    color = ("white" if (v - vmin) / (vmax - vmin + 1e-9) > 0.55
                            else "black")
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            color=color, fontsize=7)
    fig.tight_layout()
    out = FIG_DIR / "n2n_zero_shot_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


SPIDER_DOMAIN_SHORT = {
    "chemistry": "Chem.", "biology": "Biology",
    "biochemistry": "Biochem.", "agriculture": "Agric.",
    "statistics": "Stat.", "sociology": "Soc.",
    "Indigenous knowledge": "Indig.",
    "health": "Health", "geography": "Geog.",
    "engineering": "Eng.", "computer science": "CS",
}


def per_domain_metric(pred_path: Path, src_lang: str, tgt_lang: str,
                      metric: str) -> dict[str, float]:
    """Per-domain `metric` for one (system, pair) cell.

    For SSA-COMET we read the stored ``metrics.per_domain[<cat>].ssa_comet``
    block from the sibling results.json (populated by
    ``scripts/rescore_per_domain_ssa.py``). For chrF we compute on the fly
    from predictions.json + the test split's category column.
    """
    if metric == "chrf":
        return per_domain_chrf(pred_path, src_lang, tgt_lang)
    if metric != "ssa_comet":
        raise ValueError(metric)
    res = pred_path.parent / "results.json"
    if not res.exists():
        return {}
    pd = (json.loads(res.read_text()).get("metrics") or {}).get("per_domain")
    if not pd:
        return {}
    out = {}
    for cat, stats in pd.items():
        v = stats.get("ssa_comet")
        if v is None:
            continue
        out[cat] = v * 100  # 0..1 -> 0..100 for plotting
    return out


def fig_spider_per_pair_subdir(metric: str = "chrf"):
    """Per-pair spider PDFs in two flavours:

      figures/new/spider_per_pair/<pair>.pdf       -- 3 systems
        (NLLB-1.3B FT, Gemma2-9B LoRA r=64, GPT-4o template5_10shot)

      figures/new/spider_per_pair_4sys/<pair>.pdf  -- 4 systems
        (adds Tiny-Aya-Earth LoRA r=64)

    Both directions (12 pairs each). Drop into LaTeX as `\\subfigure`s.

    GPT-4o uses ``template5_10shot`` at $T{=}0$ on both directions --
    this matches the locked body configuration in
    Table~\\ref{tab:open_vs_closed_icl_comet}, so the spider plot
    aligns with the headline numbers.

    Backbone rationale:
      * best seq2seq:    NLLB-1.3B fine-tuned on dcs
      * best open LLM:   Gemma2-9B at LoRA r=64 (avg SSA-COMET 57.9 — top of
                         the open-LLM LoRA matrix)
      * best closed LLM: GPT-4o template5_10shot ICL (the locked body config)
      * + Tiny-Aya-Earth LoRA r=64 for the 4-system version (Africa-focused
        open base model; avg SSA-COMET 57.1, second only to Gemma).
    """
    domain_order = [
        "chemistry", "biology", "biochemistry", "agriculture", "statistics",
        "sociology", "Indigenous knowledge", "health", "geography",
        "engineering", "computer science",
    ]
    nllb_base   = EXP_DIR / "seq2seq" / "finetune" / "nllb_200_1_3b" / "dcs"
    gemma_base  = EXP_DIR / "llm" / "open" / "finetune" / "gemma2_9b_it" / "r64"
    aya_base    = EXP_DIR / "llm" / "open" / "finetune" / "tiny_aya_earth" / "r64"
    # GPT-4o template5_10shot at T=0 in both directions (per-domain rescored
    # 2026-05-09, commit 81083391).
    gpt_e2x     = EXP_DIR / "llm" / "closed" / "t0" / "icl" / "gpt4o" / "template5_10shot"
    gpt_x2e     = gpt_e2x

    # Each system: (name, base_dir, line_color, fill_color, linestyle, marker).
    # Distinct linestyle+marker per system so the spider remains legible in
    # colorless print.
    def systems3(pair: str):
        gpt_base = gpt_e2x if pair.startswith("eng-") else gpt_x2e
        return [
            ("NLLB-1.3B",        nllb_base,  "#3b67c4", "#a9bde6", "-",  "o"),
            ("Gemma2-9B (LoRA)", gemma_base, "#2ca25f", "#b8e0c8", "--", "s"),
            ("GPT-4o",           gpt_base,   "#d24a4a", "#f5c1c1", ":",  "^"),
        ]

    def systems4(pair: str):
        gpt_base = gpt_e2x if pair.startswith("eng-") else gpt_x2e
        return [
            ("NLLB-1.3B",        nllb_base,  "#3b67c4", "#a9bde6", "-",   "o"),
            ("Gemma2-9B (LoRA)", gemma_base, "#2ca25f", "#b8e0c8", "--",  "s"),
            ("Tiny-Aya-E (LoRA)", aya_base,  "#9467bd", "#d8c8e8", "-.",  "^"),
            ("GPT-4o",           gpt_base,   "#d24a4a", "#f5c1c1", ":",   "D"),
        ]

    pairs = [f"eng-{l}" for l in LANGS] + [f"{l}-eng" for l in LANGS]

    angles = np.linspace(0, 2 * np.pi, len(domain_order), endpoint=False).tolist()
    angles += angles[:1]

    def emit(systems_fn, subdir_name):
        # Global y_top — tightest circle that still contains every polygon
        # value plus a 1-pt buffer (avoids polygons touching the outer edge).
        all_vals = []
        for pair in pairs:
            src, tgt = pair.split("-")
            for sys in systems_fn(pair):
                base = sys[1]
                d = per_domain_metric(base / pair / "predictions.json",
                                      src, tgt, metric)
                all_vals.extend(d.values())
        if not all_vals:
            print(f"  ! {subdir_name}: no per-domain {metric} data found "
                  "(run scripts/rescore_per_domain_ssa.py first?)")
            return None
        y_top = int(np.ceil(max(all_vals))) + 1
        y_ticks = [v for v in (20, 40, 60, 80) if v < y_top]

        out_dir = FIG_DIR / subdir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for pair in pairs:
            src, tgt = pair.split("-")
            # Smaller figsize -> smaller native circle, no need for LaTeX
            # \includegraphics scaling tricks to fit on one page.
            fig = plt.figure(figsize=(1.9, 1.9))
            ax = fig.add_subplot(111, polar=True)
            for (sys_name, base, line_c, fill_c, ls, marker) in systems_fn(pair):
                d = per_domain_metric(base / pair / "predictions.json",
                                      src, tgt, metric)
                vals = [d.get(dom, 0.0) for dom in domain_order]
                vals += vals[:1]
                ax.plot(angles, vals, color=line_c, linewidth=0.6,
                        linestyle=ls, marker=marker, markersize=1.6,
                        markeredgewidth=0.3, markerfacecolor=line_c,
                        label=sys_name)
                ax.fill(angles, vals, color=fill_c, alpha=0.12)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([SPIDER_DOMAIN_SHORT[d] for d in domain_order],
                               fontsize=5)
            ax.set_ylim(0, y_top)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([str(t) for t in y_ticks],
                               fontsize=4, color="gray")
            ax.tick_params(pad=0.5)
            ax.grid(linestyle=":", alpha=0.5, linewidth=0.4)
            ax.spines["polar"].set_visible(True)
            ax.spines["polar"].set_linewidth(0.5)
            ax.patch.set_alpha(0)
            fig.tight_layout()
            fig.savefig(out_dir / f"{pair}.pdf",
                        bbox_inches="tight",
                        transparent=True)
            plt.close(fig)
        print(f"  ✓ 12 PDFs in {out_dir.relative_to(ROOT)}/")
        return out_dir

    sub = "comet" if metric == "ssa_comet" else "chrf"
    emit(systems3, f"spider_per_pair/3sys/{sub}")
    emit(systems4, f"spider_per_pair/4sys/{sub}")
    return FIG_DIR / "spider_per_pair"


def fig_spider_per_pair():
    """Page-23-style spider charts: one polygon per system per Eng→X pair, six
    panels in a 3×2 grid. Two systems compared: NLLB-1.3B (fine-tuned) and
    GPT-4o (5-shot ICL).

    Domains plotted in the same order as page 23 (clockwise from top):
        Chemistry, Biology, Biochemistry, Agriculture, Statistics, Sociology,
        Indigenous Knowledge, Health, Geography, Engineering, Computer Science.
    No titles; LaTeX caption identifies the figure.
    """
    domain_order = [
        "chemistry", "biology", "biochemistry", "agriculture", "statistics",
        "sociology", "Indigenous knowledge", "health", "geography",
        "engineering", "computer science",
    ]
    domain_labels = {
        "chemistry": "Chemistry", "biology": "Biology",
        "biochemistry": "Biochemistry", "agriculture": "Agriculture",
        "statistics": "Statistics", "sociology": "Sociology",
        "Indigenous knowledge": "Indigenous Knowledge",
        "health": "Health", "geography": "Geography",
        "engineering": "Engineering", "computer science": "Computer Science",
    }

    systems = [
        ("NLLB",
         EXP_DIR / "seq2seq" / "finetune" / "nllb_200_1_3b" / "dcs",
         "#3b67c4", "#a9bde6"),                    # line, fill
        ("GPT-4o",
         EXP_DIR / "llm" / "closed" / "t0" / "icl" / "gpt4o" / "5shot_semantic",
         "#d24a4a", "#f5c1c1"),
    ]

    angles = np.linspace(0, 2 * np.pi, len(domain_order), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(3, 2, figsize=(7.0, 9.0),
                             subplot_kw={"polar": True})
    axes = axes.flatten()

    for ai, lang in enumerate(LANGS):
        ax = axes[ai]
        pair = f"eng-{lang}"
        for sys_name, base, line_c, fill_c in systems:
            d = per_domain_chrf(base / pair / "predictions.json", "eng", lang)
            vals = [d.get(dom, 0.0) for dom in domain_order]
            vals += vals[:1]
            ax.plot(angles, vals, color=line_c, linewidth=1.4, label=sys_name)
            ax.fill(angles, vals, color=fill_c, alpha=0.45)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([domain_labels[d] for d in domain_order],
                           fontsize=6.5)
        ax.set_ylim(0, 70)
        ax.set_yticks([10, 20, 30, 40, 50, 60, 70])
        ax.set_yticklabels(["10", "20", "30", "40", "50", "60", "70"],
                           fontsize=5.5, color="gray")
        ax.tick_params(pad=2)
        # tiny pair label inside the panel (top-left), no title
        ax.text(0.05, 1.08, pair, transform=ax.transAxes,
                ha="left", va="top", fontsize=8)
        ax.grid(linestyle=":", alpha=0.6)

    # Single shared legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.01),
               ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    out = FIG_DIR / "spider_per_pair_eng_xxx.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_per_domain_radar():
    """Per-domain chrF radar — selected systems × 11 domains, eng-X averaged."""
    # GPT-4o eng-X predictions live under t0/icl/gpt4o/5shot_semantic/
    # (zero_shot/ holds only results.json — predictions weren't archived).
    systems = [
        ("NLLB-1.3B (zero-shot)",
         EXP_DIR / "seq2seq" / "zero_shot" / "nllb_200_1_3b",        "#9ecae1"),
        ("NLLB-1.3B (fine-tuned)",
         EXP_DIR / "seq2seq" / "finetune" / "nllb_200_1_3b" / "dcs", "#3182bd"),
        ("LLaMA-3.1-8B LoRA r=64",
         EXP_DIR / "llm" / "open" / "finetune" / "llama3_8b" / "r64", "#e6550d"),
        ("GPT-4o (5-shot ICL)",
         EXP_DIR / "llm" / "closed" / "t0" / "icl" / "gpt4o" / "5shot_semantic",
         "#74c476"),
    ]

    # Compute per-domain chrF, averaged over the 6 eng->X pairs
    per_sys = {}
    for name, base, _ in systems:
        domain_chrf = defaultdict(list)
        for lang in LANGS:
            pair = f"eng-{lang}"
            d = per_domain_chrf(base / pair / "predictions.json", "eng", lang)
            for cat, v in d.items():
                domain_chrf[cat].append(v)
        per_sys[name] = {
            cat: sum(vs) / len(vs) for cat, vs in domain_chrf.items()
        }

    # Radar
    angles = np.linspace(0, 2 * np.pi, len(DOMAINS), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.0, 5.0), subplot_kw={"polar": True})
    for (name, _, color) in systems:
        vals = [per_sys[name].get(d, 0) for d in DOMAINS]
        vals += vals[:1]
        ax.plot(angles, vals, label=name, color=color, linewidth=1.5)
        ax.fill(angles, vals, color=color, alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([DOMAIN_DISP[d] for d in DOMAINS], fontsize=7)
    ax.set_ylim(0, 70)
    ax.set_yticks([20, 40, 60])
    ax.set_yticklabels(["20", "40", "60"], fontsize=6)
    ax.tick_params(pad=6)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, frameon=False, fontsize=7)
    out = FIG_DIR / "radar_chart_domain_chrf.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_per_domain_weighted():
    """Bar chart: per-domain chrF for the top systems, ordered by domain size."""
    domain_sizes = {
        "chemistry": 46, "Indigenous knowledge": 92, "engineering": 93,
        "statistics": 184, "computer science": 213, "geography": 218,
        "agriculture": 388, "biology": 632, "biochemistry": 651,
        "sociology": 670, "health": 2605,
    }
    domains_sorted = sorted(DOMAINS, key=lambda d: domain_sizes[d])

    systems = [
        ("NLLB-1.3B (zero-shot)",
         EXP_DIR / "seq2seq" / "zero_shot" / "nllb_200_1_3b",        "#9ecae1"),
        ("NLLB-1.3B (fine-tuned)",
         EXP_DIR / "seq2seq" / "finetune" / "nllb_200_1_3b" / "dcs", "#3182bd"),
        ("GPT-4o (5-shot ICL)",
         EXP_DIR / "llm" / "closed" / "t0" / "icl" / "gpt4o" / "5shot_semantic",
         "#74c476"),
    ]
    per_sys = {}
    for name, base, _ in systems:
        domain_chrf = defaultdict(list)
        for lang in LANGS:
            d = per_domain_chrf(base / f"eng-{lang}" / "predictions.json", "eng", lang)
            for cat, v in d.items():
                domain_chrf[cat].append(v)
        per_sys[name] = {cat: sum(vs)/len(vs) for cat, vs in domain_chrf.items()}

    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    x = np.arange(len(domains_sorted))
    w = 0.25
    for i, (name, _, color) in enumerate(systems):
        vals = [per_sys[name].get(d, 0) for d in domains_sorted]
        ax.bar(x + (i - 1) * w, vals, width=w, label=name,
               color=color, edgecolor="black", linewidth=0.3)
    # Annotate domain sizes under labels (real newline; LaTeX-safe via mathtext-off)
    labels = [DOMAIN_DISP[d].replace("\n", " ") + f"\n(N={domain_sizes[d]})"
              for d in domains_sorted]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("ChrF (avg over Eng$\\to$xxx)")
    ax.set_ylim(0, 80)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18),
              ncol=3, frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "weighted_chrf_score.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------- main ----------------

def main():
    rows = load_summary()
    written = []

    written.append(fig_seq2seq_heatmaps(
        rows, "M2M-100", ["m2m100_418m", "m2m100_1_2b"], "m2m100_heatmaps.pdf"))
    written.append(fig_seq2seq_heatmaps(
        rows, "NLLB", ["nllb_200_distilled_600m", "nllb_200_1_3b"],
        "nllb_heatmaps.pdf"))
    written.append(fig_lora_heatmap(rows))
    written.append(fig_multilingual_nllb(rows))
    written.append(fig_n2n_heatmap(rows))

    # Figure 7-style 7×7 zero-shot vs fine-tuned, per seq2seq model × metric.
    # Output mirrored under fig7/{comet,chrf}/<short>.pdf.
    # Layout: figures/new/n2n/{small,large}/{comet,chrf}/<model>.pdf
    fig7_targets = [
        # (model_short, display, size_bucket)
        ("m2m100_418m",             "M2M100-418M", "small"),
        ("nllb_200_distilled_600m", "NLLB-600M",   "small"),
        ("m2m100_1_2b",             "M2M100-1.2B", "large"),
        ("nllb_200_1_3b",           "NLLB-1.3B",   "large"),
    ]
    for model_short, disp, size_bucket in fig7_targets:
        for metric_disp, metric, sub in [("ChrF",      "chrf",      "chrf"),
                                          ("SSA-COMET", "ssa_comet", "comet")]:
            short = (model_short
                     .replace("nllb_200_distilled_600m", "nllb_600m")
                     .replace("nllb_200_1_3b", "nllb_1_3b"))
            fname = f"n2n/{size_bucket}/{sub}/{short}.pdf"
            written.append(fig_seq2seq_7x7(rows, model_short, disp,
                                           metric_disp, metric, fname))
    # Figure 8-style 7×7 LoRA — keep both metrics side-by-side in one PDF
    written.append(fig_lora_7x7(rows, "gemma2_9b_it", "Gemma2-9B", "64",
                                "fig8/gemma2_9b_r64.pdf"))

    print("Computing per-domain chrF (this takes a few seconds per system)...")
    written.append(fig_per_domain_radar())
    written.append(fig_per_domain_weighted())
    written.append(fig_spider_per_pair())
    # Mirrored chrf/ + comet/ subfolders. The comet/ branch only fills if
    # results.json files have per-domain SSA-COMET (run
    # scripts/rescore_per_domain_ssa.py first to populate them).
    fig_spider_per_pair_subdir(metric="chrf")
    fig_spider_per_pair_subdir(metric="ssa_comet")

    for p in written:
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
