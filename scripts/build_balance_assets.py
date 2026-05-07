"""§4 domain-balance ablation assets for the new draft.

Three conditions on NLLB-1.3B + DCS train set, only the per-example
sampling weight changes:

    τ = 1   (proportional)  -> existing seq2seq/finetune/nllb_200_1_3b/dcs/
    τ = 2   (sqrt smoothing) -> seq2seq/finetune/nllb_200_1_3b/balance_t2/
    τ = ∞   (uniform)        -> seq2seq/finetune/nllb_200_1_3b/balance_tinf/

Outputs:
    tables/new/balance_comet.tex   -- 3 × 12-pair table, SSA-COMET
    tables/new/balance_chrf.tex    -- same shape, ChrF
    figures/new/balance_heatmap.pdf      -- 3-row × 11-domain heatmap (chrF)
    figures/new/balance_delta_by_size.pdf -- per-domain Δ chrF (τ=∞ − τ=1) ordered by N_d

ChrF is the per-domain metric (computed on the fly from predictions.json +
test-split category). The balance_t2/tinf results.json already have
per-domain stats but the dcs baseline does not — chrF on the fly is
consistent across all three conditions.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "tables" / "new"
FIG_DIR = (ROOT / "_ACL2026__Decolonizing_Science_in_Africa_through_Text_Translation_"
           / "figures" / "new")
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

NLLB_BASE = ROOT / "experiments" / "seq2seq" / "finetune" / "nllb_200_1_3b"
TEST_DIR = ROOT / "data" / "processed_files" / "test"

LANGS = ["amh", "hau", "lug", "nso", "yor", "zul"]
LANG_DISP = {"amh": "Amh", "hau": "Hau", "lug": "Lug",
             "nso": "NSo", "yor": "Yor", "zul": "Zul"}
PAIRS_E2X = [f"eng-{l}" for l in LANGS]
PAIRS_X2E = [f"{l}-eng" for l in LANGS]
PAIRS = PAIRS_E2X + PAIRS_X2E

# Domain name -> (display, train-set sentence count)
DOMAINS = [
    ("chemistry",            "Chem.",     46),
    ("Indigenous knowledge", "Indig.",    92),
    ("engineering",          "Eng.",      93),
    ("statistics",           "Stat.",    184),
    ("computer science",     "CS",       213),
    ("geography",            "Geog.",    218),
    ("agriculture",          "Agric.",   388),
    ("biology",              "Biology",  632),
    ("biochemistry",         "Biochem.", 651),
    ("sociology",            "Soc.",     670),
    ("health",               "Health",  2605),
]
DOMAIN_KEYS = [d[0] for d in DOMAINS]

CONDITIONS = [
    ("τ=1",   "dcs",          r"$\tau{=}1$"),
    ("τ=2",   "balance_t2",   r"$\tau{=}2$"),
    ("τ=∞",   "balance_tinf", r"$\tau{=}\infty$"),
]


# ---------------- IO ----------------

def load_test(lang: str) -> list[dict]:
    with open(TEST_DIR / f"{lang}.tsv") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def per_domain_chrf(pred_path: Path, src_lang: str, tgt_lang: str) -> dict[str, float]:
    if not pred_path.exists():
        return {}
    samples = json.load(open(pred_path)).get("samples", [])
    src = load_test(src_lang)
    tgt = load_test(tgt_lang)
    if len(samples) != len(src):
        return {}
    by_cat = defaultdict(lambda: ([], []))
    for i, s in enumerate(samples):
        cat = src[i]["category"]
        ref = tgt[i]["text"]
        hyp = s.get("hyp") or ""
        by_cat[cat][0].append(hyp)
        by_cat[cat][1].append(ref)
    return {
        cat: sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
        for cat, (hyps, refs) in by_cat.items() if hyps
    }


def overall_chrf(pred_path: Path, src_lang: str, tgt_lang: str) -> float | None:
    if not pred_path.exists():
        return None
    samples = json.load(open(pred_path)).get("samples", [])
    refs = [r["text"] for r in load_test(tgt_lang)]
    hyps = [s.get("hyp") or "" for s in samples]
    if len(hyps) != len(refs):
        return None
    return sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score


def overall_metric_from_results(cond_dir: Path, pair: str, metric: str) -> float | None:
    p = cond_dir / pair / "results.json"
    if not p.exists():
        return None
    m = json.load(open(p)).get("metrics", {}).get(metric)
    if m is None:
        return None
    return m * 100 if metric == "ssa_comet" else m


# ---------------- table builders ----------------

LANG_HEAD = " & ".join([LANG_DISP[l] for l in LANGS] * 2)

def render_balance_table(*, label: str, caption: str, blocks: list):
    """blocks = [(setup, [(τ_label, [12 floats|None]), ...]), ...]"""
    n = 12
    block_col_max = []
    for _, rs in blocks:
        cm = []
        for ci in range(n):
            vals = [r[1][ci] for r in rs if r[1][ci] is not None]
            cm.append(max(vals) if vals else None)
        block_col_max.append(cm)

    lines = [
        f"% ---- {label.split(':',1)[1]} ----",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\renewcommand{\arraystretch}{1.05}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{llrrrrrrrrrrrr}",
        r"\toprule",
        r"\textbf{Setup} & \textbf{$\tau$} & \multicolumn{6}{c}{\textbf{Eng$\to$}} & \multicolumn{6}{c}{\textbf{$\to$Eng}} \\",
        r"\cmidrule(lr){3-8}\cmidrule(lr){9-14}",
        f"  &   & {LANG_HEAD} \\\\",
        r"\midrule",
    ]
    for bi, (setup, rs) in enumerate(blocks):
        if bi > 0:
            lines.append(r"\midrule")
        col_max = block_col_max[bi]
        for ri, (tau_label, vals) in enumerate(rs):
            cells = []
            for ci, v in enumerate(vals):
                s = "--" if v is None else f"{v:.1f}"
                if v is not None and col_max[ci] is not None and abs(v - col_max[ci]) < 1e-9:
                    s = f"\\textbf{{{s}}}"
                cells.append(s)
            head_cell = (
                f"\\multirow{{{len(rs)}}}{{*}}{{\\textit{{{setup}}}}}"
                if ri == 0 else ""
            )
            lines.append(f"{head_cell} & {tau_label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def build_balance_tables():
    for metric, suffix, label_metric in [
        ("chrf",      "chrf",  "ChrF"),
        ("ssa_comet", "comet", "SSA-COMET"),
    ]:
        rows = []
        for tau_short, dir_name, tau_disp in CONDITIONS:
            cond_dir = NLLB_BASE / dir_name
            vals = []
            for pair in PAIRS:
                vals.append(overall_metric_from_results(cond_dir, pair, metric))
            rows.append((tau_disp, vals))
        out_path = TABLES_DIR / f"balance_{suffix}.tex"
        out_path.write_text(render_balance_table(
            label=f"tab:balance_{suffix}",
            caption=("Domain-balance ablation on NLLB-1.3B fine-tuned on "
                     f"AfriScience-MT ({label_metric}). $\\tau{{=}}1$ uses the "
                     "natural domain proportions; $\\tau{=}2$ applies sqrt "
                     "smoothing; $\\tau{=}\\infty$ samples uniformly across "
                     "all 11 domains. Best per column in bold."),
            blocks=[("Domain balance", rows)],
        ))
        print(f"  ✓ {out_path.relative_to(ROOT)}")


# ---------------- figure builders ----------------

def gather_per_domain_chrf():
    """Return {tau_label: {domain: avg chrF over the 12 pairs}}."""
    out = {}
    for tau_short, dir_name, tau_disp in CONDITIONS:
        cond_dir = NLLB_BASE / dir_name
        per_dom = defaultdict(list)
        for pair in PAIRS:
            src, tgt = pair.split("-")
            d = per_domain_chrf(cond_dir / pair / "predictions.json", src, tgt)
            for cat, v in d.items():
                per_dom[cat].append(v)
        out[tau_disp] = {cat: sum(vs) / len(vs) for cat, vs in per_dom.items()}
    return out


def fig_balance_heatmap():
    """3-row × 11-col heatmap: rows=τ, cols=domains (ordered by N_d), cells=avg chrF."""
    pdom = gather_per_domain_chrf()

    domains_sorted = DOMAIN_KEYS  # already ordered by N_d in the DOMAINS list
    domain_displays = [d[1] for d in DOMAINS]

    mat = np.array([
        [pdom[tau].get(dom, np.nan) for dom in domains_sorted]
        for _, _, tau in CONDITIONS
    ])
    vmin, vmax = float(np.nanmin(mat)), float(np.nanmax(mat))

    fig, ax = plt.subplots(figsize=(7.5, 1.9))
    im = ax.imshow(mat, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(domains_sorted)))
    ax.set_xticklabels([f"{lbl}\n(N={n})"
                        for (_, lbl, n) in DOMAINS], fontsize=6.5)
    ax.set_yticks(np.arange(len(CONDITIONS)))
    ax.set_yticklabels([tau for _, _, tau in CONDITIONS], fontsize=8)
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            color = "white" if (v - vmin) / (vmax - vmin + 1e-9) > 0.55 else "black"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color=color, fontsize=6.5)
    fig.tight_layout()
    out = FIG_DIR / "balance_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.relative_to(ROOT)}")
    return pdom


def fig_balance_delta(pdom):
    """Per-domain Δ chrF = τ=∞ − τ=1, ordered by N_d.

    The story: low-N domains gain from balancing; high-N domains lose.
    """
    deltas = []
    for dom in DOMAIN_KEYS:
        d_inf = pdom[r"$\tau{=}\infty$"].get(dom)
        d_1   = pdom[r"$\tau{=}1$"].get(dom)
        if d_inf is None or d_1 is None:
            deltas.append(np.nan)
        else:
            deltas.append(d_inf - d_1)

    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    x = np.arange(len(DOMAIN_KEYS))
    colors = ["#3182bd" if d >= 0 else "#e6550d" for d in deltas]
    ax.bar(x, deltas, color=colors, edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl}\n(N={n})" for (_, lbl, n) in DOMAINS],
                       fontsize=6.5)
    ax.set_ylabel(r"$\Delta$ ChrF $(\tau{=}\infty - \tau{=}1)$", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out = FIG_DIR / "balance_delta_by_size.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.relative_to(ROOT)}")


def per_domain_chrf_for_pair(cond: str, pair: str) -> dict[str, float]:
    """Per-domain chrF for one (condition, pair). Reads stored per_domain when
    available (balance_t2, balance_tinf); otherwise computes on the fly from
    predictions.json + the test split's category column (the τ=1 dcs case)."""
    base = NLLB_BASE / cond / pair
    res = base / "results.json"
    if res.exists():
        stored = json.load(open(res)).get("metrics", {}).get("per_domain")
        if stored:
            return {dom: stored[dom]["chrf"] for dom in stored}
    src, tgt = pair.split("-")
    return per_domain_chrf(base / "predictions.json", src, tgt)


# Train-set sentence counts per domain (for the "#sents" column).
DOMAIN_TRAIN_N = {dom: n for dom, _, n in DOMAINS}


def render_domain_x_lang_table(*, label: str, caption: str,
                               base: dict[str, dict[str, float]],
                               cmp:  dict[str, dict[str, float]],
                               base_overall: dict[str, float],
                               cmp_overall: dict[str, float]):
    """Table-7-style: rows = domains (+ all-domains row), columns = 6 eng-X
    targets. Each cell prints `score (Δ)` where Δ = cmp − base.

    Δ rendered with `\\textcolor{red|blue}{(±X)}`."""
    targets = LANGS  # eng→X only
    head_row = " & ".join([r"\texttt{" + LANG_DISP[l].lower() + "}" for l in targets])
    lines = [
        f"% ---- {label.split(':',1)[1]} ----",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"\textbf{domain} & \textbf{\#sents} & \multicolumn{6}{c}{\textbf{domain-level ChrF}} \\",
        r"\cmidrule(lr){3-8}",
        f"  &       & {head_row} \\\\",
        r"\midrule",
    ]

    def cell(score: float | None, delta: float | None) -> str:
        if score is None:
            return "--"
        if delta is None:
            return f"{score:.1f}"
        sign = "+" if delta >= 0 else ""
        color = "blue" if delta > 0 else "red"
        return f"{score:.1f} \\textcolor{{{color}}}{{({sign}{delta:.1f})}}"

    # All-domains row first
    cells = []
    for tgt in targets:
        pair = f"eng-{tgt}"
        cmp_v = cmp_overall.get(pair)
        base_v = base_overall.get(pair)
        delta = (cmp_v - base_v) if (cmp_v is not None and base_v is not None) else None
        cells.append(cell(cmp_v, delta))
    lines.append(r"\textit{all-domains} & 5{,}792 & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")

    for dom_key, dom_disp, dom_n in DOMAINS:
        cells = []
        for tgt in targets:
            pair = f"eng-{tgt}"
            cmp_v = cmp.get(pair, {}).get(dom_key)
            base_v = base.get(pair, {}).get(dom_key)
            delta = (cmp_v - base_v) if (cmp_v is not None and base_v is not None) else None
            cells.append(cell(cmp_v, delta))
        # Display name; lowercase to match existing draft
        disp = dom_key  # exact original casing — matches earlier doc style
        lines.append(f"{disp} & {dom_n} & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def build_domain_x_lang_table():
    """One Table 7-style table per (compared-condition × direction).

    For now: τ=∞ vs τ=1 (the cleanest balance ablation contrast),
    Eng→xxx direction, ChrF only (SSA-COMET would need GPU rescore on
    the τ=1 baseline since it lacks per_domain in results.json).
    """
    base, cmp = {}, {}
    base_overall, cmp_overall = {}, {}
    for tgt in LANGS:
        pair = f"eng-{tgt}"
        base[pair] = per_domain_chrf_for_pair("dcs", pair)
        cmp[pair]  = per_domain_chrf_for_pair("balance_tinf", pair)
        # overalls from results.json
        for cond, sink in [("dcs", base_overall), ("balance_tinf", cmp_overall)]:
            r = NLLB_BASE / cond / pair / "results.json"
            if r.exists():
                v = json.load(open(r)).get("metrics", {}).get("chrf")
                if v is not None:
                    sink[pair] = v

    out = TABLES_DIR / "balance_domain_x_lang_eng_xxx.tex"
    out.write_text(render_domain_x_lang_table(
        label="tab:balance_domain_eng_xxx",
        caption=("Domain-level ChrF for NLLB-1.3B on Eng$\\to$xxx with "
                 "uniform-temperature sampling ($\\tau{=}\\infty$). Numbers in "
                 "parentheses are the change relative to the natural-distribution "
                 "baseline ($\\tau{=}1$); blue = improvement, red = decrease. "
                 "Total training volume is identical across the two conditions "
                 "(5\\,792 sentences) --- only the per-example sampling weight "
                 "changes."),
        base=base, cmp=cmp,
        base_overall=base_overall, cmp_overall=cmp_overall,
    ))
    print(f"  ✓ {out.relative_to(ROOT)}")


DOMAIN_ORDER_SPIDER = [
    "chemistry", "biology", "biochemistry", "agriculture", "statistics",
    "sociology", "Indigenous knowledge", "health", "geography",
    "engineering", "computer science",
]
DOMAIN_SHORT_SPIDER = {
    "chemistry": "Chem.", "biology": "Biology",
    "biochemistry": "Biochem.", "agriculture": "Agric.",
    "statistics": "Stat.", "sociology": "Soc.",
    "Indigenous knowledge": "Indig.",
    "health": "Health", "geography": "Geog.",
    "engineering": "Eng.", "computer science": "CS",
}


def _draw_single_spider(ax, pair: str, systems, *, y_top: float):
    """Render one radar onto an existing polar Axes. No legend (drop the
    color-key into the LaTeX caption to avoid overlap with domain labels)."""
    angles = np.linspace(0, 2 * np.pi, len(DOMAIN_ORDER_SPIDER),
                         endpoint=False).tolist()
    angles += angles[:1]
    for sys_name, getter, line_c, fill_c in systems:
        d = getter(pair)
        vals = [d.get(dom, 0.0) for dom in DOMAIN_ORDER_SPIDER]
        vals += vals[:1]
        ax.plot(angles, vals, color=line_c, linewidth=1.1, label=sys_name)
        ax.fill(angles, vals, color=fill_c, alpha=0.40)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([DOMAIN_SHORT_SPIDER[d] for d in DOMAIN_ORDER_SPIDER],
                       fontsize=6.5)
    ax.set_ylim(0, y_top)
    ax.set_yticks([20, 40, 60])
    ax.set_yticklabels(["20", "40", "60"], fontsize=5, color="gray")
    ax.tick_params(pad=1)
    ax.grid(linestyle=":", alpha=0.5, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def _global_y_top(systems) -> float:
    vals = []
    for lang in LANGS:
        for _, getter, _, _ in systems:
            d = getter(f"eng-{lang}")
            vals.extend(d.values())
    return int(np.ceil((max(vals) + 5) / 10.0)) * 10  # round up to next 10


def _per_pair_spider_subdir(*, subdir_name: str, systems):
    """Produce one PDF per Eng→X pair into figures/new/<subdir_name>/."""
    out_dir = FIG_DIR / subdir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    y_top = _global_y_top(systems)
    written = []
    for i, lang in enumerate(LANGS):
        pair = f"eng-{lang}"
        fig = plt.figure(figsize=(2.6, 2.6))
        ax = fig.add_subplot(111, polar=True)
        # Show legend on first panel only — author can include it via
        # \subfigure[showing legend] in LaTeX or use the standalone version.
        _draw_single_spider(ax, pair, systems, y_top=y_top,
                            show_legend=(i == 0))
        fig.tight_layout()
        out = out_dir / f"{pair}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    print(f"  ✓ {len(written)} PDFs in {out_dir.relative_to(ROOT)}/")
    return written


def fig_balance_spider():
    """Per-pair balance spiders (τ=1 vs τ=∞)."""
    systems = [
        (r"$\tau{=}1$",
         lambda pair: per_domain_chrf_for_pair("dcs", pair),
         "#3b67c4", "#a9bde6"),
        (r"$\tau{=}\infty$",
         lambda pair: per_domain_chrf_for_pair("balance_tinf", pair),
         "#d24a4a", "#f5c1c1"),
    ]
    _per_pair_spider_subdir(subdir_name="spider_balance", systems=systems)


def main():
    print("Building §4 balance ablation assets...")
    print()
    print("[tables]")
    build_balance_tables()
    build_domain_x_lang_table()
    print()
    print("[figures]")
    pdom = fig_balance_heatmap()
    fig_balance_delta(pdom)


if __name__ == "__main__":
    main()
