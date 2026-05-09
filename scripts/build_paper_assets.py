"""Build paper-ready tables and figures for the new ACL 2026 draft.

Reads experiments/summary.csv and writes:
  tables/paper/headline_comet.tex
  tables/paper/headline_chrf.tex
  tables/paper/mafand.tex
  tables/paper/lora_rank.tex
  tables/paper/bilingual_vs_multilingual.tex
  tables/paper/open_vs_closed.tex
  _ACL2026__Decolonizing_Science_in_Africa_through_Text_Translation_/figures/zs_vs_ft_delta.pdf
  _ACL2026__Decolonizing_Science_in_Africa_through_Text_Translation_/figures/lora_rank_curve.pdf

All values are SSA-COMET × 100 unless the table says chrF.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "experiments" / "summary.csv"
TABLES_DIR = ROOT / "tables" / "paper"
FIG_DIR = ROOT / "_ACL2026__Decolonizing_Science_in_Africa_through_Text_Translation_" / "figures" / "new"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

LANGS = ["amh", "hau", "lug", "nso", "yor", "zul"]
PAIRS_E2X = [f"eng-{l}" for l in LANGS]
PAIRS_X2E = [f"{l}-eng" for l in LANGS]
ALL_PAIRS = PAIRS_E2X + PAIRS_X2E

# Display names
LANG_DISP = {"amh": "amh", "hau": "hau", "lug": "lug",
             "nso": "nso", "yor": "yor", "zul": "zul"}

MODEL_DISP = {
    "m2m100_418m": "M2M100-418M",
    "m2m100_1_2b": "M2M100-1.2B",
    "nllb_200_distilled_600m": "NLLB-600M",
    "nllb_200_1_3b": "NLLB-1.3B",
    "llama3_8b": "LLaMA-3.1-8B",
    "afriquellama_8b": "AfriqueLlama",
    "afriqueqwen_8b": "AfriqueQwen",
    "gemma2_9b_it": "Gemma2-9B",
    "tiny_aya_global": "Tiny-Aya-G",
    "tiny_aya_earth": "Tiny-Aya-E",
    "translategemma_12b": "TranslateGemma",
    "gpt4o": "GPT-4o",
    "gemini_15_flash": "Gemini-1.5",
}

# ---------------- IO ----------------

def load_rows() -> list[dict]:
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            for k in ("bleu", "chrf", "ssa_comet"):
                v = r.get(k)
                r[k] = float(v) if v not in (None, "", "None") else None
            rows.append(r)
    return rows


def is_ablation(r: dict) -> bool:
    return str(r.get("is_ablation", "")).strip().lower() in {"true", "1"}


def metric_val(r: dict, metric: str) -> float | None:
    v = r.get(metric)
    if v is None:
        return None
    return v * 100 if metric == "ssa_comet" else v


# ---------------- LaTeX helpers ----------------

def fmt(v, decimals=1):
    return "--" if v is None else f"{v:.{decimals}f}"


def best_idx(values):
    """Return set of column indices that hold the per-column max within a block."""
    return None  # placeholder; computed inline


def render_pair_table(
    *,
    caption: str,
    label: str,
    blocks: list[tuple[str, list[tuple[str, list[float | None]]]]],
    decimals: int = 1,
    use_table_star: bool = True,
    note: str | None = None,
) -> str:
    """Generic 12-pair table renderer with grouped blocks.

    blocks = [(block_label, [(row_label, values_for_12_pairs), ...]), ...]
    """
    cols = ALL_PAIRS
    n = len(cols)
    spec = "ll" + "r" * n
    head_e2x = "\\multicolumn{6}{c}{\\textbf{Eng$\\to$xxx}}"
    head_x2e = "\\multicolumn{6}{c}{\\textbf{xxx$\\to$Eng}}"
    sub = " & ".join("\\texttt{" + LANG_DISP[c.split("-")[1] if c.startswith("eng-") else c.split("-")[0]] + "}" for c in cols)

    # Per-column best across all rows in the table
    all_vals = [v for _, rows_ in blocks for _, vs in rows_ for v in vs]
    col_best = []
    for ci in range(n):
        col_vals = [r[1][ci] for _, rows_ in blocks for r in rows_ if r[1][ci] is not None]
        col_best.append(max(col_vals) if col_vals else None)

    env = "table*" if use_table_star else "table"
    out = [
        f"\\begin{{{env}}}[t!]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{spec}}}",
        "\\toprule",
        f" & & {head_e2x} & {head_x2e} \\\\",
        "\\cmidrule(lr){3-8}\\cmidrule(lr){9-14}",
        f" & \\textbf{{Model}} & {sub} \\\\",
        "\\midrule",
    ]
    for bi, (block_label, rows_) in enumerate(blocks):
        if bi > 0:
            out.append("\\midrule")
        nrows = len(rows_)
        for ri, (row_label, vals) in enumerate(rows_):
            cells = []
            for ci, v in enumerate(vals):
                s = fmt(v, decimals)
                if v is not None and col_best[ci] is not None and abs(v - col_best[ci]) < 1e-9:
                    s = f"\\textbf{{{s}}}"
                cells.append(s)
            head = (
                f"\\multirow{{{nrows}}}{{*}}{{\\textit{{{block_label}}}}}"
                if ri == 0 else ""
            )
            out.append(f"{head} & {row_label} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    if note:
        out.append(f"\\\\[2pt]\\footnotesize{{{note}}}")
    out.append(f"\\end{{{env}}}")
    return "\n".join(out)


def render_two_col_table(
    *,
    caption: str,
    label: str,
    pair_subset: list[str],          # e.g. PAIRS_E2X + PAIRS_X2E
    blocks: list[tuple[str, list[tuple[str, list[float | None]]]]],
    decimals: int = 1,
    note: str | None = None,
    use_table_star: bool = True,
) -> str:
    """Like render_pair_table but for arbitrary pair subset (e.g. 4 mafand langs × 2)."""
    n = len(pair_subset)
    spec = "ll" + "r" * n
    halves = []
    cur = []
    for p in pair_subset:
        cur.append(p)
    # split into eng-xxx and xxx-eng halves (for header)
    e2x = [p for p in pair_subset if p.startswith("eng-")]
    x2e = [p for p in pair_subset if p.endswith("-eng")]
    sub = " & ".join("\\texttt{" + LANG_DISP[(p.split("-")[1] if p.startswith("eng-") else p.split("-")[0])] + "}" for p in pair_subset)
    head_a = f"\\multicolumn{{{len(e2x)}}}{{c}}{{\\textbf{{Eng$\\to$xxx}}}}"
    head_b = f"\\multicolumn{{{len(x2e)}}}{{c}}{{\\textbf{{xxx$\\to$Eng}}}}"

    col_best = []
    for ci in range(n):
        col_vals = [r[1][ci] for _, rs in blocks for r in rs if r[1][ci] is not None]
        col_best.append(max(col_vals) if col_vals else None)

    env = "table*" if use_table_star else "table"
    out = [
        f"\\begin{{{env}}}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{spec}}}",
        "\\toprule",
        f" & & {head_a} & {head_b} \\\\",
        f"\\cmidrule(lr){{3-{2+len(e2x)}}}\\cmidrule(lr){{{3+len(e2x)}-{2+n}}}",
        f" & \\textbf{{Model}} & {sub} \\\\",
        "\\midrule",
    ]
    for bi, (block_label, rs) in enumerate(blocks):
        if bi > 0:
            out.append("\\midrule")
        for ri, (row_label, vals) in enumerate(rs):
            cells = []
            for ci, v in enumerate(vals):
                s = fmt(v, decimals)
                if v is not None and col_best[ci] is not None and abs(v - col_best[ci]) < 1e-9:
                    s = f"\\textbf{{{s}}}"
                cells.append(s)
            head = (
                f"\\multirow{{{len(rs)}}}{{*}}{{\\textit{{{block_label}}}}}"
                if ri == 0 else ""
            )
            out.append(f"{head} & {row_label} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    if note:
        out.append(f"\\\\[2pt]\\footnotesize{{{note}}}")
    out.append(f"\\end{{{env}}}")
    return "\n".join(out)


# ---------------- table builders ----------------

def get(rows, *, metric, **filters):
    """Return {pair: value*100 (comet) | value} for rows matching all filters."""
    out = {}
    for r in rows:
        ok = True
        for k, v in filters.items():
            if v is None:
                if r.get(k) not in ("", None):
                    ok = False; break
            elif callable(v):
                if not v(r.get(k)):
                    ok = False; break
            else:
                if str(r.get(k, "")) != str(v):
                    ok = False; break
            if not ok:
                break
        if ok:
            mv = metric_val(r, metric)
            if mv is not None:
                out[r["lang_pair"]] = mv
    return out


def vec(d: dict, pairs=ALL_PAIRS):
    return [d.get(p) for p in pairs]


def build_headline(rows, metric: str) -> str:
    metric_label = "SSA-COMET" if metric == "ssa_comet" else "ChrF"

    # Block A: zero-shot seq2seq (4 models)
    zs_seq = []
    for m in ["m2m100_418m", "m2m100_1_2b", "nllb_200_distilled_600m", "nllb_200_1_3b"]:
        d = get(rows, metric=metric, experiment_type="zero_shot", model_short=m)
        zs_seq.append((MODEL_DISP[m], vec(d)))

    # Block B: zero-shot LLMs (best 3-4 by avg)
    zs_llm_models = ["llama3_8b", "gemma2_9b_it", "afriquellama_8b",
                     "tiny_aya_global", "translategemma_12b"]
    zs_llm = []
    for m in zs_llm_models:
        d = get(rows, metric=metric, experiment_type="llm_zero_shot",
                model_short=m, is_ablation="False")
        zs_llm.append((MODEL_DISP[m], vec(d)))

    # Block C: fine-tuned seq2seq
    ft_seq = []
    for m in ["m2m100_418m", "m2m100_1_2b", "nllb_200_distilled_600m", "nllb_200_1_3b"]:
        d = get(rows, metric=metric, experiment_type="seq2seq_finetune",
                model_short=m, dataset="dcs", is_ablation="False")
        ft_seq.append((MODEL_DISP[m], vec(d)))

    # Block D: LoRA r64 (per-pair)
    lora_models = ["llama3_8b", "afriquellama_8b", "tiny_aya_global", "tiny_aya_earth",
                   "afriqueqwen_8b", "gemma2_9b_it"]
    ft_lora = []
    for m in lora_models:
        d = get(rows, metric=metric, experiment_type="lora_finetune",
                model_short=m, lora_rank="64", is_ablation="False")
        if any(v is not None for v in vec(d)):
            ft_lora.append((f"{MODEL_DISP[m]} (r=64)", vec(d)))

    # Block E: closed APIs zero-shot (t0)
    closed = []
    for m in ["gpt4o", "gemini_15_flash"]:
        d = get(rows, metric=metric, experiment_type="llm_zero_shot",
                model_short=m, is_ablation="False")
        closed.append((MODEL_DISP[m], vec(d)))

    blocks = [
        ("Zero-shot Seq2Seq", zs_seq),
        ("Zero-shot LLM", zs_llm),
        ("Fine-tuned Seq2Seq", ft_seq),
        ("LoRA Fine-tune", ft_lora),
        ("Closed-source LLM (zero-shot)", closed),
    ]
    return render_pair_table(
        caption=(f"Sentence-level {metric_label} on AfriScience-MT across all 12 "
                 "directional pairs. Best value per column in bold."),
        label=f"tab:headline_{metric}",
        blocks=blocks,
        decimals=1,
    )


def build_mafand(rows) -> str:
    """nllb_200_1_3b: mafand vs dcs vs mafand_pp on the 4 mafand langs × 2 directions."""
    maf_langs = ["hau", "lug", "yor", "zul"]
    pairs = [f"eng-{l}" for l in maf_langs] + [f"{l}-eng" for l in maf_langs]

    def block(metric):
        rows_out = []
        d_maf = get(rows, metric=metric, model_short="nllb_200_1_3b",
                    experiment_type="seq2seq_finetune", dataset="mafand")
        d_dcs = get(rows, metric=metric, model_short="nllb_200_1_3b",
                    experiment_type="seq2seq_finetune", dataset="dcs", is_ablation="False")
        d_pp = get(rows, metric=metric, model_short="nllb_200_1_3b",
                   experiment_type="seq2seq_finetune", dataset="mafand_pp")
        rows_out.append(("MAFAND", [d_maf.get(p) for p in pairs]))
        rows_out.append(("AS-MT", [d_dcs.get(p) for p in pairs]))
        rows_out.append(("AS-MT + MAFAND", [d_pp.get(p) for p in pairs]))
        return rows_out

    blocks = [
        ("ChrF", block("chrf")),
        ("SSA-COMET", block("ssa_comet")),
    ]
    return render_two_col_table(
        caption=("Cross-domain transfer: MAFAND-MT (general news) vs. "
                 "AfriScience-MT (in-domain) vs. their union, fine-tuning "
                 "NLLB-1.3B. Best value per column in bold."),
        label="tab:mafand_compare",
        pair_subset=pairs,
        blocks=blocks,
        decimals=1,
        use_table_star=True,
    )


def build_lora_rank(rows) -> str:
    """llama3_8b LoRA rank ablation r4..r128, both metrics."""
    ranks = ["4", "8", "16", "32", "64", "128"]

    def block(metric):
        rs = []
        for r in ranks:
            d = get(rows, metric=metric, model_short="llama3_8b",
                    experiment_type="lora_finetune", lora_rank=r)
            rs.append((f"$r{{=}}{r}$", vec(d)))
        return rs

    blocks = [("ChrF", block("chrf")), ("SSA-COMET", block("ssa_comet"))]
    note = (
        r"\textbf{Note.} Ranks $r{\in}\{8,16,32,128\}$ were trained with the "
        r"\texttt{unsloth/Llama-3.1-8B-Instruct (4-bit)} base; "
        r"$r{\in}\{4,64\}$ used the vanilla "
        r"\texttt{meta-llama/Llama-3.1-8B-Instruct} base. The shape of the "
        r"rank curve is comparable; absolute levels across the two bases are not."
    )
    return render_pair_table(
        caption=("LoRA rank ablation on LLaMA-3.1-8B fine-tuned on AfriScience-MT. "
                 "Best per column in bold."),
        label="tab:lora_rank",
        blocks=blocks,
        decimals=1,
        note=note,
    )


def _checkpoint_from_path(p: str | None) -> str | None:
    if not p:
        return None
    m = re.search(r"multilingual/(\d+k)_updates", p)
    return m.group(1) if m else None


def build_bilingual_vs_multilingual(rows) -> str:
    """NLLB-1.3B per-pair vs NLLB-600M multilingual (400k); LoRA r64 per-pair vs r64 mmt."""
    # NLLB rows
    nllb_per = get(rows, metric="ssa_comet", model_short="nllb_200_1_3b",
                   experiment_type="seq2seq_finetune", dataset="dcs", is_ablation="False")
    nllb_per_chrf = get(rows, metric="chrf", model_short="nllb_200_1_3b",
                        experiment_type="seq2seq_finetune", dataset="dcs", is_ablation="False")
    # multilingual_finetune nllb_200_distilled_600m, 400k
    nllb_mmt = {}
    nllb_mmt_chrf = {}
    for r in rows:
        if (r["experiment_type"] == "multilingual_finetune"
                and r["model_short"] == "nllb_200_distilled_600m"
                and _checkpoint_from_path(r.get("original_path")) == "400k"):
            v = metric_val(r, "ssa_comet")
            if v is not None:
                nllb_mmt[r["lang_pair"]] = v
            cv = metric_val(r, "chrf")
            if cv is not None:
                nllb_mmt_chrf[r["lang_pair"]] = cv

    # LoRA llama3_8b r64 per-pair
    lora_per = get(rows, metric="ssa_comet", model_short="llama3_8b",
                   experiment_type="lora_finetune", lora_rank="64", is_ablation="False")
    lora_per_chrf = get(rows, metric="chrf", model_short="llama3_8b",
                        experiment_type="lora_finetune", lora_rank="64", is_ablation="False")
    lora_mmt = get(rows, metric="ssa_comet", model_short="llama3_8b",
                   experiment_type="multilingual_lora", lora_rank="64")
    lora_mmt_chrf = get(rows, metric="chrf", model_short="llama3_8b",
                        experiment_type="multilingual_lora", lora_rank="64")

    blocks = [
        ("ChrF", [
            ("NLLB-1.3B (per-pair)",       vec(nllb_per_chrf)),
            ("NLLB-600M (multilingual, 400k)", vec(nllb_mmt_chrf)),
            ("LLaMA-3.1-8B LoRA $r{=}64$ (per-pair)", vec(lora_per_chrf)),
            ("LLaMA-3.1-8B LoRA $r{=}64$ (multilingual)", vec(lora_mmt_chrf)),
        ]),
        ("SSA-COMET", [
            ("NLLB-1.3B (per-pair)",       vec(nllb_per)),
            ("NLLB-600M (multilingual, 400k)", vec(nllb_mmt)),
            ("LLaMA-3.1-8B LoRA $r{=}64$ (per-pair)", vec(lora_per)),
            ("LLaMA-3.1-8B LoRA $r{=}64$ (multilingual)", vec(lora_mmt)),
        ]),
    ]
    note = (r"\textbf{Note.} Multilingual LoRA (mmt) was trained on "
            r"\texttt{eng}$\to$\texttt{xxx} only, so reverse-direction cells are intentionally empty.")
    return render_pair_table(
        caption=("Bilingual (per-pair) vs. multilingual training on the same model. "
                 "Best per column in bold."),
        label="tab:bil_vs_multi",
        blocks=blocks,
        decimals=1,
        note=note,
    )


def build_open_vs_closed(rows) -> str:
    """Closed (gpt4o, gemini) at t0 zero-shot and 5shot vs best open at zero-shot/5shot."""

    def best_open_zero(metric):
        cands = ["llama3_8b", "gemma2_9b_it", "afriquellama_8b",
                 "tiny_aya_global", "translategemma_12b"]
        d_best = {}
        for p in ALL_PAIRS:
            best = None
            for m in cands:
                d = get(rows, metric=metric, model_short=m,
                        experiment_type="llm_zero_shot", is_ablation="False")
                v = d.get(p)
                if v is not None and (best is None or v > best):
                    best = v
            d_best[p] = best
        return d_best

    def best_open_icl(metric):
        cands = ["llama3_8b", "gemma2_9b_it", "afriquellama_8b",
                 "tiny_aya_global", "translategemma_12b"]
        d_best = {}
        for p in ALL_PAIRS:
            best = None
            for m in cands:
                d = get(rows, metric=metric, model_short=m,
                        experiment_type="llm_icl", prompt_strategy="icl_5shot",
                        is_ablation="False")
                v = d.get(p)
                if v is not None and (best is None or v > best):
                    best = v
            d_best[p] = best
        return d_best

    def closed_zero(metric, m):
        return get(rows, metric=metric, model_short=m,
                   experiment_type="llm_zero_shot", is_ablation="False")

    def closed_icl(metric, m):
        # GPT-4o eng→X is only available as icl_5shot_semantic; Gemini has both.
        # Prefer icl_5shot, fall back to icl_5shot_semantic per pair.
        primary = get(rows, metric=metric, model_short=m,
                      experiment_type="llm_icl", prompt_strategy="icl_5shot",
                      is_ablation="False")
        sem = get(rows, metric=metric, model_short=m,
                  experiment_type="llm_icl", prompt_strategy="icl_5shot_semantic",
                  is_ablation="False")
        return {p: primary.get(p, sem.get(p)) for p in ALL_PAIRS}

    def block(metric):
        return [
            ("Best open (zero-shot)", vec(best_open_zero(metric))),
            ("Best open (5-shot ICL)", vec(best_open_icl(metric))),
            ("Gemini-1.5-Flash (zero-shot)", vec(closed_zero(metric, "gemini_15_flash"))),
            ("Gemini-1.5-Flash (5-shot ICL)", vec(closed_icl(metric, "gemini_15_flash"))),
            ("GPT-4o (zero-shot)", vec(closed_zero(metric, "gpt4o"))),
            ("GPT-4o (5-shot ICL)", vec(closed_icl(metric, "gpt4o"))),
        ]

    blocks = [("ChrF", block("chrf")), ("SSA-COMET", block("ssa_comet"))]
    return render_pair_table(
        caption=("Open vs. closed-source LLMs. ``Best open'' is the per-cell maximum "
                 "across LLaMA-3.1-8B, Gemma2-9B, AfriqueLlama-8B, Tiny-Aya-G, and "
                 "TranslateGemma-12B."),
        label="tab:open_vs_closed",
        blocks=blocks,
        decimals=1,
    )


def build_pretrained_vs_finetuned(rows) -> str:
    """Side-by-side zero-shot vs fine-tuned for each NLLB / M2M-100 size, both metrics."""
    models = ["m2m100_418m", "m2m100_1_2b", "nllb_200_distilled_600m", "nllb_200_1_3b"]

    def block(metric):
        rs = []
        for m in models:
            d_zs = get(rows, metric=metric, experiment_type="zero_shot", model_short=m)
            d_ft = get(rows, metric=metric, experiment_type="seq2seq_finetune",
                       model_short=m, dataset="dcs", is_ablation="False")
            rs.append((f"{MODEL_DISP[m]} (zero-shot)", vec(d_zs)))
            rs.append((f"{MODEL_DISP[m]} (fine-tuned)", vec(d_ft)))
        return rs

    blocks = [("ChrF", block("chrf")), ("SSA-COMET", block("ssa_comet"))]
    return render_pair_table(
        caption=("Effect of fine-tuning on AfriScience-MT for seq2seq models. "
                 "Best per column in bold."),
        label="tab:pretrained_vs_finetuned",
        blocks=blocks,
        decimals=1,
    )


# ---------------- plots ----------------

def plot_zs_vs_ft_delta(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Average per-language SSA-COMET (over both directions) — zero-shot vs fine-tune
    def avg_by_lang(experiment_type, **kw):
        out = {l: [] for l in LANGS}
        for r in rows:
            if r["experiment_type"] != experiment_type:
                continue
            ok = True
            for k, v in kw.items():
                if str(r.get(k, "")) != str(v):
                    ok = False; break
            if not ok:
                continue
            mv = metric_val(r, "ssa_comet")
            if mv is None:
                continue
            non_eng = r["target_lang"] if r["source_lang"] == "eng" else r["source_lang"]
            if non_eng in out:
                out[non_eng].append(mv)
        return {l: (sum(v)/len(v) if v else None) for l, v in out.items()}

    nllb_zs = avg_by_lang("zero_shot", model_short="nllb_200_1_3b")
    nllb_ft = avg_by_lang("seq2seq_finetune", model_short="nllb_200_1_3b",
                          dataset="dcs", is_ablation="False")
    llama_zs = avg_by_lang("llm_zero_shot", model_short="llama3_8b", is_ablation="False")
    llama_ft = avg_by_lang("lora_finetune", model_short="llama3_8b",
                           lora_rank="64", is_ablation="False")

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    x = np.arange(len(LANGS))
    w = 0.2
    bars = [
        ("NLLB-1.3B zero-shot", [nllb_zs[l] for l in LANGS], "#9ecae1"),
        ("NLLB-1.3B fine-tune", [nllb_ft[l] for l in LANGS], "#3182bd"),
        ("LLaMA-3.1-8B zero-shot", [llama_zs[l] for l in LANGS], "#fdae6b"),
        ("LLaMA-3.1-8B LoRA $r{=}64$", [llama_ft[l] for l in LANGS], "#e6550d"),
    ]
    for i, (label, vals, c) in enumerate(bars):
        offset = (i - 1.5) * w
        ax.bar(x + offset, [v if v is not None else 0 for v in vals],
               width=w, label=label, color=c, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(LANGS)
    ax.set_ylabel("SSA-COMET (avg over both directions)")
    ax.set_ylim(0, 80)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18),
              ncol=4, fontsize=7, frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out = FIG_DIR / "zs_vs_ft_delta.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_lora_rank_curve(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ranks = [4, 8, 16, 32, 64, 128]

    def avg_dir(direction):  # 'eng-x' or 'x-eng'
        out = {}
        for r_ in ranks:
            d = get(rows, metric="ssa_comet", model_short="llama3_8b",
                    experiment_type="lora_finetune", lora_rank=str(r_))
            vs = []
            for p, v in d.items():
                if direction == "eng-x" and p.startswith("eng-"):
                    vs.append(v)
                elif direction == "x-eng" and p.endswith("-eng"):
                    vs.append(v)
            out[r_] = sum(vs) / len(vs) if vs else None
        return out

    e2x = avg_dir("eng-x")
    x2e = avg_dir("x-eng")

    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    xs = ranks
    ax.plot(xs, [e2x[r] for r in xs], marker="o", label="Eng$\\to$xxx", color="#3182bd")
    ax.plot(xs, [x2e[r] for r in xs], marker="s", label="xxx$\\to$Eng", color="#e6550d")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs); ax.set_xticklabels([f"r={r}" for r in xs], fontsize=8)
    ax.set_ylabel("SSA-COMET (avg)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(linestyle=":", alpha=0.5)
    # Caveat
    # Title intentionally omitted — LaTeX caption carries it
    fig.tight_layout()
    out = FIG_DIR / "lora_rank_curve.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------- main ----------------

def main():
    rows = load_rows()
    written = []

    def write(name, content):
        path = TABLES_DIR / name
        path.write_text(content)
        written.append(path)

    write("headline_comet.tex", build_headline(rows, "ssa_comet"))
    write("headline_chrf.tex", build_headline(rows, "chrf"))
    write("pretrained_vs_finetuned.tex", build_pretrained_vs_finetuned(rows))
    write("mafand.tex", build_mafand(rows))
    write("lora_rank.tex", build_lora_rank(rows))
    write("bilingual_vs_multilingual.tex", build_bilingual_vs_multilingual(rows))
    write("open_vs_closed.tex", build_open_vs_closed(rows))

    f1 = plot_zs_vs_ft_delta(rows)
    f2 = plot_lora_rank_curve(rows)

    print("Tables written:")
    for p in written:
        print(f"  {p.relative_to(ROOT)}")
    print("Figures written:")
    print(f"  {f1.relative_to(ROOT)}")
    print(f"  {f2.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
