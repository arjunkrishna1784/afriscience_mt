"""Build paper tables in the exact ``new_results.tex`` format.

Eight tables, four pairs of (chrF, SSA-COMET):

    seq2seq        zero-shot            vs. fine-tuned
    llm_sentence   zero-shot            vs. 5-shot ICL
    lora           LoRA r=4             vs. LoRA r=64
    doc            zero-shot doc        vs. 1-shot doc

Each table follows the structure of
``_ACL2026__.../new_results.tex`` exactly (column spec, headers,
``\\multirow`` blocks, capitalised language headers, and ``\\textbf``-marked
column maxima).

Outputs land in ``tables/paper/`` with the same labels (``tab:seq2seq_chrf``
etc.) so they can be ``\\input{...}``-ed directly into the draft.

Run from repo root:
    python3 scripts/build_paper_tables.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "experiments" / "summary.csv"
OUT_DIR = ROOT / "tables" / "new"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LANGS = ["amh", "hau", "lug", "nso", "yor", "zul"]
LANG_DISP = {"amh": "Amh", "hau": "Hau", "lug": "Lug",
             "nso": "NSo", "yor": "Yor", "zul": "Zul"}
PAIRS_E2X = [f"eng-{l}" for l in LANGS]
PAIRS_X2E = [f"{l}-eng" for l in LANGS]
PAIRS = PAIRS_E2X + PAIRS_X2E

MODEL_DISP = {
    "m2m100_418m":            "M2M100\\textunderscore{}418M",
    "m2m100_1_2b":            "M2M100\\textunderscore{}1.2B",
    "nllb_200_distilled_600m":"NLLB\\textunderscore{}600M",
    "nllb_200_1_3b":          "NLLB\\textunderscore{}1.3B",
    "afriquellama_8b":        "AfriqueLlama-8B",
    "afriqueqwen_8b":         "AfriqueQwen-8B",
    "gemma2_9b_it":           "Gemma2-9B",
    "llama3_8b":              "Llama3-8B",
    "tiny_aya_earth":         "Tiny-Aya-E",
    "tiny_aya_global":        "Tiny-Aya-G",
    "translategemma_12b":     "TranslateGemma-12B",
    "gpt4o":                  "GPT-4o",
    "gemini_15_flash":        "Gemini-1.5",
    # Synthetic row label (no underlying model_short).
    "best_open":              "Best open",
}

SEQ2SEQ_MODELS_MAIN = ["m2m100_1_2b", "nllb_200_1_3b"]
SEQ2SEQ_MODELS_APPENDIX = ["m2m100_418m", "nllb_200_distilled_600m"]
# Backwards-compat alias for older callers
SEQ2SEQ_MODELS = SEQ2SEQ_MODELS_MAIN + SEQ2SEQ_MODELS_APPENDIX
LLM_MODELS = ["afriquellama_8b", "afriqueqwen_8b", "gemma2_9b_it",
              "llama3_8b", "tiny_aya_earth", "tiny_aya_global",
              "translategemma_12b"]
LORA_MODELS = ["afriquellama_8b", "afriqueqwen_8b", "gemma2_9b_it",
               "llama3_8b", "tiny_aya_earth", "tiny_aya_global"]


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


def first_with_pred(rows, *, metric, candidates) -> dict[str, float]:
    """For LLMs that store predictions under multiple prompt strategies (e.g.
    GPT-4o keeps eng→X under ``icl_5shot_semantic`` and X→eng under
    ``icl_5shot``), pick whichever variant is present per pair."""
    merged = {}
    for cand in candidates:
        d = lookup(rows, metric=metric, **cand)
        for p, v in d.items():
            merged.setdefault(p, v)
    return merged


# ---------------- rendering ----------------

def fmt(v):
    return "--" if v is None else f"{v:.1f}"


def render_grouped_table(*, label, caption, major_groups):
    """Render a table with two-level grouping plus a final Avg column.

    major_groups = [(major_name, sub_blocks), ...]
    where sub_blocks = [(sub_name | None, [(model_short, [12 floats|None])]), ...]

    The Avg column holds the per-model arithmetic mean of the 12 per-pair
    cells (None values skipped). It participates in the same bold/italic
    convention as the per-pair columns:
      \\textbf{X}  -- maximum within the major group
      \\emph{X}    -- maximum within the sub-block (when not also major)
      X plain     -- otherwise.

    Major groups separated by \\midrule\\midrule; sub-blocks by \\midrule.
    """
    def _avg(vals):
        nums = [v for v in vals if v is not None]
        return (sum(nums) / len(nums)) if nums else None

    # Augment every row with its per-row Avg as a 13th column for max-finding.
    def _augment(rs):
        return [(m, list(vals) + [_avg(vals)]) for m, vals in rs]

    aug_groups = [(name, [(sn, _augment(rs)) for sn, rs in subs])
                  for name, subs in major_groups]

    # Per-major-group and per-sub-block column max over 13 columns.
    major_col_max = []
    for _, subs in aug_groups:
        cm = []
        for ci in range(13):
            vals = [r[1][ci] for _sn, rs in subs for r in rs if r[1][ci] is not None]
            cm.append(max(vals) if vals else None)
        major_col_max.append(cm)

    sub_col_max = []
    for _, subs in aug_groups:
        sub_cm = []
        for _sn, rs in subs:
            cm = []
            for ci in range(13):
                vals = [r[1][ci] for r in rs if r[1][ci] is not None]
                cm.append(max(vals) if vals else None)
            sub_cm.append(cm)
        sub_col_max.append(sub_cm)

    lang_cols = [r"\texttt{" + l + "}" for l in LANGS] * 2
    lang_head = " & ".join(["  "] + lang_cols + [r"\textbf{Avg}"]) + r" \\"

    lines = [
        f"% ---- {label.split(':',1)[1]} ----",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{lrrrrrrrrrrrrr}",
        r"\toprule",
        r"\textbf{Model} & \multicolumn{6}{c}{\textbf{eng$\to$}} & \multicolumn{6}{c}{\textbf{$\to$eng}} & \\",
        r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}",
        lang_head,
    ]

    for mi, (major_name, subs) in enumerate(aug_groups):
        lines.append(r"\midrule\midrule" if mi > 0 else r"\midrule")
        lines.append(r"\multicolumn{14}{l}{\textit{" + major_name + r"}} \\")
        lines.append(r"\midrule")

        for si, (sub_name, rs) in enumerate(subs):
            if si > 0:
                lines.append(r"\midrule")
            for model_short, vals in rs:
                cells = []
                for ci, v in enumerate(vals):
                    s = fmt(v)
                    if v is None:
                        cells.append(s); continue
                    is_major = (major_col_max[mi][ci] is not None
                                and abs(v - major_col_max[mi][ci]) < 1e-9)
                    is_sub = (sub_col_max[mi][si][ci] is not None
                              and abs(v - sub_col_max[mi][si][ci]) < 1e-9)
                    if is_major:
                        s = f"\\textbf{{{s}}}"
                    elif is_sub:
                        s = f"\\emph{{{s}}}"
                    cells.append(s)
                model_disp = MODEL_DISP.get(model_short, model_short)
                lines.append(f"{model_disp} & " + " & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def render_table(*, label, caption, blocks):
    """blocks = [(setup_label, [(model_short, [12 floats|None]), ...]), ...]

    Per-cell formatting:
      \\textbf{X}  -- the overall column maximum (across all blocks)
      \\emph{X}    -- the per-block column maximum (when not also the overall)
      X plain     -- everything else
    """
    # per-block column maxima
    block_col_max = []
    for _, rs in blocks:
        cm = []
        for ci in range(12):
            vals = [r[1][ci] for r in rs if r[1][ci] is not None]
            cm.append(max(vals) if vals else None)
        block_col_max.append(cm)

    # overall column maxima (across all rows in all blocks)
    overall_col_max = []
    for ci in range(12):
        vals = [r[1][ci] for _, rs in blocks for r in rs if r[1][ci] is not None]
        overall_col_max.append(max(vals) if vals else None)

    # Lang headers as lowercase monospace, matching the rest of the project's
    # language-label convention (\texttt{amh}, \texttt{hau}, ...).
    lang_cols = [r"\texttt{" + l + "}" for l in LANGS] * 2
    head_row = " & ".join([" "] * 2 + lang_cols) + r" \\"

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
        r"\textbf{Setup} & \textbf{Model} & \multicolumn{6}{c}{\textbf{Eng$\to$}} & \multicolumn{6}{c}{\textbf{$\to$Eng}} \\",
        r"\cmidrule(lr){3-8}\cmidrule(lr){9-14}",
        head_row,
        r"\midrule",
    ]

    for bi, (setup, rs) in enumerate(blocks):
        if bi > 0:
            lines.append(r"\midrule")
        n = len(rs)
        col_max = block_col_max[bi]
        for ri, (model_short, vals) in enumerate(rs):
            cells = []
            for ci, v in enumerate(vals):
                s = fmt(v)
                if v is None:
                    cells.append(s); continue
                is_overall = (overall_col_max[ci] is not None
                              and abs(v - overall_col_max[ci]) < 1e-9)
                is_block = (col_max[ci] is not None
                            and abs(v - col_max[ci]) < 1e-9)
                if is_overall:
                    s = f"\\textbf{{{s}}}"
                elif is_block:
                    s = f"\\emph{{{s}}}"
                cells.append(s)
            head_cell = (
                f"\\multirow{{{n}}}{{*}}{{\\textit{{{setup}}}}}"
                if ri == 0 else ""
            )
            model_disp = MODEL_DISP.get(model_short, model_short)
            lines.append(f"{head_cell} & {model_disp} & " + " & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


# ---------------- table builders ----------------

def vec(d, pairs=PAIRS):
    return [d.get(p) for p in pairs]


def _seq2seq_blocks(rows, metric, models):
    zs_rows = [
        (m, vec(lookup(rows, metric=metric, model_short=m,
                       experiment_type="zero_shot")))
        for m in models
    ]
    ft_rows = [
        (m, vec(lookup(rows, metric=metric, model_short=m,
                       experiment_type="seq2seq_finetune", dataset="dcs",
                       is_ablation="False")))
        for m in models
    ]
    return [("Zero-shot", zs_rows), ("Fine-tuned", ft_rows)]


def build_seq2seq(rows, metric):
    """Main-paper seq2seq table: M2M100-1.2B and NLLB-1.3B only."""
    metric_label = "chrF" if metric == "chrf" else "COMET"
    return render_table(
        label=f"tab:seq2seq_{'chrf' if metric=='chrf' else 'comet'}",
        caption=f"Seq2Seq: zero-shot vs. fine-tuned ({metric_label})",
        blocks=_seq2seq_blocks(rows, metric, SEQ2SEQ_MODELS_MAIN),
    )


def build_seq2seq_appendix(rows, metric):
    """Appendix seq2seq table: smaller backbones (M2M100-418M, NLLB-600M)."""
    metric_label = "chrF" if metric == "chrf" else "COMET"
    return render_table(
        label=f"tab:seq2seq_smaller_{'chrf' if metric=='chrf' else 'comet'}",
        caption=("Smaller seq2seq backbones (M2M100-418M, NLLB-600M): "
                 f"zero-shot vs. fine-tuned ({metric_label}). "
                 "These are reported for completeness; the main paper "
                 "discusses the larger M2M100-1.2B and NLLB-1.3B."),
        blocks=_seq2seq_blocks(rows, metric, SEQ2SEQ_MODELS_APPENDIX),
    )


def build_llm_sentence(rows, metric):
    metric_label = "chrF" if metric == "chrf" else "COMET"
    zs_rows = [
        (m, vec(lookup(rows, metric=metric, model_short=m,
                       experiment_type="llm_zero_shot", is_ablation="False")))
        for m in LLM_MODELS
    ]
    icl_rows = []
    for m in LLM_MODELS:
        # most LLMs use "icl_5shot"; falls back to "icl_5shot_semantic" / "icl_0shot"
        d = first_with_pred(rows, metric=metric, candidates=[
            dict(experiment_type="llm_icl", prompt_strategy="icl_5shot",
                 model_short=m, is_ablation="False"),
            dict(experiment_type="llm_icl", prompt_strategy="icl_5shot_semantic",
                 model_short=m, is_ablation="False"),
        ])
        icl_rows.append((m, vec(d)))
    return render_table(
        label=f"tab:llm_sentence_{'chrf' if metric=='chrf' else 'comet'}",
        caption=f"LLM sentence-level: zero-shot vs. 5-shot ICL ({metric_label})",
        blocks=[("Zero-shot", zs_rows), ("5-shot ICL", icl_rows)],
    )


def build_lora(rows, metric):
    metric_label = "chrF" if metric == "chrf" else "COMET"
    r4_rows = [
        (m, vec(lookup(rows, metric=metric, model_short=m,
                       experiment_type="lora_finetune", lora_rank="4",
                       is_ablation="False")))
        for m in LORA_MODELS
    ]
    r64_rows = [
        (m, vec(lookup(rows, metric=metric, model_short=m,
                       experiment_type="lora_finetune", lora_rank="64",
                       is_ablation="False")))
        for m in LORA_MODELS
    ]
    return render_table(
        label=f"tab:lora_{'chrf' if metric=='chrf' else 'comet'}",
        caption=f"LoRA fine-tuning: $r{{=}}4$ vs. $r{{=}}64$ ({metric_label})",
        blocks=[("LoRA $r{=}4$", r4_rows), ("LoRA $r{=}64$", r64_rows)],
    )


def build_doc(rows, metric):
    """**Main paper Table 4 (or appendix).** Document-level zero-shot vs.\
    1-shot ICL. Two major groups; each has closed (Gemini-1.5, GPT-4o) and
    open (7 LLMs incl. TranslateGemma-12B) sub-blocks. Same hierarchical
    bolding as Tables 1/2 and an Avg column."""
    metric_label = "chrF" if metric == "chrf" else "COMET"
    closed_models = ["gemini_15_flash", "gpt4o"]
    open_models = LLM_MODELS  # already includes translategemma_12b

    def by_disp(model_list):
        return sorted(model_list, key=lambda m: MODEL_DISP.get(m, m).lower())

    def doc_lookup(model, shot):
        """shot in {0, 1}; accept both new and legacy prompt strategies."""
        new_key = f"doc_{shot}shot"
        legacy_key = f"doc_{shot}shot_legacy"
        return first_with_pred(rows, metric=metric, candidates=[
            dict(experiment_type="llm_doc", prompt_strategy=new_key,
                 model_short=model, is_ablation="False"),
            dict(experiment_type="llm_doc", prompt_strategy=legacy_key,
                 model_short=model, is_ablation="False"),
        ])

    def rows_for(models, shot):
        return [(m, vec(doc_lookup(m, shot))) for m in by_disp(models)]

    return render_grouped_table(
        label=f"tab:doc_{'chrf' if metric=='chrf' else 'comet'}",
        caption=("Document-level zero-shot vs. 1-shot in-context learning "
                 f"({metric_label}). Two major groups; each has closed-API "
                 "and open-LLM sub-blocks. Models alphabetically sorted "
                 "within each sub-block. \\textbf{Bold}: column maximum "
                 "within the major group (\\textit{Zero-shot} or "
                 "\\textit{1-shot}); \\emph{italic}: column maximum within "
                 "the sub-block (closed or open)."),
        major_groups=[
            ("Zero-shot", [
                (None, rows_for(closed_models, 0)),
                (None, rows_for(open_models,   0)),
            ]),
            ("1-shot", [
                (None, rows_for(closed_models, 1)),
                (None, rows_for(open_models,   1)),
            ]),
        ],
    )


def build_main_zs_ft(rows, metric):
    """**Main paper Table 1.** Two major groups (Zero-shot, Fine-tuned),
    each with two sub-blocks (Seq2Seq, LLM). Models sorted alphabetically
    within each sub-block by display name.
    """
    metric_label = "chrF" if metric == "chrf" else "COMET"
    SEQ2SEQ_ALL = ["m2m100_418m", "nllb_200_distilled_600m",
                   "m2m100_1_2b", "nllb_200_1_3b"]
    # Fine-tuned LLMs share the same six open models; the zero-shot block
    # additionally includes TranslateGemma-12B (which we evaluate but don't fine-tune).
    LLM_HEADLINE_FT = ["llama3_8b", "gemma2_9b_it", "afriquellama_8b",
                       "afriqueqwen_8b", "tiny_aya_earth", "tiny_aya_global"]
    LLM_HEADLINE_ZS = LLM_HEADLINE_FT + ["translategemma_12b"]

    def by_disp(model_list):
        return sorted(model_list, key=lambda m: MODEL_DISP.get(m, m).lower())

    SEQ2SEQ_SORTED = by_disp(SEQ2SEQ_ALL)
    LLM_ZS_SORTED = by_disp(LLM_HEADLINE_ZS)
    LLM_FT_SORTED = by_disp(LLM_HEADLINE_FT)

    def rows_for(models, **filters):
        return [(m, vec(lookup(rows, metric=metric, model_short=m, **filters)))
                for m in models]

    zs_seq = rows_for(SEQ2SEQ_SORTED, experiment_type="zero_shot")
    ft_seq = rows_for(SEQ2SEQ_SORTED, experiment_type="seq2seq_finetune",
                      dataset="dcs", is_ablation="False")
    zs_llm = rows_for(LLM_ZS_SORTED, experiment_type="llm_zero_shot",
                      is_ablation="False")
    ft_llm = rows_for(LLM_FT_SORTED, experiment_type="lora_finetune",
                      lora_rank="64", is_ablation="False")

    return render_grouped_table(
        label=f"tab:main_zs_ft_{'chrf' if metric=='chrf' else 'comet'}",
        caption=(f"Sentence-level zero-shot vs. fine-tuned performance "
                 f"({metric_label}). The four seq2seq backbones (M2M100-418M, "
                 "NLLB-600M, M2M100-1.2B, NLLB-1.3B) and six open-source "
                 "LLMs appear in both major groups; the \\textit{Zero-shot} "
                 "group additionally includes TranslateGemma-12B (evaluated but "
                 "not fine-tuned). LLMs in the \\textit{Fine-tuned} group "
                 "use per-pair LoRA $r{=}64$. Models are alphabetically "
                 "sorted within each sub-block. \\textbf{Bold}: column "
                 "maximum within the major group (\\textit{Zero-shot} or "
                 "\\textit{Fine-tuned}); \\emph{italic}: column maximum "
                 "within the sub-block (seq2seq or LLM)."),
        major_groups=[
            ("Zero-shot",  [(None, zs_seq), (None, zs_llm)]),
            ("Fine-tuned", [(None, ft_seq), (None, ft_llm)]),
        ],
    )


def build_open_vs_closed_icl(rows, metric):
    """**Main paper Table 2.** Closed-API zero-shot + ICL alongside
    per-model open-LLM ICL, all at the locked universal config
    (template5, 10 shots, T=0). Open-LLM zero-shot rows live in
    Table 1; a single per-cell \"Best open\" row keeps the zero-shot
    block readable as a baseline against the closed APIs.

    Three open LLMs (AfriqueLlama-8B, AfriqueQwen-8B,
    TranslateGemma-12B) degenerate at template5_10shot (visible as
    COMET >> ChrF in their rows) -- that's an honest result of running
    the same prompt across all models. Vanilla-vs-templated ablation is
    in the appendix table prompt_engineering_open.
    """
    metric_label = "chrF" if metric == "chrf" else "COMET"
    open_models = ["llama3_8b", "gemma2_9b_it", "afriquellama_8b",
                   "afriqueqwen_8b", "tiny_aya_earth", "tiny_aya_global",
                   "translategemma_12b"]
    HEADLINE_STRATEGY = "template5_10shot"  # locked 2026-05-08

    def by_disp(model_list):
        return sorted(model_list, key=lambda m: MODEL_DISP.get(m, m).lower())

    def closed_zs(model):
        return lookup(rows, metric=metric, model_short=model,
                      experiment_type="llm_zero_shot", is_ablation="False")

    def llm_icl(model):
        # No is_ablation filter: template5_10shot for the 4 reference models
        # came from the prompt-engineering sweep (tagged is_ablation=True),
        # while the 3 newly-rerun open LLMs are tagged False. We're now
        # promoting template5_10shot to the headline body config across the
        # board, so the ablation tag is irrelevant for this lookup.
        return lookup(rows, metric=metric, model_short=model,
                      experiment_type="llm_icl",
                      prompt_strategy=HEADLINE_STRATEGY)

    def best_open_zs():
        out = {p: None for p in PAIRS}
        for m in open_models:
            d = lookup(rows, metric=metric, model_short=m,
                       experiment_type="llm_zero_shot", is_ablation="False")
            for p, v in d.items():
                if v is None:
                    continue
                if out.get(p) is None or v > out[p]:
                    out[p] = v
        return out

    closed = ["gemini_15_flash", "gpt4o"]
    closed_zs_rows  = [(m, vec(closed_zs(m))) for m in by_disp(closed)]
    closed_icl_rows = [(m, vec(llm_icl(m)))   for m in by_disp(closed)]
    open_icl_rows   = [(m, vec(llm_icl(m)))   for m in by_disp(open_models)]

    return render_grouped_table(
        label=f"tab:open_vs_closed_icl_{'chrf' if metric=='chrf' else 'comet'}",
        caption=("Open vs. closed-source LLMs at sentence-level "
                 f"in-context learning ({metric_label}), all rows at the "
                 "uniform \\texttt{template5\\_10shot} configuration "
                 "(see \\Cref{tab:template_shot_ablation_" +
                 ("chrf" if metric == "chrf" else "comet") + "} for the "
                 "template/shot ablation that selected this config). "
                 "\\textit{Best open} in the zero-shot block is the "
                 "per-cell maximum across the seven open LLMs in "
                 "\\Cref{tab:main_zs_ft_" +
                 ("chrf" if metric == "chrf" else "comet") + "}. "
                 "AfriqueLlama-8B, AfriqueQwen-8B, and TranslateGemma-12B "
                 "degenerate at this prompt (visible as COMET~$\\gg$~ChrF in "
                 "their rows); see \\Cref{tab:prompt_engineering_open_" +
                 ("chrf" if metric == "chrf" else "comet") + "} for their "
                 "vanilla-prompt baseline. "
                 "\\textbf{Bold}: best performance across "
                 "\\textit{zero-shot} and \\textit{ICL}; "
                 "\\emph{italic}: best performance within open or closed "
                 "model group."),
        major_groups=[
            ("Zero-shot", [
                (None, [("best_open", vec(best_open_zs()))]),
                (None, closed_zs_rows),
            ]),
            ("template5, 10-shot ICL", [
                (None, closed_icl_rows),
                (None, open_icl_rows),
            ]),
        ],
    )


def build_closed_ablations(rows, metric):
    """**Appendix.** Closed-API templated-ICL ablation, three blocks:

      1. \\textit{Reported}: the eng->xxx and xxx->eng numbers used in
         Table~\\ref{tab:open_vs_closed_icl_*} at $T{=}0$ with each
         model's headline templated configuration.
      2. \\textit{Retrieval ablation} (eng->xxx, $T{=}0$):
         template-based vs semantic-similarity 5-shot exemplars.
      3. \\textit{Temperature ablation} (eng->xxx, $T{=}0$ vs $T{=}0.6$):
         each model's headline templated config replayed at the higher
         temperature.

    The closed-API templated runs are now bidirectional at $T{=}0$
    (Gemma2/Llama3 sweeps imported on 2026-05-08), so this table no
    longer falls back to vanilla `icl_5shot` rows.
    """
    metric_label = "chrF" if metric == "chrf" else "COMET"

    # Per-model headline templated config (computed from summary.csv on
    # 2026-05-08; see PAPER_TODOS bookmark "Template-vs-shot dependence"):
    HEADLINE_STRATEGY = {
        "gemini_15_flash": "template5_10shot",   # avg 65.96
        "gpt4o":            "template3_10shot",  # avg 65.87 (template5_10shot is 65.77)
    }

    def pull(model, etype, temp, *, strategies, pairs):
        merged = {}
        for ps in strategies:
            filters = dict(model_short=model, experiment_type=etype,
                           temp_setting=temp)
            if ps is not None:
                filters["prompt_strategy"] = ps
            d = lookup(rows, metric=metric, **filters)
            for p in pairs:
                if p in d:
                    merged.setdefault(p, d[p])
        return merged

    def vec(d, pairs):
        return [d.get(p) for p in pairs]

    def avg_(vs):
        nums = [v for v in vs if v is not None]
        return sum(nums) / len(nums) if nums else None

    def row(model, label, etype, temp, strategies, pairs):
        v = vec(pull(model, etype, temp, strategies=strategies, pairs=pairs), pairs)
        return (model, label, v + [avg_(v)])

    # ---- block 1: Reported headline at T=0 (eng->xxx half).
    # Per-model best templated 10-shot, capped at 10 per the project rule.
    # Whole table is eng->xxx only because Blocks 2 and 3 only have data
    # for that direction (no closed-API ablation runs on xxx->eng).
    reported = []
    for m in ("gemini_15_flash", "gpt4o"):
        s = HEADLINE_STRATEGY[m]
        reported.append(row(m, "ZS",
                            "llm_zero_shot", "t0", (None,), PAIRS_E2X))
        reported.append(row(m, f"10-shot ({s.split('_')[0]})",
                            "llm_icl", "t0", (s,), PAIRS_E2X))

    # ---- block 2: retrieval ablation (eng->xxx, T=0).
    # Per-model templated 10-shot vs semantic_10shot.
    retrieval_abl = []
    for m in ("gemini_15_flash", "gpt4o"):
        s = HEADLINE_STRATEGY[m]
        retrieval_abl.append(row(m, f"10-shot (templated, {s.split('_')[0]})",
                                 "llm_icl", "t0", (s,), PAIRS_E2X))
        retrieval_abl.append(row(m, "10-shot (semantic similarity)",
                                 "llm_icl", "t0", ("semantic_10shot",), PAIRS_E2X))

    # ---- block 3: temperature ablation (eng->xxx, T=0 vs T=0.6).
    # Each model's headline templated config replayed at T=0.6. Now clean
    # for both models since we have GPT-4o templated coverage at both
    # temperatures.
    temp_abl = []
    for m in ("gemini_15_flash", "gpt4o"):
        s = HEADLINE_STRATEGY[m]
        temp_abl.append(row(m, "ZS ($T{=}0$)",
                            "llm_zero_shot", "t0",  (None,), PAIRS_E2X))
        temp_abl.append(row(m, "ZS ($T{=}0.6$)",
                            "llm_zero_shot", "t06", (None,), PAIRS_E2X))
        temp_abl.append(row(m, f"10-shot ({s.split('_')[0]}) $T{{=}}0$",
                            "llm_icl", "t0",  (s,), PAIRS_E2X))
        temp_abl.append(row(m, f"10-shot ({s.split('_')[0]}) $T{{=}}0.6$",
                            "llm_icl", "t06", (s,), PAIRS_E2X))

    # ---- per-block column max for bold highlighting.
    def col_max(brows):
        cols = []
        for ci in range(7):
            vs = [r[2][ci] for r in brows if r[2][ci] is not None]
            cols.append(max(vs) if vs else None)
        return cols

    lang_cols = [r"\texttt{" + l + "}" for l in LANGS]
    lang_head = " & ".join(["  ", "  "] + lang_cols + [r"\textbf{Avg}"]) + r" \\"

    lines = [
        f"% ---- closed_ablations_{'chrf' if metric=='chrf' else 'comet'} ----",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Setting} & \multicolumn{6}{c}{\textbf{eng$\to$}} & \\",
        r"\cmidrule(lr){3-8}",
        lang_head,
    ]

    def emit_block(name, brows, cmax, first=False):
        if first:
            lines.append(r"\midrule")
        else:
            lines.append(r"\midrule\midrule")
        lines.append(r"\multicolumn{9}{l}{\textit{" + name + r"}} \\")
        lines.append(r"\midrule")
        for model_short, setting_label, vals in brows:
            disp = MODEL_DISP.get(model_short, model_short)
            cells = []
            for ci, v in enumerate(vals):
                s = fmt(v)
                if v is not None and cmax[ci] is not None and abs(v - cmax[ci]) < 1e-9:
                    s = f"\\textbf{{{s}}}"
                cells.append(s)
            lines.append(f"{disp} & {setting_label} & " + " & ".join(cells) + r" \\")

    rep_max  = col_max(reported)
    ret_max  = col_max(retrieval_abl)
    temp_max = col_max(temp_abl)

    emit_block("Reported ($T{=}0$)",
               reported, rep_max, first=True)
    emit_block("Retrieval ablation: templated vs.\\ semantic-similarity, $T{=}0$",
               retrieval_abl, ret_max)
    emit_block("Temperature ablation ($T{=}0$ vs $T{=}0.6$)",
               temp_abl, temp_max)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{Closed-API ablations on \texttt{eng}$\to$\texttt{xxx} "
         f"({metric_label}). "
         r"\textit{Reported} duplicates the \texttt{eng}$\to$\texttt{xxx} "
         r"numbers from Table~\ref{tab:open_vs_closed_icl_" +
         ("chrf" if metric == "chrf" else "comet") + r"} for these two "
         r"models at their headline templated 10-shot configuration "
         r"(Gemini-1.5: \texttt{template5\_10shot}, "
         r"GPT-4o: \texttt{template3\_10shot}). "
         r"\textit{Retrieval ablation} contrasts the same templated "
         r"10-shot run against semantic-similarity 10-shot exemplars at "
         r"$T{=}0$. "
         r"\textit{Temperature ablation} replays each model's headline "
         r"templated configuration at $T{=}0.6$ to isolate temperature. "
         r"All ablations are \texttt{eng}$\to$\texttt{xxx} only "
         r"(no closed-API ablation runs on \texttt{xxx}$\to$\texttt{eng}). "
         r"\textbf{Bold}: column maximum within the block.}"),
        r"\label{tab:closed_ablations_" + ("chrf" if metric == "chrf" else "comet") + "}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def build_template_shot_ablation(rows, metric):
    """**Appendix.** Template/shot ablation across the four reference
    models with full templated coverage at T=0
    (Gemini-1.5-Flash, GPT-4o, Llama3-8B, Gemma2-9B-IT).

    Layout follows Tables 11/12 of the existing draft
    (tab:models_performances_templates_*): per-model multirow block,
    one row per (template, shot) configuration, language pairs as
    columns (eng->xxx then xxx->eng) with per-row Avg, and a per-model
    avg row at the bottom of each block. Bold cells are the per-lang
    column maximum within each model block. The locked universal
    choice template5_10shot is highlighted via cmidrule emphasis in
    the caption (see docs/PAPER_TODOS.md "2026-05-08 -- DECISION").
    """
    metric_label = "chrF" if metric == "chrf" else "COMET"
    AB_MODELS = ["gemini_15_flash", "gpt4o", "llama3_8b", "gemma2_9b_it"]

    # Configs sorted: by template number, then by shot count. Templates
    # 1, 2, 4 are 0-shot only; templates 3 and 5 carry the shot sweep.
    # 50-shot banned project-wide; 20-shot kept so readers see the
    # plateau directly.
    CONFIGS = [
        ("template1_0shot",  1, 0),
        ("template2_0shot",  2, 0),
        ("template3_5shot",  3, 5),
        ("template3_10shot", 3, 10),
        ("template3_20shot", 3, 20),
        ("template4_0shot",  4, 0),
        ("template5_5shot",  5, 5),
        ("template5_10shot", 5, 10),
        ("template5_20shot", 5, 20),
    ]
    LOCKED_CONFIG = "template5_10shot"

    def avg_(vs):
        nums = [v for v in vs if v is not None]
        return sum(nums) / len(nums) if nums else None

    def pull(model, strategy):
        return lookup(rows, metric=metric, model_short=model,
                      experiment_type="llm_icl", temp_setting="t0",
                      prompt_strategy=strategy)

    # Format a (template, shots) label like "1" / "3$_{s=5}$"
    def cfg_label(tmpl, shots):
        if shots == 0:
            return str(tmpl)
        return f"{tmpl}$_{{s={shots}}}$"

    def cell_str(v):
        return "--" if v is None else f"{v:.1f}"

    lang_cols_e2x = [r"\textbf{\texttt{" + l + "}}" for l in LANGS]
    lang_cols_x2e = [r"\textbf{\texttt{" + l + "}}" for l in LANGS]
    n_lang_cols = 12  # 6 e2x + 6 x2e
    avg_col_idx = n_lang_cols  # 13th data column

    # Build column spec: model, config, 12 lang cols, Avg
    col_spec = "ll" + "r" * n_lang_cols + "r"

    lines = [
        f"% ---- template_shot_ablation_{'chrf' if metric=='chrf' else 'comet'} ----",
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        # Direction header row spanning the 12 lang cols
        (r"\textbf{Model} & \textbf{Config} & "
         r"\multicolumn{6}{c}{\textbf{eng$\to$}} & "
         r"\multicolumn{6}{c}{\textbf{$\to$eng}} & \\"),
        r"\cmidrule(lr){3-8}\cmidrule(lr){9-14}",
        # Lang headers
        (r"  &   & " + " & ".join(lang_cols_e2x + lang_cols_x2e)
         + r" & \textbf{Avg} \\"),
        r"\midrule",
    ]

    for mi, model in enumerate(AB_MODELS):
        if mi > 0:
            lines.append(r"\midrule")

        # Per-model: pull data for every config × lang pair
        config_rows = []  # list of (label, tmpl, shots, [12 vals], avg)
        for label, tmpl, shots in CONFIGS:
            d = pull(model, label)
            vals = [d.get(p) for p in PAIRS]   # PAIRS = e2x + x2e (12 entries)
            config_rows.append((label, tmpl, shots, vals, avg_(vals)))

        # Per-language column max within this model block (12 cols + Avg)
        block_col_max = []
        for ci in range(n_lang_cols + 1):
            cv = []
            for _, _, _, vals, a in config_rows:
                v = vals[ci] if ci < n_lang_cols else a
                if v is not None:
                    cv.append(v)
            block_col_max.append(max(cv) if cv else None)

        # Per-language avg row across the configs (column avg)
        col_avgs = []
        for ci in range(n_lang_cols + 1):
            cv = []
            for _, _, _, vals, a in config_rows:
                v = vals[ci] if ci < n_lang_cols else a
                if v is not None:
                    cv.append(v)
            col_avgs.append(avg_(cv))

        # Emit rows. First row carries the multirow model name.
        n_block_rows = len(config_rows) + 1   # configs + per-lang avg
        for ri, (label, tmpl, shots, vals, a) in enumerate(config_rows):
            cells = vals + [a]
            cells_str = []
            for ci, v in enumerate(cells):
                s = cell_str(v)
                if (v is not None and block_col_max[ci] is not None
                        and abs(v - block_col_max[ci]) < 1e-9):
                    s = r"\textbf{" + s + r"}"
                cells_str.append(s)
            cfg = cfg_label(tmpl, shots)
            if label == LOCKED_CONFIG:
                cfg = r"\underline{" + cfg + r"}"
            if ri == 0:
                model_cell = (r"\multirow{" + str(n_block_rows) + r"}{*}{"
                              + r"\rotatebox{90}{"
                              + MODEL_DISP.get(model, model)
                              + r"}}")
            else:
                model_cell = ""
            lines.append(f"{model_cell} & {cfg} & "
                         + " & ".join(cells_str) + r" \\")

        # cmidrule before the per-lang avg row to separate it
        lines.append(r"\cmidrule(l){2-" + str(2 + n_lang_cols + 1) + "}")
        avg_cells = [cell_str(v) for v in col_avgs]
        lines.append(r" & \textbf{avg} & "
                     + " & ".join(avg_cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        (r"\caption{Template/shot ablation for the four reference models "
         "with bidirectional templated coverage at $T{=}0$ "
         f"({metric_label}). Each row is a (template, shot-count) "
         r"configuration (template number first, shot count as subscript "
         r"$s$); the underlined row \texttt{5}$_{s=10}$ is the "
         r"project-wide locked configuration "
         r"(see \texttt{docs/PAPER\_TODOS.md} ``2026-05-08 -- DECISION''). "
         r"\textbf{Bold}: per-language column maximum within each model "
         r"block. The bottom \textbf{avg} row of each block reports the "
         r"per-language average across configurations. 50-shot is banned "
         r"project-wide; 20-shot is included so the plateau past 10-shot "
         r"is visible -- the gain from 10$\to$20 shots is at most "
         r"$0.2$ COMET, far below our $1$-COMET adoption threshold.}"),
        r"\label{tab:template_shot_ablation_" + ("chrf" if metric == "chrf" else "comet") + "}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def build_prompt_engineering_open(rows, metric):
    """**Appendix.** Per-open-LLM vanilla vs. templated ICL ablation.

    Three columns of data per model: vanilla 0-shot (basic template),
    vanilla 5-shot (basic template + random retrieval), and
    template5_10shot (the locked uniform body config). Plus a delta
    column showing the lift over vanilla 5-shot. Bidirectional avg
    over the 12 lang pairs.

    Three rows show the COMET-much-greater-than-ChrF degeneration
    signature: AfriqueLlama-8B, AfriqueQwen-8B, TranslateGemma-12B.
    These models are off-distribution for template5 and recover with
    their vanilla prompt (already documented as
    docs/PAPER_TODOS.md "2026-05-09 -- Open-LLM templated rerun").
    """
    metric_label = "chrF" if metric == "chrf" else "COMET"
    OPEN = ["afriquellama_8b", "afriqueqwen_8b", "gemma2_9b_it",
            "llama3_8b", "tiny_aya_earth", "tiny_aya_global",
            "translategemma_12b"]

    def by_disp(model_list):
        return sorted(model_list, key=lambda m: MODEL_DISP.get(m, m).lower())

    def avg_(d):
        nums = [v for v in d.values() if v is not None]
        return sum(nums) / len(nums) if nums else None

    def pull_zs(model):
        return lookup(rows, metric=metric, model_short=model,
                      experiment_type="llm_zero_shot", is_ablation="False")

    def pull_icl(model, strategy):
        # See note in build_open_vs_closed_icl: template5_10shot is tagged
        # is_ablation=True for the 4 reference models (legacy from when it
        # was an ablation), False for the 3 newly-rerun ones. Drop the filter.
        return lookup(rows, metric=metric, model_short=model,
                      experiment_type="llm_icl",
                      prompt_strategy=strategy)

    table_rows = []   # (model, vzs_avg, v5_avg, t10_avg, delta)
    for m in by_disp(OPEN):
        vzs = avg_(pull_zs(m))
        v5  = avg_(pull_icl(m, "icl_5shot"))
        t10 = avg_(pull_icl(m, "template5_10shot"))
        delta = (t10 - v5) if (t10 is not None and v5 is not None) else None
        table_rows.append((m, vzs, v5, t10, delta))

    # Column maxes for bolding (across models, per column)
    cols = list(zip(*[(r[1], r[2], r[3], r[4]) for r in table_rows]))
    col_max = [max((v for v in c if v is not None), default=None) for c in cols]

    def fmt_v(v):
        return "--" if v is None else f"{v:.1f}"
    def fmt_d(v):
        return "--" if v is None else f"{v:+.1f}"

    lines = [
        f"% ---- prompt_engineering_open_{'chrf' if metric=='chrf' else 'comet'} ----",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        (r"\textbf{Model} & \textbf{Vanilla} & \textbf{Vanilla} & "
         r"\textbf{template5} & \textbf{$\Delta$} \\"),
        r"  & 0-shot & 5-shot & 10-shot & vs.\ 5-shot \\",
        r"\midrule",
    ]
    for m, vzs, v5, t10, delta in table_rows:
        cells = [fmt_v(vzs), fmt_v(v5), fmt_v(t10), fmt_d(delta)]
        cells_b = []
        for ci, (s, mxv) in enumerate(zip(cells, col_max)):
            v = (vzs, v5, t10, delta)[ci]
            if v is not None and mxv is not None and abs(v - mxv) < 1e-9:
                s = r"\textbf{" + s + r"}"
            cells_b.append(s)
        disp = MODEL_DISP.get(m, m)
        lines.append(f"{disp} & " + " & ".join(cells_b) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{Vanilla vs.\ templated ICL on the seven open LLMs "
         f"(bidirectional avg {metric_label} over the 12 \\texttt{{eng}}"
         r"$\leftrightarrow$\texttt{xxx} pairs at $T{=}0$). "
         r"Vanilla rows use the \texttt{basic} / \texttt{few\_shot} "
         r"prompt (random retrieval at 5-shot); the templated column "
         r"is the locked body configuration "
         r"\texttt{template5\_10shot}. The $\Delta$ column shows the "
         r"lift over the vanilla 5-shot baseline. AfriqueLlama-8B, "
         r"AfriqueQwen-8B, and TranslateGemma-12B post large $\Delta$ "
         r"COMET but degenerate at the templated prompt (their ChrF "
         r"row in the chrF mirror is near-zero), so the COMET gain "
         r"reflects fluent off-target generations rather than a real "
         r"translation improvement; their vanilla numbers are the "
         r"truer signal. \textbf{Bold}: column maximum.}"),
        r"\label{tab:prompt_engineering_open_" + ("chrf" if metric == "chrf" else "comet") + "}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ---------------- main ----------------

def main():
    rows = load_rows()
    builders = [
        ("main_zs_ft",         build_main_zs_ft),
        ("open_vs_closed_icl", build_open_vs_closed_icl),
        ("seq2seq",            build_seq2seq),
        ("seq2seq_smaller",    build_seq2seq_appendix),
        ("llm_sentence",       build_llm_sentence),
        ("lora",               build_lora),
        ("doc",                build_doc),
        ("closed_ablations",   build_closed_ablations),
        ("template_shot_ablation", build_template_shot_ablation),
        ("prompt_engineering_open", build_prompt_engineering_open),
    ]
    written = []
    for name, fn in builders:
        for metric in ("chrf", "ssa_comet"):
            suffix = "chrf" if metric == "chrf" else "comet"
            content = fn(rows, metric)
            out = OUT_DIR / f"{name}_{suffix}.tex"
            out.write_text(content)
            written.append(out)

    # Concatenate into a single sheet that mirrors new_results.tex
    sheet = "\n".join(p.read_text() for p in written)
    (OUT_DIR / "new_results_regenerated.tex").write_text(sheet)
    written.append(OUT_DIR / "new_results_regenerated.tex")

    print("Tables written:")
    for p in written:
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
