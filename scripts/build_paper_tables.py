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
    # Latest closed APIs (added 2026-05-13). Short forms here mirror the
    # legacy "GPT-4o" / "Gemini-1.5" labels; the full names
    # ("Gemini-3.1-Flash-Lite", "gpt-5.4") are spelled out only in the
    # methods / model-list paragraphs.
    "gpt54":                  "GPT-5.4",
    "gemini_31_flash_lite":   "Gemini-3.1",
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
    """Load summary.csv and dedupe to one row per
    (model_short, experiment_type, prompt_strategy, lang_pair, temp_setting,
    is_ablation, lora_rank). When the same cell appears multiple times
    (e.g.\\ the broken semantic dump + the patched re-run), keep the row
    with the latest timestamp."""
    raw = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            for k in ("bleu", "chrf", "ssa_comet"):
                v = r.get(k)
                r[k] = float(v) if v not in (None, "", "None") else None
            if not r.get("temp_setting"):
                r["temp_setting"] = "t0"
            elif r["temp_setting"] == "t6":
                r["temp_setting"] = "t06"
            raw.append(r)

    by_key = {}
    for r in raw:
        key = (r.get("model_short", ""), r.get("experiment_type", ""),
               r.get("prompt_strategy", ""), r.get("lang_pair", ""),
               r.get("temp_setting", ""), r.get("is_ablation", ""),
               r.get("lora_rank", ""))
        prev = by_key.get(key)
        if prev is None or (r.get("timestamp", "") >
                            prev.get("timestamp", "")):
            by_key[key] = r
    return list(by_key.values())


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
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
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
        caption=("Smaller seq2seq models (M2M100-418M, NLLB-600M): "
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
        caption=(f"LoRA fine-tuning of six open-source LLMs at "
                 f"two ranks ({metric_label}). Each cell is one "
                 f"(language, direction) pair; each model is fine-tuned "
                 f"bilingually per pair for 10 epochs with early "
                 f"stopping. Comparing $r{{=}}4$ (low-rank, "
                 f"parameter-efficient) against $r{{=}}64$ (the main "
                 f"rank used elsewhere in the paper) isolates the effect "
                 f"of adapter capacity. \\textbf{{Bold}}: best score "
                 f"for that (language, direction) pair across both "
                 f"ranks; \\emph{{italic}}: best score for that pair "
                 f"within a single rank block."),
        blocks=[("LoRA $r{=}4$", r4_rows), ("LoRA $r{=}64$", r64_rows)],
    )


def build_doc(rows, metric):
    """**Main paper Table 4 (or appendix).** Document-level zero-shot vs.\
    1-shot ICL. Two major groups; each has closed (Gemini-1.5, GPT-4o,
    Gemini-3.1-Flash-Lite, GPT-5.4) and open (7 LLMs incl. TranslateGemma-12B)
    sub-blocks. Same hierarchical bolding as Tables 1/2 and an Avg column."""
    metric_label = "chrF" if metric == "chrf" else "COMET"
    closed_models = ["gemini_15_flash", "gpt4o",
                     "gemini_31_flash_lite", "gpt54"]
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
                 f"({metric_label}). The four seq2seq models (M2M100-418M, "
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
        # Legacy closed APIs (gpt-4o, gemini-1.5) have a dedicated
        # experiment_type=llm_zero_shot. The latest closed APIs (gpt-5.4,
        # gemini-3.1) were collected via the vanilla rerun script which
        # writes everything under experiment_type=llm_icl with
        # prompt_strategy=0shot for the zero-shot cells. Fall back to that
        # so both eras populate the zero-shot row.
        return first_with_pred(rows, metric=metric, candidates=[
            dict(model_short=model, experiment_type="llm_zero_shot",
                 is_ablation="False"),
            dict(model_short=model, experiment_type="llm_icl",
                 prompt_strategy="0shot"),
        ])

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

    # Both legacy (gpt-4o, gemini-1.5-flash) and latest (gpt-5.4,
    # gemini-3.1-flash-lite) closed APIs side-by-side so the table shows
    # closed-API evolution rather than replacing the legacy numbers.
    closed = ["gemini_15_flash", "gpt4o",
              "gemini_31_flash_lite", "gpt54"]
    closed_zs_rows  = [(m, vec(closed_zs(m))) for m in by_disp(closed)]
    closed_icl_rows = [(m, vec(llm_icl(m)))   for m in by_disp(closed)]
    open_icl_rows   = [(m, vec(llm_icl(m)))   for m in by_disp(open_models)]

    return render_grouped_table(
        label=f"tab:open_vs_closed_icl_{'chrf' if metric=='chrf' else 'comet'}",
        caption=("Open vs. closed-source LLMs at sentence-level "
                 f"in-context learning ({metric_label}), all rows at the "
                 "uniform \\texttt{template5\\textunderscore{}10shot} configuration "
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
    # 2026-05-08; see PAPER_TODOS bookmark "Template-vs-shot dependence").
    # Latest closed APIs use template5_10shot, the locked universal config.
    HEADLINE_STRATEGY = {
        "gemini_15_flash":      "template5_10shot",   # avg 65.96
        "gpt4o":                 "template3_10shot",  # avg 65.87 (template5_10shot is 65.77)
        "gemini_31_flash_lite": "template5_10shot",   # locked
        "gpt54":                 "template5_10shot",  # locked
    }
    # Family + version grouping — rendered with \multirow on the first
    # two columns so the table reads "Gemini {1.5, 3.1}; GPT {4o, 5.4}"
    # instead of repeating "Gemini-1.5 / GPT-4o / ..." on every row.
    FAMILIES = [
        ("Gemini", [("1.5",  "gemini_15_flash"),
                    ("3.1",  "gemini_31_flash_lite")]),
        ("GPT",    [("4o",   "gpt4o"),
                    ("5.4",  "gpt54")]),
    ]
    CLOSED = tuple(m for _, vs in FAMILIES for _, m in vs)

    def pull(model, etype, temp, *, strategies, pairs):
        merged = {}
        # Legacy closed APIs (gpt4o, gemini_15_flash) store ZS under
        # experiment_type=llm_zero_shot; the latest closed APIs
        # (gpt54, gemini_31_flash_lite) store the same vanilla ZS run
        # under experiment_type=llm_icl, prompt_strategy="0shot".
        # When the caller asks for llm_zero_shot, also fall back to the
        # llm_icl/0shot variant so the row populates for both vintages.
        if etype == "llm_zero_shot":
            etypes = [("llm_zero_shot", strategies),
                      ("llm_icl",       ("0shot",))]
        else:
            etypes = [(etype, strategies)]
        for et, strats in etypes:
            for ps in strats:
                filters = dict(model_short=model, experiment_type=et,
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

    # ---- blocks 1a + 1b: Reported at T=0, vanilla and templated split.
    # Each sub-block has TWO rows per model so the ZS vs ICL contrast is
    # explicit within the same prompt-style class:
    #   * Vanilla:   ZS = vanilla 0shot;  ICL = vanilla 5shot
    #   * Templated: ZS = mean of template{1,2,4}_0shot;
    #                ICL = best templated 10-shot per model
    # The legacy closed APIs (gpt-4o, gemini-1.5) never had a vanilla
    # 5-shot run on the closed-API side; those cells render as "--".
    # Whole table is eng->xxx only because the ablation blocks only
    # have data for that direction (no closed-API ablation runs on xxx->eng).
    ZS_TEMPLATES = ("template1_0shot", "template2_0shot", "template4_0shot")

    def merge_pair_means(model, etype, temp, strategies):
        """Per-pair mean over `strategies`; returns {pair: avg} dict."""
        per_strat = [pull(model, etype, temp, strategies=(s,),
                          pairs=PAIRS_E2X) for s in strategies]
        merged = {}
        for p in PAIRS_E2X:
            vs = [d[p] for d in per_strat if p in d]
            if vs:
                merged[p] = sum(vs) / len(vs)
        return merged

    def avg_row(model, label, etype, temp, strategies):
        d = merge_pair_means(model, etype, temp, strategies)
        v = vec(d, PAIRS_E2X)
        return (model, label, v + [avg_(v)])

    def keep_present(rows_list):
        """Drop rows whose data cells (excluding the trailing avg) are all None."""
        return [r for r in rows_list if any(v is not None for v in r[2][:-1])]

    reported_vanilla = []
    for m in CLOSED:
        reported_vanilla.append(row(m, "ZS",
                                    "llm_zero_shot", "t0", (None,),
                                    PAIRS_E2X))
        reported_vanilla.append(row(m, "ICL (5-shot)",
                                    "llm_icl", "t0", ("5shot",),
                                    PAIRS_E2X))
    reported_vanilla = keep_present(reported_vanilla)

    reported_templated = []
    for m in CLOSED:
        s = HEADLINE_STRATEGY[m]
        reported_templated.append(avg_row(m,
            r"ZS (avg \texttt{template\{1,2,4\}})",
            "llm_icl", "t0", ZS_TEMPLATES))
        reported_templated.append(row(m,
            f"ICL (10-shot, {s.split('_')[0]})",
            "llm_icl", "t0", (s,), PAIRS_E2X))
    reported_templated = keep_present(reported_templated)

    # ---- block 2: retrieval ablation (eng->xxx, T=0).
    # Per-model templated 10-shot vs semantic_10shot.
    retrieval_abl = []
    for m in CLOSED:
        s = HEADLINE_STRATEGY[m]
        retrieval_abl.append(row(m, f"10-shot (templated, {s.split('_')[0]})",
                                 "llm_icl", "t0", (s,), PAIRS_E2X))
        retrieval_abl.append(row(m, "10-shot (semantic similarity)",
                                 "llm_icl", "t0", ("semantic_10shot",), PAIRS_E2X))

    # ---- block 3: temperature ablation (eng->xxx, T=0 vs T=0.6).
    # Two templated configurations comparable across all four closed APIs:
    #   * ZS = mean of template{1,2,4}_0shot (paper templates 1/2/4
    #     are 0-shot prompts). Vanilla 0shot is excluded because
    #     GPT-4o's t06 dump duplicates its t0 numbers and the latest
    #     closed APIs have no vanilla 0shot at T=0.6.
    #   * 10-shot at each model's best templated config.
    temp_abl = []
    for m in CLOSED:
        s = HEADLINE_STRATEGY[m]
        candidates = [
            avg_row(m, "ZS ($T{=}0$)",   "llm_icl", "t0",  ZS_TEMPLATES),
            avg_row(m, "ZS ($T{=}0.6$)", "llm_icl", "t06", ZS_TEMPLATES),
            row(m, f"10-shot ({s.split('_')[0]}) $T{{=}}0$",
                "llm_icl", "t0",  (s,), PAIRS_E2X),
            row(m, f"10-shot ({s.split('_')[0]}) $T{{=}}0.6$",
                "llm_icl", "t06", (s,), PAIRS_E2X),
        ]
        for cand in candidates:
            if any(v is not None for v in cand[2][:-1]):
                temp_abl.append(cand)

    # ---- per-block column max for bold highlighting.
    def col_max(brows):
        cols = []
        for ci in range(7):
            vs = [r[2][ci] for r in brows if r[2][ci] is not None]
            cols.append(max(vs) if vs else None)
        return cols

    lang_cols = [r"\texttt{" + l + "}" for l in LANGS]
    # Three left-side columns (Model, Version, Setting) + 6 lang + Avg = 10
    n_cols = 10
    lang_head = " & ".join(["  ", "  ", "  "] + lang_cols + [r"\textbf{Avg}"]) + r" \\"

    # Reverse lookup: model_short -> (family, version)
    model_to_famver = {m: (fam, ver)
                       for fam, vs in FAMILIES for ver, m in vs}

    lines = [
        f"% ---- closed_ablations_{'chrf' if metric=='chrf' else 'comet'} ----",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{lllrrrrrrr}",
        r"\toprule",
        (r"\textbf{Model} & \textbf{Version} & \textbf{Setting} & "
         r"\multicolumn{6}{c}{\textbf{eng$\to$}} & \\"),
        r"\cmidrule(lr){4-9}",
        lang_head,
    ]

    def emit_block(name, brows, cmax, first=False):
        if first:
            lines.append(r"\midrule")
        else:
            lines.append(r"\midrule\midrule")
        lines.append(r"\multicolumn{" + str(n_cols)
                     + r"}{l}{\textit{" + name + r"}} \\")
        lines.append(r"\midrule")

        # Group rows by (family, version) preserving FAMILIES order.
        by_famver = {}
        for r_model, r_label, r_vals in brows:
            by_famver.setdefault(r_model, []).append((r_label, r_vals))

        # Per-family row counts (sum over its present versions)
        fam_counts = {fam: sum(len(by_famver.get(m, []))
                               for _, m in vs)
                      for fam, vs in FAMILIES}

        for fi, (fam, vs) in enumerate(FAMILIES):
            if fi > 0:
                lines.append(r"\cmidrule(lr){1-" + str(n_cols) + "}")
            fam_n = fam_counts[fam]
            if fam_n == 0:
                continue
            fam_emitted = False
            for vi, (ver, model) in enumerate(vs):
                rows_here = by_famver.get(model, [])
                if not rows_here:
                    continue
                ver_n = len(rows_here)
                # Per-(version) per-column max for italic emphasis: highlights
                # the better of the within-model rows (e.g. ZS vs ICL inside
                # one model's vanilla block) so the reader can scan whether
                # ICL beats ZS for that model. Only emit italic when the
                # column has >=2 present values among the model's rows
                # (otherwise there's nothing to compare against).
                ver_col_max = []
                ver_col_n   = []
                for ci in range(7):
                    cv = [vals[ci] for _, vals in rows_here
                          if vals[ci] is not None]
                    ver_col_max.append(max(cv) if cv else None)
                    ver_col_n.append(len(cv))
                if vi > 0:
                    # divider between versions within a family
                    lines.append(r"\cmidrule(lr){2-" + str(n_cols) + "}")
                for ri, (setting_label, vals) in enumerate(rows_here):
                    cells = []
                    for ci, v in enumerate(vals):
                        s = fmt(v)
                        if (v is not None and ver_col_max[ci] is not None
                                and abs(v - ver_col_max[ci]) < 1e-9
                                and ver_col_n[ci] >= 2):
                            s = f"\\emph{{{s}}}"
                        if (v is not None and cmax[ci] is not None
                                and abs(v - cmax[ci]) < 1e-9):
                            s = f"\\textbf{{{s}}}"
                        cells.append(s)
                    if not fam_emitted:
                        fam_cell = (r"\multirow{" + str(fam_n) + r"}{*}{"
                                    + fam + r"}")
                        fam_emitted = True
                    else:
                        fam_cell = ""
                    if ri == 0:
                        ver_cell = (r"\multirow{" + str(ver_n) + r"}{*}{"
                                    + ver + r"}")
                    else:
                        ver_cell = ""
                    lines.append(f"{fam_cell} & {ver_cell} & "
                                 f"{setting_label} & "
                                 + " & ".join(cells) + r" \\")

    rep_v_max = col_max(reported_vanilla)
    rep_t_max = col_max(reported_templated)
    ret_max   = col_max(retrieval_abl)
    temp_max  = col_max(temp_abl)

    emit_block("Reported, vanilla prompts ($T{=}0$): ZS vs ICL (5-shot)",
               reported_vanilla, rep_v_max, first=True)
    emit_block(r"Reported, templated prompts ($T{=}0$): ZS (avg "
               r"\texttt{template\{1,2,4\}\textunderscore{}0shot}) vs ICL (best 10-shot)",
               reported_templated, rep_t_max)
    emit_block("Retrieval ablation: templated vs. semantic-similarity, $T{=}0$",
               retrieval_abl, ret_max)
    emit_block("Temperature ablation ($T{=}0$ vs $T{=}0.6$, all templated)",
               temp_abl, temp_max)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{Closed-API ablations on \texttt{eng}$\to$\texttt{xxx} "
         f"({metric_label}) for the four closed APIs "
         r"(legacy: Gemini-1.5, GPT-4o; latest: Gemini-3.1, GPT-5.4). "
         r"Four blocks: vanilla ZS vs ICL; templated ZS vs ICL; retrieval "
         r"(templated vs semantic 10-shot); and temperature ($T{=}0$ vs "
         r"$T{=}0.6$, all templated). \textbf{Bold}: per-block column "
         r"maximum. \emph{Italic}: per-version best across that model's "
         r"rows. See \Cref{app:closed_ablations_notes} for the full "
         r"setup and the rationale behind each block.}"),
        r"\label{tab:closed_ablations_" + ("chrf" if metric == "chrf" else "comet") + "}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def build_closed_ablations_notes():
    """Appendix paragraph that accompanies tab:closed_ablations_*. The
    table caption is intentionally short to avoid breaking the page; the
    full setup + rationale lives here so it can be \\input{} as a
    standalone subsection in the appendix."""
    return "\n".join([
        r"\subsection{Closed-API ablation setup}",
        r"\label{app:closed_ablations_notes}",
        r"",
        r"The closed-API ablation is reported on \texttt{eng}$\to$"
        r"\texttt{xxx} only. The four blocks are constructed so vanilla "
        r"and templated numbers are never compared inside the same "
        r"block.",
        r"",
        r"\paragraph{Reported, vanilla.} Vanilla ZS (\texttt{0shot}) vs "
        r"vanilla ICL (\texttt{5shot}) at $T{=}0$. Vanilla 5-shot is "
        r"available only for Gemini-3.1 and GPT-5.4; Gemini-1.5 and "
        r"GPT-4o appear as ZS only.",
        r"",
        r"\paragraph{Reported, templated.} Templated ZS vs templated ICL "
        r"at $T{=}0$. Templated ZS is the per-pair mean of "
        r"\texttt{template\{1,2,4\}\textunderscore{}0shot} (the three 0-shot "
        r"paper templates). Templated ICL is each model's best 10-shot "
        r"configuration: \texttt{template5\textunderscore{}10shot} for "
        r"Gemini-1.5, the latest Gemini, and the latest GPT; "
        r"\texttt{template3\textunderscore{}10shot} for GPT-4o.",
        r"",
        r"\paragraph{Retrieval ablation.} The same templated 10-shot "
        r"run as in the previous block, contrasted against "
        r"semantic-similarity 10-shot exemplars at $T{=}0$.",
        r"",
        r"\paragraph{Temperature ablation.} $T{=}0$ vs $T{=}0.6$ at the "
        r"two templated configurations comparable across all four APIs: "
        r"templated ZS (\texttt{template\{1,2,4\}\textunderscore{}0shot} mean) "
        r"and each model's best templated 10-shot. Vanilla 0-shot at "
        r"$T{=}0.6$ is excluded for comparability across all four APIs.",
        r"",
        r"\paragraph{Highlighting.} \textbf{Bold} marks the per-column "
        r"maximum within a block. \emph{Italic} marks the per-language "
        r"best across the rows of one model version (e.g. ICL vs ZS "
        r"within the same prompt-style class for that model); italic is "
        r"suppressed when only one row in the model's group has data "
        r"for that column.",
        r"",
    ])


def build_template_shot_ablation(rows, metric):
    """**Appendix.** Template/shot ablation across the six reference
    models with full templated coverage at T=0: legacy closed APIs
    (Gemini-1.5-Flash, GPT-4o), latest closed APIs (Gemini-3.1-Flash-Lite,
    GPT-5.4), and two open LLMs (Llama3-8B, Gemma2-9B-IT).

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
    # Reference models with bidirectional templated coverage at T=0.
    # Both legacy (gpt-4o, gemini-1.5) and latest (gpt-5.4, gemini-3.1) closed
    # APIs are included so the template/shot ablation surfaces closed-API
    # evolution alongside the two open-LLM reference models.
    AB_MODELS = ["gemini_15_flash", "gpt4o",
                 "gemini_31_flash_lite", "gpt54",
                 "llama3_8b", "gemma2_9b_it"]

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

    # Longtable lets the appendix render this across page breaks.
    # Requires \usepackage{longtable} in the preamble. For a
    # 2-column-wide ACL layout, wrap the \input{...} of this file in
    # an \onecolumn ... \twocolumn switch (or use the cuted/xltabular
    # packages if available) so the 15-column table spans the page.
    direction_hdr = (r"\textbf{Model} & \textbf{Config} & "
                     r"\multicolumn{6}{c}{\textbf{eng$\to$}} & "
                     r"\multicolumn{6}{c}{\textbf{$\to$eng}} & \\")
    cmidrules = r"\cmidrule(lr){3-8}\cmidrule(lr){9-14}"
    lang_hdr = (r"  &   & " + " & ".join(lang_cols_e2x + lang_cols_x2e)
                + r" & \textbf{Avg} \\")
    n_total_cols = 2 + n_lang_cols + 1   # Model + Config + 12 + Avg

    caption_text = (
        r"\caption{Template/shot ablation at $T{=}0$ for the six reference "
        r"models -- two open (Llama3-8B, Gemma2-9B-IT) and four closed "
        r"APIs (legacy: Gemini-1.5, GPT-4o; latest: Gemini-3.1, GPT-5.4) "
        f"({metric_label}). Each row is a (template, shot-count) "
        r"configuration (template number first, shot count as subscript "
        r"$s$); the underlined row \texttt{5}$_{s=10}$ is the "
        r"configuration used throughout the paper. "
        r"\textbf{Bold}: per-language column maximum within each model "
        r"block. The bottom \textbf{avg} row of each block reports the "
        r"per-language average across configurations. Shot counts are "
        r"capped at 20; the gain from 10$\to$20 shots is at most $0.2$ "
        r"COMET.}"
    )
    label = (r"\label{tab:template_shot_ablation_"
             + ("chrf" if metric == "chrf" else "comet") + "}")

    lines = [
        f"% ---- template_shot_ablation_{'chrf' if metric=='chrf' else 'comet'} ----",
        r"% Requires \usepackage{longtable}. For 2-column ACL layouts wrap",
        r"% the \input{...} of this file in \onecolumn ... \twocolumn so",
        r"% the 15-column table can span the page.",
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{longtable}{" + col_spec + r"}",
        r"\toprule",
        direction_hdr,
        cmidrules,
        lang_hdr,
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{" + str(n_total_cols) + r"}{l}{\textit{(continued)}} \\",
        r"\toprule",
        direction_hdr,
        cmidrules,
        lang_hdr,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{" + str(n_total_cols) + r"}{r}{\textit{(continued on next page)}} \\",
        r"\endfoot",
        r"\bottomrule",
        caption_text + " " + label + r" \\",
        r"\endlastfoot",
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
        r"\end{longtable}",
        r"\endgroup",
    ]
    return "\n".join(lines) + "\n"


def build_doc_template_shot_ablation(rows, metric):
    """**Appendix.** Document-level template/shot ablation for the two
    closed APIs (GPT-4o, Gemini-1.5-Flash) -- the only models with the
    full 3-templates x 3-shots grid at doc level.

    Open LLMs only have a single doc_0shot/doc_1shot config each (no
    template variants), so they appear in tab:doc_comet (the headline
    doc table) but not here. SSA-COMET is already populated for all
    432 llm_doc rows; no rescore needed.

    Layout mirrors tab:template_shot_ablation_comet (sentence-level):
    per-model multirow block, one row per (template, shot) config, 12
    pair columns (eng->xxx then xxx->eng), per-row Avg.
    """
    metric_label = "chrF" if metric == "chrf" else "COMET"
    # Both legacy (gpt-4o, gemini-1.5) and latest (gpt-5.4, gemini-3.1)
    # closed APIs share the same 3-template x 3-shot doc-level grid.
    AB_MODELS = ["gemini_15_flash", "gpt4o",
                 "gemini_31_flash_lite", "gpt54"]

    # 6 configs: 3 templates x 2 shot counts (0, 1).
    # Legacy closed APIs (gpt-4o, gemini-1.5) also have a 5-shot dump
    # per template, but the latest closed APIs (gpt-5.4, gemini-3.1)
    # were only run at doc_0shot/doc_1shot per template to conserve
    # credits, so the combined ablation drops 5-shot to keep the grid
    # uniform across all four models.
    CONFIGS = []
    for tmpl in (1, 2, 3):
        for shots in (0, 1):
            CONFIGS.append((f"doc_{shots}shot_template{tmpl}", tmpl, shots))

    def avg_(vs):
        nums = [v for v in vs if v is not None]
        return sum(nums) / len(nums) if nums else None

    def pull(model, strategy):
        return lookup(rows, metric=metric, model_short=model,
                      experiment_type="llm_doc",
                      prompt_strategy=strategy)

    def cfg_label(tmpl, shots):
        return f"{tmpl}$_{{s={shots}}}$"

    def cell_str(v):
        return "--" if v is None else f"{v:.1f}"

    lang_cols_e2x = [r"\textbf{\texttt{" + l + "}}" for l in LANGS]
    lang_cols_x2e = [r"\textbf{\texttt{" + l + "}}" for l in LANGS]
    n_lang_cols = 12
    col_spec = "ll" + "r" * n_lang_cols + "r"

    lines = [
        f"% ---- doc_template_shot_ablation_{'chrf' if metric=='chrf' else 'comet'} ----",
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        (r"\textbf{Model} & \textbf{Config} & "
         r"\multicolumn{6}{c}{\textbf{eng$\to$}} & "
         r"\multicolumn{6}{c}{\textbf{$\to$eng}} & \\"),
        r"\cmidrule(lr){3-8}\cmidrule(lr){9-14}",
        (r"  &   & " + " & ".join(lang_cols_e2x + lang_cols_x2e)
         + r" & \textbf{Avg} \\"),
        r"\midrule",
    ]

    for mi, model in enumerate(AB_MODELS):
        if mi > 0:
            lines.append(r"\midrule")

        config_rows = []
        for label, tmpl, shots in CONFIGS:
            d = pull(model, label)
            vals = [d.get(p) for p in PAIRS]
            config_rows.append((label, tmpl, shots, vals, avg_(vals)))

        # Per-language column max within this model block
        block_col_max = []
        for ci in range(n_lang_cols + 1):
            cv = [vals[ci] if ci < n_lang_cols else a
                  for _, _, _, vals, a in config_rows
                  if (vals[ci] if ci < n_lang_cols else a) is not None]
            block_col_max.append(max(cv) if cv else None)

        # Per-language avg row across configs
        col_avgs = []
        for ci in range(n_lang_cols + 1):
            cv = [vals[ci] if ci < n_lang_cols else a
                  for _, _, _, vals, a in config_rows
                  if (vals[ci] if ci < n_lang_cols else a) is not None]
            col_avgs.append(avg_(cv))

        n_block_rows = len(config_rows) + 1
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
            if ri == 0:
                model_cell = (r"\multirow{" + str(n_block_rows) + r"}{*}{"
                              + r"\rotatebox{90}{"
                              + MODEL_DISP.get(model, model)
                              + r"}}")
            else:
                model_cell = ""
            lines.append(f"{model_cell} & {cfg} & "
                         + " & ".join(cells_str) + r" \\")

        lines.append(r"\cmidrule(l){2-" + str(2 + n_lang_cols + 1) + "}")
        avg_cells = [cell_str(v) for v in col_avgs]
        lines.append(r" & \textbf{avg} & "
                     + " & ".join(avg_cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        (r"\caption{Document-level template/shot ablation for the four "
         r"closed APIs (legacy: Gemini-1.5, GPT-4o; latest: Gemini-3.1, "
         r"GPT-5.4) across the 3-template $\times$ 2-shot grid "
         f"({metric_label}). Each row is a (template, shot-count) "
         r"configuration for that model (template number first, shot "
         r"count as subscript $s$). Open LLMs use a single doc-level "
         r"configuration each (no template variants); see "
         r"\Cref{tab:doc_comet} for the open-vs-closed doc-level "
         r"comparison. \textbf{Bold}: per-language column maximum "
         r"within each model block. The bottom \textbf{avg} row of "
         r"each block reports the per-language average across "
         r"configurations.}"),
        r"\label{tab:doc_template_shot_ablation_"
        + ("chrf" if metric == "chrf" else "comet") + "}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def build_prompt_engineering_open(rows, metric=None):
    """**Appendix.** Per-LLM vanilla vs. templated ICL ablation,
    with COMET and chrF shown side-by-side in two column groups, and
    the prompt-vs-shot effects disentangled.

    Layout per row (per metric block):

      V0  V5  T5  Dp  T10  Ds

    where V0 = vanilla 0-shot (basic template), V5 = vanilla 5-shot
    (basic template + random retrieval), T5 = template5_5shot, T10 =
    template5_10shot, Dp = T5 - V5 (prompt effect, shot count fixed
    at 5), Ds = T10 - T5 (shot effect, template fixed at 5).

    Two model groups separated by a midrule: seven open LLMs followed
    by the latest two closed APIs (Gemini-3.1, GPT-5.4). The legacy
    closed pair (Gemini-1.5, GPT-4o) is excluded because they have no
    vanilla 5-shot run, which would collapse V5 / Dp to "--". Open and
    closed groups use different vanilla-5shot tags in summary.csv
    (open: ``icl_5shot``; closed-latest: ``5shot``).

    Only Llama3-8B and Gemma2-9B-IT have T5 data among the open LLMs;
    rows for the other 5 show -- for T5/Dp/Ds. The two latest closed
    APIs have full T5 + V5 coverage, so every column is populated.

    The metric-pair layout exposes the COMET-much-greater-than-chrF
    degeneration signal for AfriqueLlama-8B, AfriqueQwen-8B, and
    TranslateGemma-12B: their COMET deltas are positive but chrF
    deltas are -5.9 to -23.9, indicating fluent off-target generations
    rather than real translations.

    The ``metric`` parameter is accepted but ignored (this builder
    emits a single combined table; the main loop's per-metric
    iteration is special-cased for it).
    """
    OPEN = ["afriquellama_8b", "afriqueqwen_8b", "gemma2_9b_it",
            "llama3_8b", "tiny_aya_earth", "tiny_aya_global",
            "translategemma_12b"]
    # Latest closed APIs only — the legacy pair (Gemini-1.5, GPT-4o) is
    # excluded because they have no vanilla 5-shot run, which collapses
    # the V5 / Delta_{V->T} columns to "--" for those rows.
    CLOSED = ["gemini_31_flash_lite", "gpt54"]

    def by_disp(model_list):
        return sorted(model_list, key=lambda m: MODEL_DISP.get(m, m).lower())

    def avg_(d):
        nums = [v for v in d.values() if v is not None]
        return sum(nums) / len(nums) if nums else None

    def pull_zs(model, m):
        # Legacy closed APIs and the open LLMs store vanilla ZS under
        # llm_zero_shot; the latest closed APIs (gpt54, gemini-3.1)
        # store the same vanilla ZS run under llm_icl/0shot.
        return first_with_pred(rows, metric=m, candidates=[
            dict(model_short=model, experiment_type="llm_zero_shot",
                 is_ablation="False"),
            dict(model_short=model, experiment_type="llm_icl",
                 prompt_strategy="0shot"),
        ])

    def pull_icl(model, strategies, m):
        # ``strategies`` is a tuple of fallback prompt_strategy values
        # tried in order; the first present wins (matches the
        # closed-API split where the latest closed APIs use ``5shot``
        # while the open LLMs use ``icl_5shot``).
        return first_with_pred(rows, metric=m, candidates=[
            dict(model_short=model, experiment_type="llm_icl",
                 prompt_strategy=s)
            for s in strategies
        ])

    # For each model: [(v0, v5, t10, d_v5_t10, d_t5_t10) per metric]
    raw_rows = []
    grouped_models = ([(m, "open")   for m in by_disp(OPEN)] +
                      [(m, "closed") for m in CLOSED])
    for model, group in grouped_models:
        per_metric = []
        v5_strats = ("icl_5shot", "5shot")
        for m in ("ssa_comet", "chrf"):
            v0  = avg_(pull_zs(model, m))
            v5  = avg_(pull_icl(model, v5_strats, m))
            t5  = avg_(pull_icl(model, ("template5_5shot",), m))
            t10 = avg_(pull_icl(model, ("template5_10shot",), m))
            d_v5_t10 = (t10 - v5) if (t10 is not None and v5 is not None) else None
            d_t5_t10 = (t10 - t5) if (t10 is not None and t5 is not None) else None
            per_metric.append((v0, v5, t10, d_v5_t10, d_t5_t10))
        raw_rows.append((model, group, per_metric))

    # Within each group, sort models with complete columns first
    # (no -- in either metric block), then by display name.
    def is_complete(per_metric):
        return all(v is not None for vals in per_metric for v in vals)

    def sort_key(entry):
        model, _, per_metric = entry
        return (0 if is_complete(per_metric) else 1,
                MODEL_DISP.get(model, model).lower())

    table_rows = []
    for grp in ("open", "closed"):
        in_grp = [r for r in raw_rows if r[1] == grp]
        table_rows.extend(sorted(in_grp, key=sort_key))

    # 10 cols: 5 per metric block
    n_per = 5
    n_cols = 2 * n_per
    col_max = []
    for ci in range(n_cols):
        metric_idx = 0 if ci < n_per else 1
        cell_idx = ci % n_per
        vals = [r[2][metric_idx][cell_idx] for r in table_rows
                if r[2][metric_idx][cell_idx] is not None]
        col_max.append(max(vals) if vals else None)

    def fmt_v(v):
        return "--" if v is None else f"{v:.1f}"
    def fmt_d(v):
        return "--" if v is None else f"{v:+.1f}"

    DELTA_COL_IDX = {3, 4}  # both delta columns within a 5-col block

    lines = [
        r"% ---- prompt_engineering_open ----",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{lrrrrrrrrrr}",
        r"\toprule",
        (r"\textbf{Model} & \multicolumn{5}{c}{\textbf{COMET}} & "
         r"\multicolumn{5}{c}{\textbf{chrF}} \\"),
        r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
        (r" & V$_0$ & V$_5$ & T$_{10}$ & $\Delta_{V\to T}$ & $\Delta_{T_{5\to 10}}$ "
         r"& V$_0$ & V$_5$ & T$_{10}$ & $\Delta_{V\to T}$ & $\Delta_{T_{5\to 10}}$ \\"),
        r"\midrule",
    ]
    last_group = None
    last_complete = None
    for model, group, per_metric in table_rows:
        complete = is_complete(per_metric)
        if last_group is not None and group != last_group:
            lines.append(r"\midrule")
        elif (last_complete is not None and complete != last_complete
              and last_group == group):
            # within-group: thin rule between complete- and incomplete-cols
            lines.append(r"\cmidrule(lr){1-" + str(n_cols + 1) + "}")
        last_group = group
        last_complete = complete
        cells_out = []
        ci = 0
        for vals in per_metric:
            for slot, v in enumerate(vals):
                s = fmt_d(v) if slot in DELTA_COL_IDX else fmt_v(v)
                mxv = col_max[ci]
                if v is not None and mxv is not None and abs(v - mxv) < 1e-9:
                    s = r"\textbf{" + s + r"}"
                cells_out.append(s)
                ci += 1
        disp = MODEL_DISP.get(model, model)
        lines.append(f"{disp} & " + " & ".join(cells_out) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{Vanilla vs. templated ICL on the seven open LLMs "
         r"(top) and the two latest closed APIs (bottom: Gemini-3.1, "
         r"GPT-5.4), bidirectional avg over the 12 "
         r"\texttt{eng}$\leftrightarrow$\texttt{xxx} pairs at $T{=}0$, "
         r"COMET and chrF side-by-side. V$_0$: vanilla 0-shot "
         r"(\texttt{basic}); V$_5$: vanilla 5-shot "
         r"(\texttt{few\textunderscore{}shot}, random retrieval); "
         r"T$_{10}$: \texttt{template5\textunderscore{}10shot} (the locked body "
         r"configuration); $\Delta_{V\to T} = $T$_{10} - $V$_5$ is the "
         r"main lift from vanilla to templated; "
         r"$\Delta_{T_{5\to 10}} = $T$_{10} - $T$_5$ isolates the "
         r"shot-count effect (template fixed at \texttt{template5}). "
         r"Within each group, models with all columns populated are "
         r"listed above the within-group divider (Gemma2-9B-IT, "
         r"Llama3-8B for open; Gemini-3.1, GPT-5.4 for closed); the "
         r"remaining open LLMs render ``--'' for T$_5$-derived cells. "
         r"The COMET~$\gg$~chrF divergence on AfriqueLlama-8B, "
         r"AfriqueQwen-8B, and TranslateGemma-12B exposes a degeneration "
         r"signal: large positive $\Delta_{V\to T}$ COMET paired with "
         r"large negative $\Delta_{V\to T}$ chrF "
         r"($-21.7$ / $-5.9$ / $-23.9$), indicating fluent off-target "
         r"generations rather than real translations. \textbf{Bold}: "
         r"column maximum across all 9 models.}"),
        r"\label{tab:prompt_engineering_open}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def build_multilingual_vs_bilingual(rows, metric=None):
    """**Appendix.** Multilingual vs. bilingual training comparison on
    eng->xxx (the only direction MMT covers). Three model families:

      * NLLB-600M: multilingual checkpoints {40k, 80k, 200k, 400k} updates
        compared to the bilingual NLLB-600M and NLLB-1.3B fine-tunes.
      * Llama-3.1-8B-Instruct: same model used for bilingual LoRA and
        multilingual LoRA at matching ranks (apples-to-apples).
      * Gemma-2-9B (non-instruct base): same model used for bilingual
        LoRA and multilingual LoRA at matching ranks (apples-to-apples).

    The chrF and COMET columns appear side-by-side so the metric story
    holds together and the COMET-much-greater-than-chrF degeneration
    signal stays visible.
    """
    PAIRS = ['eng-amh','eng-hau','eng-lug','eng-nso','eng-yor','eng-zul']

    def avg_(vs):
        nums = [v for v in vs if v is not None]
        return sum(nums) / len(nums) if nums else None

    def vec_pair(model, metric, **filters):
        d = lookup(rows, metric=metric, model_short=model, **filters)
        return [d.get(p) for p in PAIRS]

    def row_for(model, label, **filters):
        v_chrf  = vec_pair(model, 'chrf',      **filters)
        v_comet = vec_pair(model, 'ssa_comet', **filters)
        return (label, v_chrf, avg_(v_chrf), v_comet, avg_(v_comet))

    def vec_nllb_ckpt(metric, step):
        # NLLB-600M MMT checkpoints are distinguished only by original_path
        # substring (no schema field). Filter by '{step}_updates'.
        out = {}
        marker = f"{step}_updates"
        for r in rows:
            if r['model_short'] != 'nllb_200_distilled_600m': continue
            if r['experiment_type'] != 'multilingual_finetune': continue
            if marker not in r.get('original_path',''): continue
            if r['lang_pair'] not in PAIRS: continue
            v = metric_val(r, metric)
            if v is not None:
                out[r['lang_pair']] = v
        return [out.get(p) for p in PAIRS]

    def row_nllb_ckpt(label, step):
        v_chrf  = vec_nllb_ckpt('chrf',      step)
        v_comet = vec_nllb_ckpt('ssa_comet', step)
        return (label, v_chrf, avg_(v_chrf), v_comet, avg_(v_comet))

    # --- assemble per-family rows ----------------------------------------
    # Each family is (family_header, [(subblock_label, [(row_label, tuple), ...]), ...])
    # Sub-block label headers a thin sub-rule above the rows; row labels are
    # just the varying parameter (rank, checkpoint, model size).
    families = []

    # NLLB family. is_ablation="False" matters here: NLLB-1.3B has 3 rows
    # per pair in summary.csv (1 headline + 2 ablations), and without the
    # filter `lookup` returns whichever was loaded last (an ablation).
    nllb_bi = [
        ("NLLB-1.3B",
         row_for("nllb_200_1_3b", "",
                 experiment_type="seq2seq_finetune", dataset="dcs",
                 is_ablation="False")),
        ("NLLB-600M",
         row_for("nllb_200_distilled_600m", "",
                 experiment_type="seq2seq_finetune", dataset="dcs",
                 is_ablation="False")),
    ]
    nllb_mmt = [(f"@ {step}", row_nllb_ckpt("", step))
                for step in ("40k", "80k", "200k", "400k")]
    families.append((
        "NLLB-600M / 1.3B",
        [("Bilingual fine-tune", nllb_bi),
         ("Multilingual fine-tune (NLLB-600M, training updates)", nllb_mmt)],
    ))

    # Llama-3.1-8B-Instruct family (apples-to-apples at matching ranks)
    llama_bi = [(f"$r{{=}}{rk}$",
                 row_for("llama3_8b", "",
                         experiment_type="lora_finetune", lora_rank=str(rk)))
                for rk in (8, 16, 32, 64, 128)]
    llama_mmt = [(f"$r{{=}}{rk}$",
                  row_for("llama3_8b", "",
                          experiment_type="multilingual_lora",
                          lora_rank=str(rk)))
                 for rk in (8, 16, 32, 64, 128)]
    families.append((
        "Llama-3.1-8B-Instruct (same model both sides, LoRA rank sweep)",
        [("Bilingual LoRA", llama_bi),
         ("Multilingual LoRA", llama_mmt)],
    ))

    # Gemma-2-9B (non-instruct base) family
    gemma_bi = [(f"$r{{=}}{rk}$",
                 row_for("gemma2_9b", "",
                         experiment_type="lora_finetune", lora_rank=str(rk)))
                for rk in (8, 16, 32, 64, 128)]
    gemma_mmt = [(f"$r{{=}}{rk}$",
                  row_for("gemma2_9b", "",
                          experiment_type="multilingual_lora",
                          lora_rank=str(rk)))
                 for rk in (8, 16, 32, 64, 128)]
    families.append((
        "Gemma-2-9B non-instruct (same base both sides, LoRA rank sweep)",
        [("Bilingual LoRA", gemma_bi),
         ("Multilingual LoRA", gemma_mmt)],
    ))

    # --- column maxes (per metric, across the whole table) ----------------
    # Per-row tuple: (label, v_chrf, a_chrf, v_comet, a_comet).
    # Layout: 6 chrf eng->xxx + chrF Avg + 6 comet eng->xxx + COMET Avg.
    n_pairs = len(PAIRS)
    n_cols = 2 * (n_pairs + 1)
    all_rows = [tup for _, subs in families for _, rs in subs for _, tup in rs]

    def cell_at(tup, ci):
        _, v_chrf, a_chrf, v_comet, a_comet = tup
        if   ci <  n_pairs:        return v_chrf[ci]
        elif ci == n_pairs:        return a_chrf
        elif ci < 2 * n_pairs + 1: return v_comet[ci - n_pairs - 1]
        else:                      return a_comet

    col_max = []
    for ci in range(n_cols):
        vals = [v for tup in all_rows
                if (v := cell_at(tup, ci)) is not None]
        col_max.append(max(vals) if vals else None)

    def fmt_v(v):
        return "--" if v is None else f"{v:.1f}"

    lang_cols = [r"\textbf{\texttt{" + l + "}}"
                 for l in ['amh','hau','lug','nso','yor','zul']]
    n_data_cols = n_cols + 1  # +1 for the leading Setup column

    lines = [
        r"% ---- multilingual_vs_bilingual ----",
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{l" + "r" * (n_pairs + 1) + "r" * (n_pairs + 1) + r"}",
        r"\toprule",
        (r"\textbf{Setup} & \multicolumn{7}{c}{\textbf{chrF}} & "
         r"\multicolumn{7}{c}{\textbf{COMET}} \\"),
        r"\cmidrule(lr){2-8}\cmidrule(lr){9-15}",
        (r" & " + " & ".join(lang_cols) + r" & \textbf{Avg}"
         r" & " + " & ".join(lang_cols) + r" & \textbf{Avg} \\"),
        r"\midrule",
    ]
    for fi, (fam_name, subs) in enumerate(families):
        if fi > 0:
            lines.append(r"\midrule")
        lines.append(
            r"\multicolumn{" + str(n_data_cols) + r"}{l}{\textbf{"
            + fam_name + r"}} \\"
        )
        for si, (sub_label, rs) in enumerate(subs):
            lines.append(r"\cmidrule(l){1-" + str(n_data_cols) + r"}")
            lines.append(
                r"\multicolumn{" + str(n_data_cols) + r"}{l}{\textit{\,"
                + sub_label + r"}} \\"
            )
            for row_label, tup in rs:
                cells_out = []
                for ci in range(n_cols):
                    v = cell_at(tup, ci)
                    s = fmt_v(v)
                    mxv = col_max[ci]
                    if v is not None and mxv is not None and abs(v - mxv) < 1e-9:
                        s = r"\textbf{" + s + r"}"
                    cells_out.append(s)
                lines.append(f"\\quad {row_label} & " + " & ".join(cells_out) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{Bilingual vs. multilingual fine-tuning on "
         r"\texttt{eng}$\to$\texttt{xxx} (chrF and COMET side-by-side). "
         r"Three model families: NLLB-600M (multilingual checkpoint "
         r"sweep against NLLB-600M and 1.3B bilingual fine-tunes); "
         r"Llama-3.1-8B-Instruct (same instruct model on both sides); "
         r"Gemma-2-9B non-instruct base. For NLLB, bilingual "
         r"fine-tuning is substantially stronger than the multilingual "
         r"checkpoints. For Gemma LoRA, bilingual leads by $\sim$3.7 "
         r"COMET / $\sim$5 chrF at the best rank. For Llama LoRA, "
         r"bilingual and multilingual end up essentially tied at COMET "
         r"($\sim$57). \textbf{Bold}: best score for that target "
         r"language (or Avg) within each metric, across all rows of "
         r"the table.}"),
        r"\label{tab:multilingual_vs_bilingual}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def build_closed_evolution(rows, metric=None):
    """**Appendix.** Closed-API evolution: legacy (gpt-4o, gemini-1.5-flash)
    against latest (gpt-5.4, gemini-3.1-flash-lite) at the headline
    sentence-level and document-level configurations. Single combined
    table with chrF and SSA-COMET side-by-side and a per-row Avg.

    The point of this table is to make the closed-API improvement-over-time
    visible at a glance, so a reader can see why the body's "fine-tuning
    beats closed APIs" finding (anchored on the legacy column) softens at
    the latest column.

    Layout, per row (metric pair):
       chrF eng-> | chrF ->eng | chrF Avg | COMET eng-> | COMET ->eng | COMET Avg

    Two blocks:
      - Sentence-level at the locked template5_10shot config (T=0).
      - Document-level at doc_0shot vanilla (matches what tab:doc_comet
        reports for these models).
    """
    PAIRS_E2X = ['eng-amh','eng-hau','eng-lug','eng-nso','eng-yor','eng-zul']
    PAIRS_X2E = ['amh-eng','hau-eng','lug-eng','nso-eng','yor-eng','zul-eng']

    # Family + version ordering so each family reads legacy -> latest -> Delta
    # top-to-bottom. Renders with \multirow on the Model column so the
    # evolution is visible at a glance.
    FAMILIES = [
        ("Gemini", [("1.5", "gemini_15_flash"),
                    ("3.1", "gemini_31_flash_lite")]),
        ("GPT",    [("4o",  "gpt4o"),
                    ("5.4", "gpt54")]),
    ]
    ALL = [m for _, vs in FAMILIES for _, m in vs]

    def avg_(vs):
        nums = [v for v in vs if v is not None]
        return sum(nums) / len(nums) if nums else None

    def cell_avg(model, *, metric_, etype, strategies, pairs):
        """Average a metric over a set of pairs for one (model, strategy).
        ``strategies`` is a tuple of candidate prompt_strategy values; the
        first one with data wins (matches build_doc's first_with_pred
        fallback so latest models use ``doc_0shot`` and legacy models fall
        through to ``doc_0shot_legacy``)."""
        d = first_with_pred(rows, metric=metric_, candidates=[
            dict(model_short=model, experiment_type=etype, prompt_strategy=s)
            for s in strategies
        ])
        return avg_([d.get(p) for p in pairs])

    def sent_row(model):
        # Sent-level: template5_10shot at T=0 (locked headline).
        out = {}
        for metric_ in ("chrf", "ssa_comet"):
            out[(metric_, "e2x")] = cell_avg(
                model, metric_=metric_, etype="llm_icl",
                strategies=("template5_10shot",), pairs=PAIRS_E2X)
            out[(metric_, "x2e")] = cell_avg(
                model, metric_=metric_, etype="llm_icl",
                strategies=("template5_10shot",), pairs=PAIRS_X2E)
        return out

    def doc_row(model):
        # Doc-level "vanilla" (basic prompt, no template variant). Latest
        # closed APIs land at doc_0shot; legacy closed APIs only have
        # doc_0shot_legacy (the original pre-template doc runs) -- same
        # fallback build_doc uses so closed-API rows in the headline
        # tab:doc_comet match what we report here.
        out = {}
        for metric_ in ("chrf", "ssa_comet"):
            out[(metric_, "e2x")] = cell_avg(
                model, metric_=metric_, etype="llm_doc",
                strategies=("doc_0shot", "doc_0shot_legacy"),
                pairs=PAIRS_E2X)
            out[(metric_, "x2e")] = cell_avg(
                model, metric_=metric_, etype="llm_doc",
                strategies=("doc_0shot", "doc_0shot_legacy"),
                pairs=PAIRS_X2E)
        return out

    # Pre-build all rows so we can compute column maxes.
    sent_rows = [(m, sent_row(m)) for m in ALL]
    doc_rows  = [(m, doc_row(m))  for m in ALL]

    def fmt(v):
        return "--" if v is None else f"{v:.1f}"

    def to_cells(row, group_label):
        # Returns 6 cells: chrF e2x, x2e, Avg, COMET e2x, x2e, Avg.
        ch_e = row[("chrf", "e2x")]
        ch_x = row[("chrf", "x2e")]
        co_e = row[("ssa_comet", "e2x")]
        co_x = row[("ssa_comet", "x2e")]
        ch_avg = avg_([ch_e, ch_x])
        co_avg = avg_([co_e, co_x])
        return [ch_e, ch_x, ch_avg, co_e, co_x, co_avg]

    def col_maxes(block_rows):
        cols = list(zip(*[to_cells(r, lbl) for lbl, r in block_rows]))
        return [max((v for v in c if v is not None), default=None) for c in cols]

    def delta_cells(legacy_row, latest_row):
        l = to_cells(legacy_row, "")
        n = to_cells(latest_row, "")
        return [(b - a) if (a is not None and b is not None) else None
                for a, b in zip(l, n)]

    def fmt_d(v):
        return "--" if v is None else f"{v:+.1f}"

    sent_maxes = col_maxes([(MODEL_DISP.get(m, m), r) for m, r in sent_rows])
    doc_maxes  = col_maxes([(MODEL_DISP.get(m, m), r) for m, r in doc_rows])
    by_model = {"sent": dict(sent_rows), "doc": dict(doc_rows)}

    n_data_cols = 6
    n_cols = 2 + n_data_cols  # Model + Version + 6 data

    def render_block_rows(block_key, maxes):
        out = []
        for fi, (fam, versions) in enumerate(FAMILIES):
            if fi > 0:
                out.append(r"\cmidrule(lr){1-" + str(n_cols) + "}")
            # 3 rows per family: legacy, latest, Delta
            rowspan = len(versions) + 1
            rows_by_ver = [(ver, by_model[block_key][model])
                           for ver, model in versions]
            d = delta_cells(rows_by_ver[0][1], rows_by_ver[-1][1])
            for vi, (ver, rdata) in enumerate(rows_by_ver):
                cells = to_cells(rdata, "")
                cell_strs = []
                for v, mx in zip(cells, maxes):
                    s = fmt(v)
                    if v is not None and mx is not None and abs(v - mx) < 1e-9:
                        s = r"\textbf{" + s + r"}"
                    cell_strs.append(s)
                if vi == 0:
                    fam_cell = r"\multirow{" + str(rowspan) + r"}{*}{" + fam + r"}"
                else:
                    fam_cell = ""
                out.append(fam_cell + " & " + ver + " & " + " & ".join(cell_strs) + r" \\")
            out.append(r" & $\Delta$ & " + " & ".join(fmt_d(v) for v in d) + r" \\")
        return out

    lines = [
        r"% ---- closed_evolution ----",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{ll" + "r" * n_data_cols + r"}",
        r"\toprule",
        (r"\textbf{Model} & \textbf{Version} & \multicolumn{3}{c}{\textbf{chrF}}"
         r" & \multicolumn{3}{c}{\textbf{COMET}} \\"),
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        (r" &  & \texttt{eng$\to$} & \texttt{$\to$eng} & \textbf{Avg}"
         r" & \texttt{eng$\to$} & \texttt{$\to$eng} & \textbf{Avg} \\"),
        r"\midrule",
        r"\multicolumn{" + str(n_cols) + r"}{l}{\textit{"
        r"Sentence-level ICL (template~5, 10-shot, $T{=}0$)}} \\",
        r"\midrule",
    ]
    lines.extend(render_block_rows("sent", sent_maxes))
    lines += [
        r"\midrule\midrule",
        r"\multicolumn{" + str(n_cols) + r"}{l}{\textit{"
        r"Document-level (vanilla \texttt{doc\textunderscore{}0shot})}} \\",
        r"\midrule",
    ]
    lines.extend(render_block_rows("doc", doc_maxes))
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{Closed-API evolution, grouped by family (Gemini, GPT) "
         r"with legacy $\to$ latest top-to-bottom and a $\Delta$ row "
         r"reporting (latest $-$ legacy) per column. Two blocks: the main "
         r"sentence-level configuration (\texttt{template5\textunderscore{}10shot}, "
         r"$T{=}0$) and the main document-level configuration (vanilla "
         r"\texttt{doc\textunderscore{}0shot}). Per-direction averages over the six "
         r"target languages, plus an overall Avg. The $\Delta$ rows make "
         r"the closed-API generation-over-generation improvement visible "
         r"at a glance; both families improve on COMET in both directions, "
         r"with the doc-level COMET gain (+13--16) much larger than the "
         r"sentence-level gain (+2--3). \textbf{Bold}: column maximum "
         r"within each block.}"),
        r"\label{tab:closed_evolution}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def build_doc_evaluation():
    """**Appendix.** LLM-judge evaluation of all four closed APIs on
    the six common doc-level template configurations
    (\\texttt{doc\\_\\{0,1\\}shot\\_template\\{1,2,3\\}}).

    Scores are the per-cell mean of two judges (GPT-5.5 reading
    ``judge.json`` and Gemini-3.1-Flash-Lite reading
    ``judge_gemini.json``). The table contains two direction
    sub-blocks (\\texttt{eng}$\\to$\\texttt{xxx},
    \\texttt{xxx}$\\to$\\texttt{eng}); within each sub-block, all four
    closed APIs are listed (Gemini-1.5, Gemini-3.1, GPT-4o, GPT-5.4),
    each with one row per (template, shot) configuration plus a
    per-model avg row.

    Restricted to the six configs common to legacy and latest closed
    APIs so legacy-vs-latest comparison is apples-to-apples (legacy
    has additional 5-shot configs and latest has vanilla 0/1-shot;
    both are excluded). Emitted as a longtable so the ~56 data rows
    can break across pages.
    """
    import json
    from pathlib import Path

    JUDGE_FILES = ("judge.json", "judge_gemini.json")
    # Family-grouped: Gemini family then GPT family; within each,
    # legacy then latest.
    models = [
        ("Gemini-1.5", "gemini_15_flash"),
        ("Gemini-3.1", "gemini_31_flash_lite"),
        ("GPT-4o",     "gpt4o"),
        ("GPT-5.4",    "gpt54"),
    ]

    # Six configurations common to all four closed APIs.
    STRATEGIES = [
        ("1_0", "doc_0shot_template1"), ("1_1", "doc_1shot_template1"),
        ("2_0", "doc_0shot_template2"), ("2_1", "doc_1shot_template2"),
        ("3_0", "doc_0shot_template3"), ("3_1", "doc_1shot_template3"),
    ]
    METRICS = [
        ("fluency_mean",              "fluency"),
        ("accuracy_mistakes_mean",    "accuracy errors"),
        ("lexical_mistakes_mean",     "lexical mistakes"),
        ("grammatical_mistakes_mean", "grammatical mistakes"),
    ]
    LANGS_LOCAL = ["amh", "hau", "lug", "nso", "yor", "zul"]
    DIRECTIONS = [
        ("e2x", r"eng$\to$"),
        ("x2e", r"$\to$eng"),
    ]

    def judge_value(model_short, strat_dir, lang_pair, metric_key, judge_file):
        p = (Path("experiments/llm/closed/doc") / model_short / strat_dir
             / lang_pair / judge_file)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        agg = d.get("aggregate") or {}
        v = agg.get(metric_key)
        return v if isinstance(v, (int, float)) else None

    def averaged_value(model_short, strat_dir, lang_pair, metric_key):
        vs = [judge_value(model_short, strat_dir, lang_pair,
                          metric_key, jf) for jf in JUDGE_FILES]
        vs = [v for v in vs if v is not None]
        return (sum(vs) / len(vs)) if vs else None

    def fmt_v(v):
        return "--" if v is None else f"{v:.1f}"

    # Pre-compute every cell:
    #   data[direction][model_short][strat_label][metric_key][lang]
    data = {}
    for direction_key, _ in DIRECTIONS:
        data[direction_key] = {}
        for _, model_short in models:
            data[direction_key][model_short] = {}
            for strat_label, strat_dir in STRATEGIES:
                data[direction_key][model_short][strat_label] = {}
                for metric_key, _ in METRICS:
                    data[direction_key][model_short][strat_label][metric_key] = {}
                    for lang in LANGS_LOCAL:
                        pair = (f"eng-{lang}" if direction_key == "e2x"
                                else f"{lang}-eng")
                        data[direction_key][model_short][strat_label][metric_key][lang] = \
                            averaged_value(model_short, strat_dir, pair, metric_key)

    n_data = 6 * len(METRICS)
    n_cols = 3 + n_data    # direction + model + setting + data
    col_spec = "lll|" + "|".join(["rrrrrr"] * len(METRICS))

    lang_hdr_chunk = " & ".join(r"\textbf{\texttt{" + l + r"}}"
                                for l in LANGS_LOCAL)

    lines = [
        r"% ---- doc_evaluation ----",
        r"\begin{table*}[t]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        (r"\multirow{2}{*}{\textbf{dir}} & "
         r"\multirow{2}{*}{\textbf{model}} & "
         r"\multirow{2}{*}{\textbf{setting}} & "
         + " & ".join(r"\multicolumn{6}{l|}{\textbf{" + label + r"}}"
                      if i < len(METRICS) - 1
                      else r"\multicolumn{6}{l}{\textbf{" + label + r"}}"
                      for i, (_, label) in enumerate(METRICS))
         + r" \\"),
        " ".join(
            (r"\cmidrule(lr){" + str(4 + 6*i) + "-" + str(9 + 6*i) + "}")
            for i in range(len(METRICS))
        ),
        (r"& & & " + " & ".join([lang_hdr_chunk] * len(METRICS)) + r" \\"),
        r"\midrule",
    ]

    def cells_for_row(direction_key, model_short, strat_label):
        out = []
        for metric_key, _ in METRICS:
            for lang in LANGS_LOCAL:
                v = data[direction_key][model_short][strat_label][metric_key][lang]
                out.append(fmt_v(v))
        return out

    def avg_cells_for_model(direction_key, model_short):
        out = []
        for metric_key, _ in METRICS:
            for lang in LANGS_LOCAL:
                vs = [data[direction_key][model_short][s][metric_key][lang]
                      for s, _ in STRATEGIES
                      if data[direction_key][model_short][s][metric_key][lang] is not None]
                v = (sum(vs) / len(vs)) if vs else None
                out.append(fmt_v(v))
        return out

    rows_per_model = len(STRATEGIES) + 1   # 6 strategies + avg
    rows_per_direction = rows_per_model * len(models)

    for di, (dir_key, dir_disp) in enumerate(DIRECTIONS):
        if di > 0:
            lines.append(r"\midrule")
        # The direction cell spans every row of this sub-block.
        for mi, (mdisp, ms) in enumerate(models):
            if mi > 0:
                lines.append(r"\cmidrule{2-" + str(n_cols) + "}")
            for ri, (strat_label, _) in enumerate(STRATEGIES):
                row_cells = cells_for_row(dir_key, ms, strat_label)
                # direction multirow on the very first row of the block
                if mi == 0 and ri == 0:
                    dir_cell = (r"\multirow{" + str(rows_per_direction)
                                + r"}{*}{" + dir_disp + r"}")
                else:
                    dir_cell = ""
                # model multirow on the first row of each model's sub-section
                if ri == 0:
                    model_cell = (r"\multirow{" + str(rows_per_model)
                                  + r"}{*}{" + mdisp + r"}")
                else:
                    model_cell = ""
                tmpl, shot = strat_label.split("_")
                row_label = f"${tmpl}_{{{shot}}}$"
                lines.append(f"{dir_cell} & {model_cell} & {row_label} & "
                             + " & ".join(row_cells) + r" \\")
            # per-model avg row inside this model's sub-section
            lines.append(r"\cmidrule{3-" + str(n_cols) + "}")
            lines.append(" & & avg & "
                         + " & ".join(avg_cells_for_model(dir_key, ms))
                         + r" \\")

    caption = (
        r"\caption{LLM-judge evaluation of the four closed APIs' "
        r"document-level translations on the six template configurations "
        r"common to legacy and latest "
        r"(\texttt{doc\textunderscore{}\{0,1\}shot\textunderscore{}template\{1,2,3\}}). "
        r"Scores are the per-cell mean of two judges (GPT-5.5 and "
        r"Gemini-3.1-Flash-Lite). Higher fluency is better; lower "
        r"accuracy / lexical / grammatical mistake counts are better.}"
    )
    label = r"\label{tab:doc_evaluation}"
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        caption,
        label,
        r"\end{table*}",
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
        ("doc_template_shot_ablation", build_doc_template_shot_ablation),
    ]
    # Builders that emit a single combined table (no per-metric mirror).
    combined_builders = [
        ("prompt_engineering_open",   build_prompt_engineering_open),
        ("multilingual_vs_bilingual", build_multilingual_vs_bilingual),
        ("closed_evolution",          build_closed_evolution),
    ]
    written = []
    for name, fn in builders:
        for metric in ("chrf", "ssa_comet"):
            suffix = "chrf" if metric == "chrf" else "comet"
            content = fn(rows, metric)
            out = OUT_DIR / f"{name}_{suffix}.tex"
            out.write_text(content)
            written.append(out)
    for name, fn in combined_builders:
        # Stale per-metric files from a previous build can confuse readers;
        # remove them now that the combined table replaces them.
        for stale in (OUT_DIR / f"{name}_chrf.tex",
                      OUT_DIR / f"{name}_comet.tex"):
            if stale.exists():
                stale.unlink()
        content = fn(rows)
        out = OUT_DIR / f"{name}.tex"
        out.write_text(content)
        written.append(out)

    # Standalone appendix-paragraph file that accompanies tab:closed_ablations_*.
    notes_path = OUT_DIR / "closed_ablations_notes.tex"
    notes_path.write_text(build_closed_ablations_notes())
    written.append(notes_path)

    # Doc-level LLM-judge table: all four closed APIs (Gemini family
    # then GPT family), 2 judges averaged, both directions in one table.
    for stale in (OUT_DIR / "doc_evaluation_gpt.tex",
                  OUT_DIR / "doc_evaluation_gemini.tex"):
        if stale.exists():
            stale.unlink()
    out = OUT_DIR / "doc_evaluation.tex"
    out.write_text(build_doc_evaluation())
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
