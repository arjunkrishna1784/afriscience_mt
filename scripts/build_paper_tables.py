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
    "m2m100_418m":            "m2m100\\textunderscore{}418M",
    "m2m100_1_2b":            "m2m100\\textunderscore{}1.2B",
    "nllb_200_distilled_600m":"nllb\\textunderscore{}600M",
    "nllb_200_1_3b":          "nllb\\textunderscore{}1.3B",
    "afriquellama_8b":        "AfriqueLlama",
    "afriqueqwen_8b":         "AfriqueQwen",
    "gemma2_9b_it":           "Gemma2-9B",
    "llama3_8b":              "Llama3-8B",
    "tiny_aya_earth":         "Tiny-Aya-E",
    "tiny_aya_global":        "Tiny-Aya-G",
    "translategemma_12b":     "TransGemma",
    "gpt4o":                  "GPT-4o",
    "gemini_15_flash":        "Gemini-1.5",
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


def render_table(*, label, caption, blocks):
    """blocks = [(setup_label, [(model_short, [12 floats|None]), ...]), ...]
    Bolds the per-column max **within each block** (matches new_results.tex)."""
    # per-block column maxima
    block_col_max = []
    for _, rs in blocks:
        cm = []
        for ci in range(12):
            vals = [r[1][ci] for r in rs if r[1][ci] is not None]
            cm.append(max(vals) if vals else None)
        block_col_max.append(cm)

    head_row = (
        " & ".join([" "] * 2 + [LANG_DISP[l] for l in LANGS] * 2) + r" \\"
    )

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
                if v is not None and col_max[ci] is not None and abs(v - col_max[ci]) < 1e-9:
                    s = f"\\textbf{{{s}}}"
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
        caption=f"Seq2Seq: zero-shot vs.\\ fine-tuned ({metric_label})",
        blocks=_seq2seq_blocks(rows, metric, SEQ2SEQ_MODELS_MAIN),
    )


def build_seq2seq_appendix(rows, metric):
    """Appendix seq2seq table: smaller backbones (M2M100-418M, NLLB-600M)."""
    metric_label = "chrF" if metric == "chrf" else "COMET"
    return render_table(
        label=f"tab:seq2seq_smaller_{'chrf' if metric=='chrf' else 'comet'}",
        caption=("Smaller seq2seq backbones (M2M100-418M, NLLB-600M): "
                 f"zero-shot vs.\\ fine-tuned ({metric_label}). "
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
        caption=f"LLM sentence-level: zero-shot vs.\\ 5-shot ICL ({metric_label})",
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
        caption=f"LoRA fine-tuning: $r{{=}}4$ vs.\\ $r{{=}}64$ ({metric_label})",
        blocks=[("LoRA $r{=}4$", r4_rows), ("LoRA $r{=}64$", r64_rows)],
    )


def build_doc(rows, metric):
    metric_label = "chrF" if metric == "chrf" else "COMET"
    zs_rows = []
    one_rows = []
    for m in LLM_MODELS:
        # accept both new (doc_0shot) and legacy (doc_0shot_legacy) prompt strategies
        d_zs = first_with_pred(rows, metric=metric, candidates=[
            dict(experiment_type="llm_doc", prompt_strategy="doc_0shot",
                 model_short=m, is_ablation="False"),
            dict(experiment_type="llm_doc", prompt_strategy="doc_0shot_legacy",
                 model_short=m, is_ablation="False"),
        ])
        d_one = first_with_pred(rows, metric=metric, candidates=[
            dict(experiment_type="llm_doc", prompt_strategy="doc_1shot",
                 model_short=m, is_ablation="False"),
            dict(experiment_type="llm_doc", prompt_strategy="doc_1shot_legacy",
                 model_short=m, is_ablation="False"),
        ])
        zs_rows.append((m, vec(d_zs)))
        one_rows.append((m, vec(d_one)))
    return render_table(
        label=f"tab:doc_{'chrf' if metric=='chrf' else 'comet'}",
        caption=f"Document-level: zero-shot vs.\\ 1-shot ({metric_label})",
        blocks=[("Zero-shot", zs_rows), ("1-shot", one_rows)],
    )


# ---------------- main ----------------

def main():
    rows = load_rows()
    builders = [
        ("seq2seq",          build_seq2seq),
        ("seq2seq_smaller",  build_seq2seq_appendix),
        ("llm_sentence",     build_llm_sentence),
        ("lora",             build_lora),
        ("doc",              build_doc),
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
