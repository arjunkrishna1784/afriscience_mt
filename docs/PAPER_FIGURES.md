# Paper figures and tables — index for the new draft

All figure PDFs live under
[`_ACL2026__.../figures/new/`](../_ACL2026__Decolonizing_Science_in_Africa_through_Text_Translation_/figures/new/);
all `.tex` tables live under [`tables/new/`](../tables/new/). Titles,
suptitles, and on-figure captions are intentionally stripped from the
PDFs — drop captions in via the LaTeX `\caption{}` of the surrounding
`figure` (or `table*`) environment.

> **Authoring rule.** Every new figure or table added below **must** carry
> a "Suggested LaTeX caption" so the draft author can drop it into the
> paper without re-deriving what the asset shows. If you add a row,
> add a caption.

Re-build any of these from scratch:

```bash
python3 scripts/build_paper_assets.py    # zs_vs_ft_delta, lora_rank_curve
python3 scripts/build_paper_tables.py    # 8 new_results.tex-style tables
python3 scripts/build_balance_assets.py  # §4 balance ablation assets
python3 scripts/build_figure4.py         # closed-API template/shot curves
python3 scripts/redraw_paper_plots.py    # heatmaps + spider charts
```

The metric hierarchy is **COMET primary, ChrF secondary (appendix /
ablation)**. Where a figure exists in both metrics, the layout
mirrors: `<root>/comet/<name>.pdf` and `<root>/chrf/<name>.pdf` carry the
same filename so the LaTeX include can be swapped by changing one path
component.

---

## Main paper

### Figures

| File | What it shows | Suggested LaTeX caption |
|---|---|---|
| `zs_vs_ft_delta.pdf` | Per-language COMET (avg both directions) for NLLB-1.3B and LLaMA-3.1-8B, before vs after fine-tuning. | Effect of in-domain fine-tuning on AfriScience-MT for the strongest seq2seq backbone (NLLB-1.3B) and the per-pair LoRA on LLaMA-3.1-8B. Bars show pre-/post-FT COMET averaged over both translation directions. |
| `lora_rank_curve.pdf` | Avg COMET vs LoRA rank (LLaMA-3.1-8B), separate Eng→X and X→Eng curves. | LoRA rank vs. COMET on LLaMA-3.1-8B fine-tuned on AfriScience-MT. Note: $r{=}4$ and $r{=}64$ use the vanilla \texttt{Llama-3.1-8B-Instruct} base; other ranks use Unsloth 4-bit, so absolute levels across bases are not directly comparable. |
| `gpt-vs-gemini/comet/eng_xxx.pdf` | 6 lang × (Gemini \| GPT-4o), COMET vs shots, lines for Templates 1–5 + Template 5 (sem.). | COMET on Eng→xxx for Gemini-1.5-Flash and GPT-4o across five prompt templates and shot counts $\in\{0,5,10,20,50\}$. Template 5 (sem. prompt) is the semantic-similarity-retrieved variant available only for this direction. |
| `gpt-vs-gemini/comet/xxx_eng.pdf` | Same shape for X→Eng. | COMET on xxx→Eng for Gemini-1.5-Flash and GPT-4o across five prompt templates and shot counts. |
| `fig7/comet/m2m100_1_2b.pdf` | M2M100-1.2B 7×7 panels: zero-shot \| fine-tuned, COMET. Fine-tuned X↔X cells empty. | Sentence-level COMET for M2M100-1.2B before and after fine-tuning on AfriScience-MT. African↔African fine-tunes were not run for the larger backbone. |
| `fig7/comet/nllb_1_3b.pdf` | NLLB-1.3B 7×7, same caveat. | Sentence-level COMET for NLLB-1.3B before and after fine-tuning on AfriScience-MT. |
| `spider_per_pair/3sys/comet/<pair>.pdf` (12 files) | Per-pair radar: NLLB-1.3B (FT) vs Gemma2-9B (LoRA $r{=}64$) vs GPT-4o (template-3, 5-shot ICL). 11 domains. **Populated only after `scripts/rescore_per_domain_ssa.py` runs on a GPU.** | Per-domain COMET for NLLB-1.3B, Gemma2-9B (LoRA), and GPT-4o (5-shot ICL) on the 12 directional pairs. Solid: NLLB-1.3B, dashed: Gemma2-9B, dotted: GPT-4o. |
| `spider_per_pair/4sys/comet/<pair>.pdf` (12 files) | Same plus Tiny-Aya-E (LoRA $r{=}64$). | Per-domain COMET adding Tiny-Aya-E LoRA $r{=}64$ as a fourth comparator. |

### Tables

| File | Suggested LaTeX caption |
|---|---|
| `seq2seq_comet.tex` (`tab:seq2seq_comet`) | Sentence-level COMET on AfriScience-MT for the larger Seq2Seq backbones (M2M100-1.2B, NLLB-1.3B), zero-shot vs. fine-tuned, across all 12 directional pairs. Best per column in bold. |
| `llm_sentence_comet.tex` (`tab:llm_sentence_comet`) | Sentence-level COMET for open-source LLMs in zero-shot and 5-shot ICL across all 12 pairs. Best per column in bold. |
| `lora_comet.tex` (`tab:lora_comet`) | LoRA fine-tuning of open-source LLMs on AfriScience-MT, $r{=}4$ vs. $r{=}64$, COMET. Best per column in bold. |
| `doc_comet.tex` (`tab:doc_comet`) | Document-level COMET for open-source LLMs, zero-shot vs. 1-shot. Best per column in bold. |
| `balance_comet.tex` (`tab:balance_comet`) | Domain-balance ablation on NLLB-1.3B fine-tuned on AfriScience-MT (COMET). $\tau{=}1$ uses the natural domain proportions; $\tau{=}2$ applies sqrt smoothing; $\tau{=}\infty$ samples uniformly across all 11 domains. Best per column in bold. |
| `mafand_nllb.tex` (`tab:mafand_nllb`) | Cross-domain transfer for NLLB-1.3B fine-tuned on AfriScience-MT (AS-MT), MAFAND-MT (MF), or their union (\texttt{+MF}); COMET (top block) and ChrF (bottom block) on the four MAFAND-supported language pairs in both directions. Best per column within each metric block is in bold. **Note:** the two metrics disagree — under COMET, AS-MT alone wins every cell over AS-MT+MF; under ChrF, AS-MT+MF wins most cells. Worth one sentence in §3 prose. |

---

## §4 Domain-balance ablation

### Figures

| File | What it shows | Suggested LaTeX caption |
|---|---|---|
| `balance_heatmap.pdf` | 3-row × 11-column heatmap. Rows = $\tau\in\{1,2,\infty\}$; columns = domains ordered by training-set size $N_d$; cells = mean ChrF over the 12 directional pairs. | Per-domain ChrF for NLLB-1.3B fine-tuned with three temperature-sampling settings ($\tau{=}1$ natural, $\tau{=}2$ sqrt-smoothed, $\tau{=}\infty$ uniform). Domains ordered by training-set size; balancing hurts every domain. |
| `balance_delta_by_size.pdf` | Bars of $\Delta$ ChrF $= \mathrm{ChrF}(\tau{=}\infty) - \mathrm{ChrF}(\tau{=}1)$ per domain. | Effect of uniform domain sampling on ChrF, by domain size. The hypothesis that low-resource domains gain from balancing is rejected — every bar is negative. |

### Tables

| File | Suggested LaTeX caption |
|---|---|
| `balance_domain_x_lang_eng_xxx.tex` (`tab:balance_domain_eng_xxx`) | Domain-level ChrF for NLLB-1.3B on Eng→xxx with uniform-temperature sampling ($\tau{=}\infty$). Numbers in parentheses are the change relative to the natural-distribution baseline ($\tau{=}1$); blue = improvement, red = decrease. Total training volume is identical across the two conditions (5\,792 sentences) — only the per-example sampling weight changes. Needs `\usepackage{xcolor}`. |
| `balance_domain_x_lang_eng_xxx_dual.tex` (`tab:balance_domain_eng_xxx_dual`) | Same content as above but with **two subtables** — ChrF (top) and COMET (bottom) — matching the page-26 Table 7 layout. Needs `\usepackage{xcolor}` and `\usepackage{subcaption}`. |

---

## Appendix

### Figures

| File | What it shows | Suggested LaTeX caption |
|---|---|---|
| `gpt-vs-gemini/chrf/eng_xxx.pdf` | ChrF mirror of `gpt-vs-gemini/comet/eng_xxx.pdf`. | ChrF scores comparison of GPT and Gemini models across different templates and in-context examples on Eng→xxx (cross-metric reference for the COMET version above). |
| `gpt-vs-gemini/chrf/xxx_eng.pdf` | ChrF mirror of `gpt-vs-gemini/comet/xxx_eng.pdf`. | ChrF scores comparison of GPT and Gemini models across different templates and in-context examples on xxx→Eng. |
| `fig7/chrf/m2m100_1_2b.pdf` | ChrF mirror of fig7 main. | ChrF for M2M100-1.2B zero-shot vs. fine-tuned (cf. Fig.~\ref{fig:fig7_m2m100_1_2b_comet}). |
| `fig7/chrf/nllb_1_3b.pdf` | ChrF mirror of fig7 main. | ChrF for NLLB-1.3B zero-shot vs. fine-tuned. |
| `fig7/comet/m2m100_418m.pdf` | M2M100-418M 7×7. **Both panels filled** (X↔X imported from `dcs_n2n/`). | Smaller M2M100 backbone (418M): zero-shot vs. fine-tuned COMET over the full eng+African 7×7 grid. Reported for completeness alongside the larger M2M100-1.2B. |
| `fig7/chrf/m2m100_418m.pdf` | ChrF mirror. | Same panel, ChrF. |
| `fig7/comet/nllb_600m.pdf` | NLLB-600M 7×7, both panels filled. | Smaller NLLB backbone (600M): zero-shot vs. fine-tuned COMET over the full 7×7 grid. |
| `fig7/chrf/nllb_600m.pdf` | ChrF mirror. | Same panel, ChrF. |
| `fig8/gemma2_9b_r64.pdf` | Gemma2-9B LoRA $r{=}64$, 7×7 fine-tuned only, COMET \| ChrF side-by-side. | Gemma2-9B fine-tuned with LoRA $r{=}64$: COMET (left) and ChrF (right) over the 7×7 grid. African↔African cells are not populated (no LoRA n×n fine-tunes). |
| `multilingual-nllb.pdf` | Multilingual NLLB-600M, ChrF across 4 training-update checkpoints (40k, 80k, 200k, 400k), Eng→X and X→Eng panels. | Multilingual NLLB-600M ChrF as a function of fairseq training updates. Shown for completeness; the main paper's bilingual–multilingual comparison uses LLaMA-3.1-8B (matched backbone). |
| `n2n_zero_shot_heatmap.pdf` | African↔African zero-shot 6×6 heatmaps (diagonals masked) for NLLB-1.3B and M2M100-1.2B, ChrF and COMET. | Zero-shot direct African→African translation by the larger seq2seq backbones; diagonal cells (src=tgt) omitted. |
| `m2m100_heatmaps.pdf` | M2M100 (418M, 1.2B) zero-shot vs. fine-tuned, ChrF + COMET, 12-pair layout. | Pair-level performance of M2M100 backbones on AfriScience-MT (zero-shot vs. fine-tuned). |
| `nllb_heatmaps.pdf` | NLLB (600M, 1.3B) zero-shot vs. fine-tuned, same shape. | Pair-level performance of NLLB backbones. |
| `gemma_m2m_heatmaps.pdf` | LoRA $r{=}64$: Gemma2-9B vs. LLaMA-3.1-8B, ChrF + COMET, 12-pair layout. | LoRA $r{=}64$ fine-tunes for Gemma2-9B and LLaMA-3.1-8B. |
| `radar_chart_domain_chrf.pdf` | Per-domain ChrF radar (avg over the 6 Eng→X pairs) for four selected systems. | Per-domain ChrF averaged across Eng→xxx pairs for selected systems. |
| `weighted_chrf_score.pdf` | Per-domain ChrF bar chart, domains ordered by training-set size $N_d$. | Per-domain ChrF for selected systems, domains ordered by training-set size. |
| `spider_per_pair/3sys/chrf/<pair>.pdf` (12 files) | ChrF mirror of the 3-system spiders. | ChrF version of the 3-system per-pair domain comparison (cf. `spider_per_pair/3sys/comet/`). |
| `spider_per_pair/4sys/chrf/<pair>.pdf` (12 files) | ChrF mirror, 4-system. | ChrF version of the 4-system per-pair domain comparison. |
| `spider_per_pair_eng_xxx.pdf` | Single-PDF 3×2 grid alternative to the per-pair subdir. | Per-domain ChrF for NLLB-1.3B (fine-tuned) and GPT-4o (5-shot ICL) on Eng→xxx translation, broken down by scientific domain. |

### Tables (appendix / cross-metric)

| File | Suggested LaTeX caption |
|---|---|
| `seq2seq_chrf.tex` (`tab:seq2seq_chrf`) | ChrF version of `tab:seq2seq_comet` for the larger Seq2Seq backbones; reported for completeness. |
| `seq2seq_smaller_comet.tex` / `seq2seq_smaller_chrf.tex` (`tab:seq2seq_smaller_*`) | Smaller Seq2Seq backbones (M2M100-418M, NLLB-600M): zero-shot vs. fine-tuned, full 12-pair grid. Reported for completeness; the main paper discusses the larger M2M100-1.2B and NLLB-1.3B. |
| `llm_sentence_chrf.tex` (`tab:llm_sentence_chrf`) | ChrF version of `tab:llm_sentence_comet`. |
| `lora_chrf.tex` (`tab:lora_chrf`) | ChrF version of `tab:lora_comet`. |
| `doc_chrf.tex` (`tab:doc_chrf`) | ChrF version of `tab:doc_comet`. |
| `balance_chrf.tex` (`tab:balance_chrf`) | ChrF version of `tab:balance_comet` (domain-balance ablation). |
| `new_results_regenerated.tex` | (Internal) concatenation of the eight `new_results`-style tables for diffing against the original `new_results.tex`. Not for the paper. |
