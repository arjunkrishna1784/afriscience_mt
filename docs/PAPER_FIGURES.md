# Paper figures — index for the new draft

All PDFs live under
[`_ACL2026__.../figures/new/`](../_ACL2026__Decolonizing_Science_in_Africa_through_Text_Translation_/figures/new/).
Titles, suptitles, and on-figure captions are intentionally stripped — drop
captions in via the LaTeX `\caption{}` of the surrounding `figure` (or
`table*`) environment.

Re-build any of these from scratch:

```bash
python3 scripts/build_paper_assets.py    # zs_vs_ft_delta, lora_rank_curve
python3 scripts/redraw_paper_plots.py    # everything else
```

## Main paper

| File | What it shows | Suggested LaTeX caption |
|---|---|---|
| `zs_vs_ft_delta.pdf` | Per-language SSA-COMET (avg both directions) for NLLB-1.3B and LLaMA-3.1-8B, before vs after fine-tuning. | Effect of in-domain fine-tuning on AfriScience-MT for the strongest seq2seq backbone (NLLB-1.3B) and the per-pair LoRA on LLaMA-3.1-8B. |
| `fig7_m2m100_1_2b_chrf.pdf` | M2M100-1.2B 7×7, zero-shot \| fine-tuned, ChrF. Right panel currently has only eng row + eng column (no fine-tuned X↔X for the bigger backbone). | Sentence-level ChrF for M2M100-1.2B before and after fine-tuning. Fine-tuned X↔X experiments were not run for the larger backbone (left blank). |
| `fig7_m2m100_1_2b_comet.pdf` | Same as above, SSA-COMET. | Same metric pair, SSA-COMET. |
| `fig7_nllb_1_3b_chrf.pdf` | NLLB-1.3B 7×7, zero-shot \| fine-tuned, ChrF. Right panel sparse (same caveat). | Sentence-level ChrF for NLLB-1.3B before and after fine-tuning. |
| `fig7_nllb_1_3b_comet.pdf` | Same as above, SSA-COMET. | Same panel pair, SSA-COMET. |
| `lora_rank_curve.pdf` | Avg SSA-COMET vs LoRA rank (LLaMA-3.1-8B), separate eng→X and X→eng curves. | LoRA rank vs.\ SSA-COMET on LLaMA-3.1-8B fine-tuned on AfriScience-MT. r=4 and r=64 use the vanilla base; other ranks use Unsloth 4-bit (footnote). |
| `spider_per_pair_eng_xxx.pdf` | Six spider charts (3×2), one per Eng→X target. Two systems: NLLB-1.3B fine-tuned vs.\ GPT-4o 5-shot ICL, 11 domains. Page-23 style. | Domain-level ChrF for NLLB-1.3B (fine-tuned) and GPT-4o (5-shot ICL) on Eng→xxx translation, broken down by scientific domain. |

## Appendix

| File | What it shows | Suggested LaTeX caption |
|---|---|---|
| `fig7_m2m100_418m_chrf.pdf` | M2M100-418M 7×7 zero-shot \| fine-tuned, ChrF. Both panels filled (X↔X data was imported from `dcs_n2n/`). | Smaller M2M100 backbone (418M): zero-shot vs.\ fine-tuned ChrF over the full eng+African 7×7 grid. Reported for completeness. |
| `fig7_m2m100_418m_comet.pdf` | Same, SSA-COMET. | Same backbone, SSA-COMET. |
| `fig7_nllb_600m_chrf.pdf` | NLLB-600M 7×7, zero-shot \| fine-tuned, ChrF. Both panels filled. | Smaller NLLB backbone (600M): zero-shot vs.\ fine-tuned ChrF over the full 7×7 grid. |
| `fig7_nllb_600m_comet.pdf` | Same, SSA-COMET. | Same backbone, SSA-COMET. |
| `fig8_gemma2_9b_r64.pdf` | Gemma2-9B LoRA r=64, 7×7 fine-tuned only, COMET \| ChrF side-by-side. Only eng↔X is filled (no n×n LoRA fine-tunes). | Gemma2-9B fine-tuned with LoRA r=64: COMET and ChrF. X↔X cells are not populated. |
| `multilingual-nllb.pdf` | Multilingual NLLB-600M, ChrF across 4 training-update checkpoints (40k, 80k, 200k, 400k), Eng→X and X→Eng panels. | Multilingual NLLB-600M ChrF as a function of fairseq training updates. Shown for completeness; the main paper's bilingual–multilingual comparison uses LLaMA-3.1-8B (matched backbone). |
| `n2n_zero_shot_heatmap.pdf` | African↔African zero-shot 6×6 heatmaps (diagonals masked) for NLLB-1.3B and M2M100-1.2B, ChrF and SSA-COMET. | Zero-shot direct African–to–African translation by the larger seq2seq backbones; diagonal cells (src=tgt) omitted. |
| `m2m100_heatmaps.pdf` | M2M100 (418M, 1.2B) zero-shot vs.\ fine-tuned, ChrF + SSA-COMET, 12-pair layout. | Pair-level performance of M2M100 backbones on AfriScience-MT (zero-shot vs.\ fine-tuned). |
| `nllb_heatmaps.pdf` | NLLB (600M, 1.3B) zero-shot vs.\ fine-tuned, same shape. | Pair-level performance of NLLB backbones. |
| `gemma_m2m_heatmaps.pdf` | LoRA r=64: Gemma2-9B vs.\ LLaMA-3.1-8B, ChrF + SSA-COMET, 12-pair layout. | LoRA r=64 fine-tunes for Gemma2-9B and LLaMA-3.1-8B. |
| `radar_chart_domain_chrf.pdf` | Per-domain ChrF radar (avg over the 6 Eng→X pairs) for four selected systems. | Per-domain ChrF averaged across Eng→xxx pairs for selected systems. |
| `weighted_chrf_score.pdf` | Per-domain ChrF bar chart, domains ordered by training-set size $N_d$. | Per-domain ChrF for selected systems, domains ordered by training-set size. |

## Tables

All eight (plus the two appendix variants) are in
[`tables/new/`](../tables/new/). Filenames mirror the labels.

| File | Suggested LaTeX label |
|---|---|
| `seq2seq_chrf.tex` / `seq2seq_comet.tex` | `tab:seq2seq_chrf` / `tab:seq2seq_comet` |
| `seq2seq_smaller_chrf.tex` / `seq2seq_smaller_comet.tex` | `tab:seq2seq_smaller_chrf` / `tab:seq2seq_smaller_comet` |
| `llm_sentence_chrf.tex` / `llm_sentence_comet.tex` | `tab:llm_sentence_chrf` / `tab:llm_sentence_comet` |
| `lora_chrf.tex` / `lora_comet.tex` | `tab:lora_chrf` / `tab:lora_comet` |
| `doc_chrf.tex` / `doc_comet.tex` | `tab:doc_chrf` / `tab:doc_comet` |
| `new_results_regenerated.tex` | concatenated mirror of all eight, useful for diffing against the original `new_results.tex` |
