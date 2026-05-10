# Paper findings — one paragraph per table/figure

Drafting reference for the writing pass. Each entry leads with the
headline number, then the qualifier, then any caveat worth surfacing
in the body. Numbers come from the latest tables under `tables/new/`
and the local `summary.csv` (re-aggregate before quoting).

---

## Body tables

### `tab:data_splits_stats` — domain-stratified data splits

The AfriScience-MT corpus is split 80/10/10 at the **document level
within each of the 11 scientific domains**, not at the sentence
level. This guarantees every split contains complete papers from
every domain (with at least one document per (split, domain) cell)
and prevents single-paper sentences from leaking across train/dev/
test. Per-language sentence counts are roughly balanced (the table
shows the exact per-language, per-domain breakdown). The
domain-stratified design is what makes the per-domain evaluation
in `\Cref{fig:spider_per_pair}` meaningful: the test set carries
representative coverage of each domain rather than being dominated
by whichever domain had the most papers translated.

### `tab:gloss_number` — bilingual scientific glossary

The translators co-developed scientific terminology while translating;
the resulting `eng`$\leftrightarrow$`xxx` glossaries range from a few
hundred to a couple of thousand entries depending on language, with
larger glossaries for the languages where existing scientific
terminology was thinnest. The glossary is a side artefact of the
dataset construction, not used at training/inference, but it is a
concrete deliverable for the language communities and partly
motivates the "decolonising science" framing.

### `tab:main_zs_ft_comet` — sentence-level zero-shot vs fine-tuned

NLLB-1.3B fine-tuned on AfriScience-MT is the strongest system in
the table at **67.3 SSA-COMET avg** across the 12 directional pairs,
with the smaller NLLB-600M only 0.6 COMET behind (66.7) -- a
striking efficiency result for a model less than half the size.
Among the open-source LLMs (per-pair LoRA $r{=}64$), Tiny-Aya-G
leads at 64.9 avg, but its xxx$\to$eng numbers carry the
chrF-up/COMET-down degeneration signature flagged in
`\Cref{tab:prompt_engineering_open}`; Gemma2-9B at 57.9 is the
better-behaved second pick. M2M100-1.2B is the only seq2seq baseline
that comes close to NLLB (62.6), confirming NLLB's massively
multilingual pre-training is doing the heavy lifting. **Bold** marks
the per-language column max within each major group; **emph** marks
the column max within each sub-block (seq2seq vs LLM).

### `tab:open_vs_closed_icl_comet` — open vs closed-API LLMs at the locked config

All rows are at the **uniform** `template5_10shot` configuration at
$T{=}0$ (selected from `\Cref{tab:template_shot_ablation_comet}`),
so the comparison is apples-to-apples across providers. Closed APIs
(GPT-4o, Gemini-1.5-Flash) lead the open LLMs at sentence-level ICL,
but the gap is much smaller than it looks if you only read chrF:
several open LLMs (AfriqueLlama-8B, AfriqueQwen-8B,
TranslateGemma-12B) **degenerate at this prompt** -- visible as
COMET~$\gg$~chrF in their rows -- so their numbers here are an
upper bound on perceived adequacy rather than a faithful chrF
score. The "Best open" zero-shot row is the per-cell maximum across
the seven open LLMs in `\Cref{tab:main_zs_ft_comet}`. The
degeneration story is unpacked in `\Cref{tab:prompt_engineering_open}`.

### `tab:mafand_nllb` — cross-domain transfer (AS-MT vs MAFAND-MT)

NLLB-1.3B fine-tuned on **AfriScience-MT (AS-MT)** outperforms the
same model fine-tuned on **MAFAND-MT (MF)** on the four
MAFAND-supported pairs in both directions, by **5--10 SSA-COMET on
average**. Adding MAFAND on top of AS-MT (`+MF`) does not help and
in some cells slightly hurts -- the in-domain AS-MT signal is
strong enough that mixing in news-domain MAFAND data dilutes
rather than augments it. ChrF tells the same story slightly more
strongly. This is the cleanest "AfriScience-MT is a useful
fine-tuning corpus" evidence in the paper.

### `tab:doc_comet` — document-level zero-shot vs 1-shot ICL

At document level the **closed APIs lose their lead**: TranslateGemma-12B
is the per-cell max in 11 of 12 cells in the 1-shot block, and the
two best open LLMs (TranslateGemma-12B, AfriqueLlama-8B) clearly
beat GPT-4o and Gemini-1.5 in both directions and both shot
counts. This inverts the sentence-level ranking of
`\Cref{tab:open_vs_closed_icl_comet}` and is the doc-level
headline. Open LLMs only have a single `doc_0shot` / `doc_1shot`
configuration each (no template variants), so the row count is
modest; the closed-API template/shot grid is in
`\Cref{tab:doc_template_shot_ablation_comet}` (appendix).

---

## Appendix tables

### `tab:closed_ablations_comet` — closed-API ablations

Three blocks at $T{=}0$, eng$\to$xxx only. **Reported** duplicates
the closed-API headline rows from
`\Cref{tab:open_vs_closed_icl_comet}` for context. **Retrieval
ablation** swaps templated 10-shot for semantic-similarity 10-shot
exemplars at the same template -- the lift is small ($\le 1$ COMET
on average) for both APIs, confirming that templating, not
retrieval, is doing the work. **Temperature ablation** replays the
headline templated config at $T{=}0.6$; this is essentially
identical to $T{=}0$ for both APIs (within 0.5 COMET), so we can
ignore temperature as a tuning knob throughout. The takeaway is
that, for closed APIs, the prompt template choice dominates both
retrieval strategy and decoding temperature.

### `tab:prompt_engineering_open` — vanilla vs templated ICL on open LLMs

This table disentangles the **prompt** effect (vanilla vs templated)
from the **shot-count** effect on the seven open LLMs, with COMET
and chrF side-by-side. For the two clean models (Llama3-8B,
Gemma2-9B-IT) the prompt jump V$_5\to$T$_{10}$ is roughly +5--10
COMET while the shot-count effect T$_5\to$T$_{10}$ is at most
1.5 COMET / 2.4 chrF -- so almost all of the headline ICL lift
comes from the prompt template, not from adding more exemplars.
For AfriqueLlama-8B, AfriqueQwen-8B, and TranslateGemma-12B, the
templated prompt produces a **large positive $\Delta$ COMET paired
with a large negative $\Delta$ chrF** ($-21.7$ / $-5.9$ / $-23.9$
chrF respectively): a clear degeneration signature where the model
emits fluent but off-target text that COMET tolerates but chrF
penalises. The vanilla baseline is therefore the more honest
reading for those three models.

### `tab:template_shot_ablation_comet` — sentence-level template/shot ablation

Four reference models with bidirectional templated coverage at
$T{=}0$: Gemini-1.5, GPT-4o, Llama3-8B, Gemma2-9B-IT. Each row is
a (template, shot-count) configuration; the underlined
`5`$_{s=10}$ row is the **locked universal config** used in
`\Cref{tab:open_vs_closed_icl_comet}` and downstream. The
\textbf{avg} row at the bottom of each block summarises across
configurations. Two findings drove the lock: (i)
template5\_10shot is within 0.5 COMET of every model's per-row
maximum (so we are not sacrificing per-model headroom by
standardising), and (ii) the gain from $10\to20$ shots is at most
0.2 COMET, well below the 1-COMET threshold we set for adopting
a higher shot count -- so 20-shot would buy nothing and 50-shot
(banned project-wide for cost reasons) is decisively unjustified.

### `tab:balance_domain_eng_xxx_dual` — domain-balance ablation (NLLB-1.3B)

Three sampling regimes for NLLB-1.3B fine-tuning, holding total
training volume constant at 5{,}792 sentences: **$\tau{=}1$**
(natural domain proportions), **$\tau{=}2$** (sqrt smoothing), and
**$\tau{=}\infty$** (uniform across domains). The natural
distribution wins: $\tau{=}\infty$ degrades performance by up to
8.3 chrF on English$\to$Hausa relative to $\tau{=}1$, with
consistent (smaller) drops on most other (domain, language) cells
in red, and only scattered improvements (blue) on the smallest
domains. **Forced balance hurts more than it helps**: the model
benefits more from exposure to naturally distributed data --
where the largest domains (Health: 2{,}605 sentences) supply
disproportionate signal -- than from artificial uniformity that
starves the data-rich domains. ChrF and COMET tell the same
story.

### `tab:lora_comet` — LoRA $r{=}4$ vs $r{=}64$ across 6 open LLMs

Comparing two LoRA ranks isolates the effect of adapter capacity.
At $r{=}4$ AfriqueLlama-8B is the strongest model on
`amh\to eng` (67.6 COMET, the table-wide max), reflecting that
its African pre-training already provides a strong starting point
that low-rank adapters can quickly tune. At $r{=}64$ Tiny-Aya-G
takes over and dominates 9 of 12 cells (it is also the per-block
column max in the rest), showing that the bigger adapter unlocks
its capacity for the harder languages. AfriqueQwen-8B is the
clear loser at both ranks -- a low-resource model whose adapter
budget cannot compensate for the weaker base. The chrF mirror
(`\Cref{tab:lora_chrf}` if included) gives the same ranking.

### `tab:multilingual_vs_bilingual` — bilingual vs multilingual fine-tuning

Three model families, each comparing same-model bilingual fine-tunes
against same-model multilingual fine-tunes at matching ranks /
checkpoints. **NLLB-600M / 1.3B**: bilingual is dramatically
stronger (avg 67.2 COMET vs 39.8 at the latest MMT checkpoint, a
27-point gap) -- with the caveat that the multilingual NLLB-600M
checkpoints are short, early-stage runs, so the gap conflates
training-steps with the bilingual-vs-multilingual axis. **Gemma-2-9B
LoRA**: bilingual leads multilingual by **$\sim$3.7 COMET / $\sim$5
chrF** at the best rank, a clean apples-to-apples win. **Llama-3.1-8B
LoRA**: bilingual and multilingual end up **essentially tied at
$\sim$57 COMET** across matching ranks -- Llama is the outlier
where the headline "bilingual wins" claim does not hold. The Llama
$r{=}64$ bilingual row also shows a chrF-up/COMET-down degeneration
relative to the other Llama bilingual ranks, so it should be cited
with care.

### `tab:doc_template_shot_ablation_comet` — closed-API doc-level template/shot grid

GPT-4o and Gemini-1.5-Flash are the only models with the full
3-template $\times$ 3-shot grid at document level (open LLMs only
have single `doc_0shot` / `doc_1shot` configs). Across the nine
configurations both APIs are **remarkably stable**: per-config avg
varies by at most 1.0 SSA-COMET for either model, and no template
or shot count is meaningfully better. This is a notable contrast
with the sentence-level result (`\Cref{tab:template_shot_ablation_comet}`)
where template choice contributes a 1--3 COMET lift -- at document
level the prompt format matters less, presumably because the
input itself supplies enough context that the templated framing
adds little.

---

## Body figures

### `fig:dcs_langs_speakers` — geographic distribution of the six languages

Map of the six AfriScience-MT languages with speaker-population
shading: West Africa (Hausa, \yoruba), East Africa (Amharic,
Luganda), Southern Africa (Northern Sotho, isiZulu). Used solely to
illustrate the regional and family coverage of the corpus
(Niger-Congo plus Afro-Asiatic Amharic), motivating the choice of
languages as a representative slice rather than a convenience
sample. Speaker counts and Ethnologue references are in the
caption.

### `fig:data-creation-framework` — translation-pipeline diagram

The original plan to translate full scientific papers proved
infeasible (translators reported up to 10$\times$ longer
processing time per sentence on technical content), so the
pipeline pivots to a **two-stage** process: (1) expert
summarisation of each source paper to a tractable abstract-style
extract, then (2) professional translation of the extract into the
target language plus collaborative terminology development for any
gaps. The figure traces this flow and the QA loop. Worth referencing
when discussing dataset construction trade-offs in §3.

---

## Appendix figures

### `fig:multilingual_nllb` — multilingual NLLB-600M training-update sweep

Four sub-panels: chrF and SSA-COMET, eng$\to$xxx and xxx$\to$eng.
Each language is a separate bar group with a trend line across the
40k / 80k / 200k / 400k checkpoint sweep. **COMET and chrF disagree
substantially on which languages are hardest**: chrF identifies
Yoruba and Amharic as the most difficult (low absolute scores)
while COMET ranks Hausa and isiZulu as relatively harder -- a
metric-divergence signal that motivates reporting both throughout
the paper rather than collapsing to one. The cross-reference
`\Cref{sec:multilingual_nllb_metric_divergence}` carries the
per-language discussion. (The caption's TODO marker should be
removed once that discussion is finalised.)

### `fig:gpt_gemini_eng_xxx` and `fig:gpt_gemini_xxx_eng` — closed-API template/shot sweep

Two figures, one per direction, each with chrF and COMET sub-panels.
For each (model, language pair) cell we show how chrF varies across
the five prompt templates and the {0, 5, 10, 20}-shot ladder. Two
takeaways: (i) **template choice matters more than shot count** --
the spread across templates at fixed shots is typically $\geq$ the
spread across shots at fixed template; (ii) **template5
consistently leads or ties** for both APIs across most pairs,
which is the empirical basis for the
`\Cref{tab:template_shot_ablation_comet}` selection of
template5\_10shot as the locked body config. Per-template median
spreads ($\sim$1 chrF), p90 ($\sim$4 chrF), and max (6.3 chrF on
GPT-4o eng$\to$amh) are quoted in §X.

### `fig:spider_per_pair` — per-domain spider charts (12 pairs $\times$ 3 systems)

Stacked sub-figures: SSA-COMET on top, ChrF on bottom; each panel
shows the 11 scientific domains as spider axes for one
(source$\to$target) pair, with three systems overlaid:
**NLLB-1.3B (fine-tuned)**, **Gemma-2-9B-IT (LoRA $r{=}64$)**, and
**GPT-4o (10-shot ICL, template5)**. The three systems were chosen
to give one strong representative per family (best seq2seq, best
open LLM, best closed API at the locked config). The recurring
pattern is that NLLB-1.3B traces a **larger and more uniform
polygon**; the LLMs collapse on Indigenous-Knowledge, Statistics
and Engineering (the small-N domains where pre-training contributes
less), and the gap widens for low-resource targets like Yoruba and
Amharic. Same systems appear in
`\Cref{fig:weighted_chrf_score}` so the per-domain story is
consistent across the two figures.

### `fig:n2n_zero_shot` — African$\leftrightarrow$African zero-shot heatmap

Each cell is one source$\to$target pair (rows = source, columns =
target, diagonals omitted). Top row: NLLB-1.3B; bottom row:
M2M100-1.2B. Left column: chrF; right column: COMET. The headline
result is that **NLLB beats M2M100 on every African$\to$African
direction** (no fine-tuning), often by 10+ chrF / 5+ COMET, even
on pairs that share neither alphabet nor family (e.g.
amh$\to$zul). This is the strongest piece of evidence that NLLB's
massively multilingual pre-training generalises to language
combinations it never explicitly saw, and motivates using NLLB
rather than M2M100 as the seq2seq baseline elsewhere in the paper.

### `fig:doc_error_dist` — document-level error distributions

Two pies (eng$\to$xxx and xxx$\to$eng) summarising the
LLM-judged document-level error categories (accuracy, lexical,
grammatical, fluency) across the systems in
`\Cref{tab:doc_evaluation_eng_xxx}` / `_xxx_eng`. The dominant
error type in both directions is **lexical** -- specialised
scientific terminology that systems either translate
inconsistently or substitute with generic words -- which echoes
why the bilingual glossary work in `\Cref{tab:gloss_number}` is
load-bearing for the project. Accuracy errors are notably worse
into African languages than into English; grammatical errors are
relatively rare in both directions for the LLM-judged systems.

### Candidate: `fig:weighted_chrf_score` — per-domain bars by domain size

Two stacked panels (SSA-COMET top, chrF bottom) with three systems
identical to `\Cref{fig:spider_per_pair}`: NLLB-1.3B FT,
Gemma-2-9B-IT LoRA $r{=}64$, GPT-4o template5\_10shot. Domains
are ordered by training-set size, ascending (Chemistry $N=46$ on
the left through Health $N=2{,}605$ on the right). The size-axis
ordering reveals two facts the spider obscures: (i) **NLLB is
domain-stable** -- its bars are roughly flat across the
size-ordered axis on both metrics; (ii) **GPT-4o has a slightly
positive trend with domain size** on chrF (smaller domains hurt
it more than larger ones), suggesting it leans more on
in-distribution lexical familiarity that the smaller domains do
not provide. Same caption as the spider for the system-list, plus
the ordering rationale.

### Candidate: domain-balance heatmap + delta-by-size (`balance_*.pdf`)

If we add this figure pair, place it under
`\Cref{tab:balance_domain_eng_xxx_dual}`. The heatmap visualises
per-domain chrF for the three sampling regimes ($\tau{=}1$,
$\tau{=}2$, $\tau{=}\infty$) at a glance; the delta-by-size bar
chart sorts the 11 domains by training-set size and plots the
$\Delta$ chrF $(\tau{=}\infty - \tau{=}1)$ for each, making the
"forced uniformity hurts the data-rich domains and helps only the
smallest ones" pattern visible at one scan. Together they reinforce
the table's numerical story but in a way that scales across the
appendix at glance-time, useful if a reader is skimming the
domain-balance section.
