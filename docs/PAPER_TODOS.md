# Paper writing — pending decisions and bookmarks

Items captured during drafting that need either user input or specific data
before they can be finalized. Each entry has the proposed wording verbatim
so it can be edited later without re-deriving the argument.

---

## 2026-05-06 — Prompt-template asymmetry between closed APIs and open LLMs

**Status:** bookmarked. Waiting for the user to drop the per-template results
in this file so the wording can be finalized with concrete numbers.

**Why we need this bookmark.** GPT-4o and Gemini-1.5-Flash were each evaluated
across multiple prompt templates (the appendix already has
`tab:models_performances_templates_eng_xxx` / `_xxx_eng` and
`figures/gpt-gemini-template-shots.pdf`). The open-source LLMs were tested
on a single canonical instruction-tuned chat template — i.e. no template
sweep. A reviewer will notice the asymmetry; we should disclose it
ourselves rather than be asked.

### Proposed text — main paper (one sentence + footnote)

> For closed APIs (GPT-4o, Gemini-1.5-Flash) we evaluated several prompt
> templates and report the best-performing variant per language pair;\footnote{Per-template
> scores and the template inventory are in
> Tables~\ref{tab:models_performances_templates_eng_xxx} and
> ~\ref{tab:models_performances_templates_xxx_eng} and
> Figure~\ref{fig:gpt_gemini_comparison}.} open-source LLMs use a single
> canonical instruction-tuned chat template throughout.

### Proposed text — Limitations section

> Open-source LLMs were not subjected to a comparable prompt-template sweep;
> their reported numbers may therefore be a lower bound on what is achievable
> with prompt engineering.

### Concrete spread (computed 2026-05-06, after closed-LLM template import)

Predictions for templates 1–5 across {0, 5, 10, 20, 50}-shot were converted
into the canonical schema and scored on chrF (`scripts/convert_closed_llm_templates.py`).
The summary below is over `(model, pair, n_shots, temperature)` cells (i.e. one
"cell" is a fixed setup where only the template varies):

| model | n cells | median Δ ChrF | p90 Δ ChrF | max Δ ChrF |
|---|---|---|---|---|
| GPT-4o | 60 | 1.1 | 3.4 | **6.3** |
| Gemini-1.5-Flash | 90 | 0.8 | 4.1 | **5.2** |

**Largest gap observed:** GPT-4o, eng→amh, 0-shot — template 5 = 27.2 chrF
vs.\ template 2 = 21.0 chrF (Δ 6.3).

**Best-template frequency (which template wins each cell):**
- GPT-4o: template 3 wins **53%**, template 5 wins **33%**.
- Gemini-1.5-Flash: template 5 wins **64%**, template 4/3 split the rest.

These suggest **the current draft's "up to 10 ChrF" claim is overstated** —
true max is ≈ 6 ChrF and median is ≈ 1 ChrF. The footnote should report
the median or p90, not the max.

### Final wording (updated with concrete numbers)

**Methodology (one sentence + footnote):**

> For closed APIs (GPT-4o, Gemini-1.5-Flash) we evaluated five prompt
> templates across {0, 5, 10, 20, 50}-shot ICL and report the
> best-performing variant per language pair;\footnote{Per-template
> ChrF varies by a median of $\sim$1 ChrF (p90 $\sim$4 ChrF; max
> 6.3 ChrF on \texttt{eng→amh}) across our 5 templates --- a smaller
> sensitivity than the 10-ChrF figure quoted in earlier drafts.
> Template 3 wins 53\% of GPT-4o cells; template 5 wins 64\% of
> Gemini cells. Full template inventory and per-cell scores are in
> Tables~\ref{tab:models_performances_templates_eng_xxx} and
> ~\ref{tab:models_performances_templates_xxx_eng}.}
> open-source LLMs use a single canonical instruction-tuned chat
> template throughout.

**Limitations:**

> Open-source LLMs were not subjected to a comparable prompt-template sweep;
> their reported numbers may therefore be a lower bound on what is achievable
> with prompt engineering. For the closed APIs, template choice moves ChrF
> by a median of $\sim$1 point but as much as 6 points on hard pairs
> (e.g.\ \texttt{eng→amh}, \texttt{eng→yor}).

---

---

## 2026-05-07 — Balance ablation: surprising negative result

**Status:** finding nailed down; needs ~2 sentences in §4 of results.

NLLB-1.3B fine-tuned with τ ∈ {1, 2, ∞} sampling. **Every domain loses
ChrF under balancing**, no exceptions, including the smallest (chem,
N=46, drops 1.5 ChrF). Numbers and assets:

- `tables/new/balance_chrf.tex` — 3 × 12-pair, ChrF
- `tables/new/balance_comet.tex` — same shape, SSA-COMET
- `figures/new/balance_heatmap.pdf` — 3 × 11 heatmap (τ × domain)
- `figures/new/balance_delta_by_size.pdf` — Δ ChrF per domain (all negative)

τ=1 wins **11 of 12** SSA-COMET cells. τ=∞ wins 0. τ=2 is uniformly
between the two.

### Proposed text (one paragraph + figure pair)

> Across the three temperature settings, every per-domain ChrF score is
> highest at $\tau{=}1$ (the natural distribution) and falls monotonically
> as the sampling distribution flattens. The smallest domain in
> AfriScience-MT (chemistry, $N{=}46$) drops $1.5$ ChrF moving from
> $\tau{=}1$ to $\tau{=}\infty$; \emph{biology} drops $1.9$. The
> hypothesis that balanced sampling buys cross-domain transfer for
> low-data domains is not supported by our data --- volume of in-domain
> examples (across any domain) dominates. This complements the truncation
> evidence in \S~\ref{sec:cross-domain-transfer}: re-weighting and hard
> truncation both lose to the natural-distribution baseline.

### Connect to existing claim

The introduction (`acl_latex.tex` contribution 4) already mentions that
"artificially balancing dataset sizes to match domain proportions
degrades overall model quality by up to 8.3 ChrF points." That earlier
result was based on **truncation**. The new τ-sampling experiment shows
a much smaller degradation (max 1.9 ChrF). The right framing in the paper
is *not* to drop the 8.3-ChrF figure but to **decompose** it.

### Why truncation showed up to 8.3 ChrF degradation while τ-sampling shows only ≈1–2 ChrF

Truncation cuts the training set from **5,792 → ≈506 sentences (a 91%
reduction)** to match the smallest domain's count, then trains. The
resulting performance drop conflates two distinct effects:

1. **Data starvation** — NLLB-1.3B trained on 506 in-domain sentences
   under-fits the task. This is the dominant component of the 8.3 ChrF
   drop.
2. **Domain re-weighting** — independently of volume, would balanced
   per-domain proportions help? Truncation can't answer this because it
   changes both variables at once.

Temperature sampling (τ=2, τ=∞) holds total volume **constant** at
5,792 sentences and changes only the per-example sampling weight. Under
that clean setting the degradation is at most 1.9 ChrF and present on
every domain — including chemistry (N=46), the very domain the
hypothesis predicted should *gain*.

**Decomposition:** of the 8.3 ChrF the truncation experiment loses,
≈1–2 ChrF is attributable to balance per se, and ≈6–7 ChrF is
attributable to data starvation from the 91% volume cut.

### Proposed text for the paper

> Two complementary experiments. **Truncation** (Table~\ref{tab:trunc})
> reduces the corpus to $\approx$506 sentences to match the smallest
> domain, then trains; the result drops up to 8.3 ChrF on
> \texttt{eng→hau}. This conflates volume and balance. **Temperature
> sampling}~(Table~\ref{tab:balance_chrf}, Fig.~\ref{fig:balance_heatmap})
> holds the corpus at 5\,792 sentences but reweights examples toward
> uniform per-domain proportions. Under that cleaner test, balancing
> degrades performance on \emph{every} domain — even chemistry (N=46,
> $-1.5$ ChrF) — by at most 1.9 ChrF. Most of the truncation degradation
> is therefore attributable to **data starvation**, not domain balance:
> NLLB-1.3B simply does not have enough sentences to fine-tune on at the
> 506-sentence scale. The narrower 1.9-ChrF cost is the true price of
> re-weighting; under data-constrained fine-tuning of a 1.3 B-parameter
> model, **volume of any-domain examples dominates per-domain balance.**

### Assets ready to drop in

- [`tables/new/balance_chrf.tex`](../tables/new/balance_chrf.tex) — 3 × 12-pair, ChrF (`tab:balance_chrf`)
- [`tables/new/balance_comet.tex`](../tables/new/balance_comet.tex) — same shape, SSA-COMET (`tab:balance_comet`)
- [`tables/new/balance_domain_x_lang_eng_xxx.tex`](../tables/new/balance_domain_x_lang_eng_xxx.tex) — Table-7-style 11-domain × 6-target with red/blue Δ deltas (`tab:balance_domain_eng_xxx`)
- [`figures/new/balance_heatmap.pdf`](../_ACL2026__Decolonizing_Science_in_Africa_through_Text_Translation_/figures/new/balance_heatmap.pdf) — 3 × 11 (τ × domain) heatmap
- [`figures/new/balance_delta_by_size.pdf`](../_ACL2026__Decolonizing_Science_in_Africa_through_Text_Translation_/figures/new/balance_delta_by_size.pdf) — Δ ChrF per domain, ordered by N_d (all bars red)

---

---

## 2026-05-07 — Prompt-template structure audit (closed APIs only)

**Status:** audit complete; concrete prose ready to drop into the
methodology paragraph and Figure 4 caption. Numbers already in the
2026-05-06 bookmark above.

### What we have (5 sentence-level templates from `prompt-templates.tex`)

| # | Description | `{examples}` slot? | `{glossary}` slot? | Few-shot capable? |
|---|---|---|---|---|
| T1 | Basic translation | No | No | **0-shot only** |
| T2 | Scientific translation | No | No | **0-shot only** |
| T3 | Translation with examples | **Yes** | No | Yes |
| T4 | Translation with glossary | No | Yes | **0-shot only** |
| T5 | Translation with examples + glossary | **Yes** | Yes | Yes |

Plus a **"Template 5 (sem.\ prompt)"** variant: identical to T5 but with
**semantically-retrieved** few-shot examples instead of randomly-sampled
ones, evaluated only on Eng→xxx (no semantic ICL run for X→Eng).

Doc-level templates are a **separate set** (D1 basic, D2 domain-specific,
D3 simplified) that share numbers with the sentence-level set — a
paper-internal collision worth disambiguating.

### Three findings the paper text should address

1. **Templates 1, 2, 4 are 0-shot only.** They have no `{examples}`
   placeholder, so few-shot examples have nowhere to go. Figure 4 shows
   them as isolated points at $x=0$ while T3 / T5 / T5-sem show full
   curves across shots. Without explanation, a reader sees "three flat
   lines" and wonders if the data is missing.
2. **Template 5 (sem.\ prompt)** is template 5 with semantic-retrieval,
   not a 6th template — the figure legend phrasing already implies
   this but the methodology section doesn't.
3. **Doc-level Template 3 ≠ sentence-level Template 3.** They share a
   number across `prompt-templates.tex` figures. As long as no figure
   in the paper sits adjacent to both (which the current ordering
   avoids), the numbering can stand; otherwise rename one set as
   D1/D2/D3.

### Proposed addition — methodology section, one sentence

> Sentence-level closed-API templates differ in what context they admit:
> templates 1, 2, and 4 contain no \texttt{\{examples\}} placeholder and
> are reported at 0-shot only; templates 3 and 5 vary across shot counts.
> "Template 5 (sem.\ prompt)" denotes template 5 with
> semantically-retrieved few-shot examples in place of randomly-sampled
> ones, evaluated only on \texttt{eng}$\to$\texttt{xxx}.

### Proposed Figure 4 caption (replaces the current one)

> ChrF (or COMET, depending on metric figure) for GPT-4o and
> Gemini-1.5-Flash on \texttt{eng}$\to$\texttt{xxx}, across five prompt
> templates and shot counts $\in\{0, 5, 10, 20, 50\}$. Templates 1, 2,
> and 4 contain no \texttt{\{examples\}} slot and are reported at
> 0-shot only (single markers); templates 3, 5, and 5 (sem.\ prompt)
> vary across shot counts (full curves). 5 (sem.\ prompt) uses
> semantically-retrieved few-shot examples; templates 3 and 5 use
> randomly-sampled ones.

### Cross-reference

The "up to 10 ChrF" overstatement claim is fixed in the bookmark above
(2026-05-06): true max is 6.3 ChrF, median ~1 ChrF, p90 ~4 ChrF. Apply
the methodology footnote there in the same edit pass.

---

---

## 2026-05-07 — Multilingual NLLB-600M: per-language metric disagreement

**Status:** appendix paragraph drafted; revisit when writing §6 / appendix.
The figure caption stays tight (just a pointer); this paragraph carries the
per-language analysis so readers don't have to absorb five language names
and ten numbers from a caption.

### Why we need it

At the 400k multilingual NLLB-600M checkpoint, COMET and ChrF rank the
six languages **differently** — not just by a small margin. Yorùbá is
the most striking case (worst under ChrF on Eng→Yor, third under COMET;
best under COMET on Yor→Eng, fifth under ChrF). Without explanation,
readers will see different "winners" depending on which metric panel
they look at.

### Tight figure caption (caption only points to the paragraph)

> Multilingual NLLB-600M performance as a function of training updates
> (40k, 80k, 200k, 400k); each language is plotted as its own bar group
> with an in-group trend line. COMET and ChrF disagree substantially on
> which languages are hardest --- see
> §\ref{sec:multilingual_nllb_metric_divergence} for the per-language
> breakdown.

### Appendix paragraph (drop-in)

\paragraph{Per-language metric disagreement.}
\label{sec:multilingual_nllb_metric_divergence}

At the 400k checkpoint, COMET and ChrF rank the six languages differently in both directions. Yor\`ub\'a is the most striking case: it scores worst of the six on \texttt{eng}$\to$ ChrF (13.1) but third on COMET (44.6), and on the reverse direction it is the best language under COMET (\texttt{yor}$\to$\texttt{eng}, 49.0) yet only fifth under ChrF (24.8). Hausa moves in the opposite direction, second on \texttt{eng}$\to$ ChrF (42.4) but fifth on COMET (35.6). Luganda is first under \texttt{eng}$\to$ COMET (46.2) but third on ChrF (33.9).

A plausible explanation is that ChrF's character-level matching penalises \yoruba's diacritics and tonal orthography heavily, while SSA-COMET, trained specifically on sub-Saharan African languages, treats those orthographic differences semantically. The same effect likely inflates Hausa's ChrF (Latin-script with consistent ASCII-friendly forms) above its COMET rank.

Direction also affects the training-curve shape: \texttt{eng}$\to$ \texttt{xxx} keeps improving with more training updates (avg. $+9.8$ COMET, $+12.7$ ChrF across 40k$\to$400k), while \texttt{xxx}$\to$\texttt{eng} plateaus early (avg. $+2.7$ on both metrics). This is consistent with the model's much stronger English representation as the target.

### Why appendix and not caption

The dense five-language list with ten numbers crowds a caption; readers
who only skim figures still get the headline finding (metrics disagree)
from the tight caption above. Per-language detail belongs where readers
expect detailed analysis. The diacritics / SSA-trained interpretation
is also the kind of claim a reviewer will want to engage with — prose
in an appendix subsection invites that, captions don't.

### Decision pending

Either:
- (a) **Add the appendix paragraph** plus the tight caption above
  (preferred), OR
- (b) **Use the one-line caption fallback** without per-language detail
  (drop the paragraph entirely).

If space-constrained, (b) is acceptable; if not, (a) is the cleaner
report.

---

## 2026-05-07 — Tiny-Aya-G xxx→eng degeneration discovered via spot-check

**Status:** bookmarked. Pending: re-run inference with
`scripts/regenerate_tiny_aya_global_predictions.py` (vLLM) and update Table 1.
The finding itself is worth a one-sentence note in the paper recommending
that practitioners eyeball generations rather than trust scalar metrics
blindly.

**What we found.** While verifying Table 1, the user asked for a random spot
check of Tiny-Aya-G (`CohereLabs/tiny-aya-global` LoRA r=64). The fine-tuned
xxx→eng predictions are *sentence-doubled* — each output is roughly
"`<sentence>.<sentence>`". ChrF/BLEU survive duplication (n-grams just
appear twice), but SSA-COMET catches the broken fluency:

| pair | COMET | exact-half-dup |
|---|---:|---:|
| lug-eng | 21.9 | 63.1% |
| yor-eng | 20.5 | 67.2% |
| zul-eng | 33.1 | 60.1% |
| nso-eng | 43.3 | 29.2% |
| amh-eng | 49.0 | 19.0% |
| hau-eng | 55.4 | 14.1% |
| eng→xxx | 53–59 | 0–8% |

The bug is direction-specific (eng→xxx is essentially clean) and severity-
graded by language. Likely cause: at-inference EOS / end-of-turn handling
on the chat-template-trained adapter. Fixed by regenerating with explicit
chat-template stop tokens + `repetition_penalty=1.15` +
`frequency_penalty=0.3`.

### Proposed text — Discussion / Limitations (one sentence)

> A spot-check of fine-tuned outputs revealed a systematic
> sentence-doubling failure on one of our LoRA adapters that lowered
> SSA-COMET by 20–40 points while leaving ChrF largely unaffected (n-gram
> overlap is preserved under repetition). We therefore caution
> practitioners that scalar metrics — especially surface-form metrics like
> ChrF/BLEU — should be paired with manual inspection of a sample of
> generations: had we relied on ChrF alone we would have shipped degraded
> predictions for six of the twelve pairs.

### What this implies for tooling

- Inference scripts should default to `repetition_penalty>1` and explicit
  stop tokens for chat-template models.
- Result-aggregation pipelines should flag rows where COMET ≪ ChrF
  (large divergence is a degeneration signal worth eyeballing).

### What's still missing

1. Run `python scripts/regenerate_tiny_aya_global_predictions.py` on a GPU
   box and verify duplication rate drops to ≤2%.
2. Re-run `python scripts/build_paper_tables.py` and confirm
   Table 1's Tiny-Aya-G row in `→eng` recovers to the 50–60 range.
3. Decide whether the Limitations sentence above goes in the main paper
   or as an appendix-level methodology note.

---

## How to add a new bookmark

1. Add a dated `## YYYY-MM-DD — short title` heading.
2. Open with **Status:** + a one-line "what we're waiting on".
3. Paste the proposed wording verbatim under a `### Proposed text` header.
4. Note **What's still missing** so future-you knows the unblocker.
