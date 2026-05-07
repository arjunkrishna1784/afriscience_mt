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

## How to add a new bookmark

1. Add a dated `## YYYY-MM-DD — short title` heading.
2. Open with **Status:** + a one-line "what we're waiting on".
3. Paste the proposed wording verbatim under a `### Proposed text` header.
4. Note **What's still missing** so future-you knows the unblocker.
