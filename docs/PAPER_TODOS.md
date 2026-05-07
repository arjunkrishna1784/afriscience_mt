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

## How to add a new bookmark

1. Add a dated `## YYYY-MM-DD — short title` heading.
2. Open with **Status:** + a one-line "what we're waiting on".
3. Paste the proposed wording verbatim under a `### Proposed text` header.
4. Note **What's still missing** so future-you knows the unblocker.
