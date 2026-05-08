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

**Status:** RESOLVED in tables; prose locked in for both Methodology
appendix (full paragraph) and Discussion (one sentence). Run that fixed
the issue: vLLM regeneration with unique `lora_int_id` per pair,
`repetition_penalty=1.08`, explicit chat-template stop tokens. Tiny-Aya-G
COMET row recovered to the 60+ range across all 12 pairs (avg 64.9, best
LLM in Table~\ref{tab:main_zs_ft_comet}).

**What we found.** While verifying Table 1, a random spot-check of
Tiny-Aya-G (`CohereLabs/tiny-aya-global` LoRA r=64) revealed sentence-
doubled outputs on the xxx→eng direction — each hyp roughly
"`<sentence>.<sentence>`". ChrF/BLEU survive duplication (n-grams appear
twice), but SSA-COMET caught the broken fluency:

| pair | COMET (broken) | exact-half-dup |
|---|---:|---:|
| lug-eng | 21.9 | 63.1% |
| yor-eng | 20.5 | 67.2% |
| zul-eng | 33.1 | 60.1% |
| nso-eng | 43.3 | 29.2% |
| amh-eng | 49.0 | 19.0% |
| hau-eng | 55.4 | 14.1% |
| eng→xxx | 53–59 | 0–8% |

The bug was direction-specific and severity-graded by language. The fix
was inference-side only (no retraining): vLLM with
`repetition_penalty=1.08`, `frequency_penalty=0.0`, explicit chat-template
end-of-turn stop tokens, and a unique `lora_int_id` per adapter (an
earlier `--all` run with `lora_int_id=1` reused the slot and produced
wrong-target-language outputs).

### Final prose — Methodology appendix (Decoding controls and manual inspection)

> All fine-tuned LLM inference runs greedy decoding with the model's
> chat-template end-of-turn tokens as explicit stop conditions and a
> mild repetition penalty (1.08). We adopted these controls after a
> routine spot-check caught one adapter (Tiny-Aya-Global) emitting
> sentence-doubled outputs (\texttt{<sent>.<sent>}) on the six
> $\rightarrow$English pairs. This halved its COMET score (20–33 instead
> of 60+) while leaving ChrF largely intact, because n-gram overlap
> survives duplication. The lesson is broader than one bug: we recommend
> pairing scalar metrics with a quick manual look at a sample of
> generations, and treating large inter-metric (COMET and ChrF here)
> gaps as a cheap degeneration signal in their own right.

### Final prose — Discussion (one sentence)

> Pairing scalar metrics with a quick manual look at generations matters:
> a late spot-check caught one of our LoRA adapters emitting duplicated
> sentences in a way that halved COMET while leaving ChrF intact, and
> the COMET--ChrF gap itself was the cheapest signal that anything was
> wrong.

### What this implies for tooling (kept for future reference)

- Inference scripts should default to `repetition_penalty>1` and explicit
  stop tokens for chat-template models.
- vLLM with `max_loras=1` requires a unique `lora_int_id` per adapter;
  reusing the same id silently serves stale weights.
- Result-aggregation pipelines should flag rows where COMET ≪ ChrF
  (large divergence is a degeneration signal worth eyeballing).

---

## 2026-05-08 — Closed-API ablations: temperature is flat, semantic-shot count is the real lever

**Status:** TABLE READY. Discussion sentence drafted; appendix section
needs writing.

**What we have.** [tables/new/closed_ablations_comet.tex](../tables/new/closed_ablations_comet.tex)
(and chrf mirror) is a single appendix table with three blocks for the
six \texttt{eng}$\to$\texttt{xxx} pairs (closed-API ablations were not
run on \texttt{xxx}$\to$\texttt{eng}):

  1. **Reported (T=0)** — the headline rows reproduced from Table 2.
  2. **Temperature ablation (T=0.6)** — same configurations re-run at
     higher temperature. Gemini moves $\le$0.2 COMET on every cell;
     GPT-4o zero-shot is identical. The GPT-4o ICL row at T=0.6 used
     random-retrieval whereas its T=0 headline uses semantic-similarity,
     so that one row mixes the two axes (footnoted with $^\dagger$).
  3. **Retrieval ablation (semantic similarity, T=0)** — same model and
     temperature as Block 1, only the ICL exemplar count varies through
     {5, 10, 20, 50}.

**Headline finding.** Temperature does almost nothing
($\le$0.2 COMET delta on Gemini, 0 on GPT-4o ZS). Going from 5- to
10-shot semantic-similarity retrieval jumps both Gemini and GPT-4o
by **+7 COMET** (Gemini 58.3 → 65.8, GPT-4o 58.2 → 65.0). Returns
plateau past 10 shots (20-shot adds <0.5 COMET over 10-shot). At 10+
semantic shots both closed APIs reach roughly 65--66 COMET on
\texttt{eng}$\to$\texttt{xxx}, on par with fine-tuned NLLB-1.3B
(67.3 avg in Table~\ref{tab:main_zs_ft_comet}).

**Scope of the closed_ablations table** (after 2026-05-08 trim):
the retrieval block is restricted to 5-shot only (random vs semantic
for each model). The 10- and 20-shot semantic numbers are kept in
`summary.csv` and will go into a separate **template/shot ablation
table** once the open-LLM (Llama3, Gemma2) template/shot runs are
imported, so that one table compares both groups uniformly. Until
those imports land, the closed-side template/shot story is conveyed
by the existing `figures/new/gpt-vs-gemini/...pdf` curves.

**50-shot dropped (2026-05-08).** We are capping all reported shot
counts at 20: 50-shot adds <1 COMET over 10-shot for both closed APIs
in our data, while doubling token cost and visual clutter. Removed
from `scripts/build_figure4.py` (`SHOTS = [0, 5, 10, 20]`) and from
the bookmark text below; the 50-shot rows remain in `summary.csv` as
provenance but no rendered output references them.

### Final prose — Discussion (one sentence)

> Closed-API quality is far more sensitive to the number of semantic-
> similarity exemplars than to sampling temperature: 10 well-retrieved
> shots close most of the gap to a fine-tuned NLLB-1.3B
> (Appendix~\ref{tab:closed_ablations_comet}).

### Still needs writing — Appendix paragraph (introducing Table~\ref{tab:closed_ablations_comet})

A short paragraph (3--5 sentences) that:

1. Names the three blocks of the table and what they vary.
2. Discloses the one direction asymmetry (no closed-API ablation runs
   on \texttt{xxx}$\to$\texttt{eng}).
3. States the temperature finding (flat) in one sentence so the reader
   does not have to derive it from the numbers.
4. States the shot-count finding ($+7$ COMET from 5$\to$10, then
   plateaus).
5. Mentions the GPT-4o ICL row caveat ($T{=}0$ uses
   semantic-similarity, $T{=}0.6$ uses random retrieval, so its
   temperature row mixes axes — already in the caption, can be very
   brief here).

Draft (paste-ready, edit to taste):

> \paragraph{Closed-API ablations.}
> Table~\ref{tab:closed_ablations_comet} reports three closed-API
> ablations on the six \texttt{eng}$\to$\texttt{xxx} pairs (we did not
> run closed-API ablations on \texttt{xxx}$\to$\texttt{eng}). At
> $T{=}0.6$, Gemini-1.5 moves by at most $0.2$ COMET against its
> $T{=}0$ headline and GPT-4o is unchanged at zero-shot, so we report
> the deterministic ($T{=}0$) results in the body. Holding temperature
> fixed and sweeping the number of semantic-similarity exemplars tells
> a different story: going from 5 to 10 shots adds about $7$ COMET for
> both models (Gemini-1.5 $58.3 \to 65.8$, GPT-4o $58.2 \to 65.0$),
> with returns essentially saturated by 20 shots. At ten or more
> semantic shots the closed APIs are within a couple of COMET points
> of fine-tuned NLLB-1.3B on this direction (Table~\ref{tab:main_zs_ft_comet}).
> The dagger row in the temperature block reflects a configuration
> mismatch (GPT-4o's $T{=}0$ ICL headline uses semantic-similarity
> retrieval, while its $T{=}0.6$ run used random retrieval) and should
> be read as an upper bound on the temperature effect rather than as
> a clean comparison.

### What's still missing

1. Decide where the appendix paragraph lives (Extended Experimental
   Setup, or a dedicated "Closed-API analysis" appendix subsection).
2. Decide whether to also include the chrF mirror inline or just
   reference it.

---

## 2026-05-08 — Template-vs-shot dependence; open-LLM template/shot sweep plan

**Status:** WAITING FOR DATA. Llama3 + Gemma2 (and possibly others)
template/shot sweep is being run / queued for run by user. Once the
results land, follow the plan below, then write either a body
replacement or an appendix paragraph on prompt-engineering effects.

### What we already know (from the closed-API sweep, 2026-05-08)

Holding the prompt template fixed and varying shot count barely moves
performance; the dominant lever is the template itself. Numbers are
COMET avg across the six \texttt{eng}$\to$\texttt{xxx} pairs for
Gemini-1.5 (the only model with full template$\times$shot coverage):

| strategy                       | 5-shot | 10-shot | 20-shot |
|--------------------------------|-------:|--------:|--------:|
| `template3_*` (good template)  | 64.66  | 64.90   | 65.01   |
| `template5_*` (good template)  | 64.80  | 64.96   | 64.98   |
| `semantic_*` (different prompt) | 59.50  | 65.82   | 66.05   |
| `icl_5shot`        (vanilla random)   | 58.33 | -- | -- |
| `icl_5shot_semantic` (vanilla semantic) | 59.50 | -- | -- |

Two interpretations:

1. With a *good* template, 5--20 shots are essentially indistinguishable
   ($\Delta < 1$ COMET).
2. The earlier "$+6$ COMET from 5 to 10 shots" claim was an artifact of
   the prompt template changing between `icl_5shot_semantic` (vanilla)
   and `semantic_10shot` (better-formatted) --- not a shot-count effect.

The open LLMs in Table~\ref{tab:open_vs_closed_icl_comet} currently use
the vanilla `icl_5shot` prompt, so they are likely under-performing
relative to what a template3/5 prompt would deliver.

### Decision rules (lock these in before re-running anything)

- **Shot cap: 10.** If 10-shot beats 5-shot by $> 1$ COMET, we adopt
  10-shot; otherwise we keep 5-shot. We do not consider 20-shot for
  the open LLMs --- the closed-API evidence shows returns plateau
  immediately past 10. (50-shot was dropped from the project entirely
  on 2026-05-08; see the closed-API ablation bookmark above.)
- **Template: choose the per-model best.** For each open LLM we pick
  the (template, shot-count$\le 10$) pair with the highest avg COMET
  on \texttt{eng}$\to$\texttt{xxx} (and report the same configuration
  for \texttt{xxx}$\to$\texttt{eng} when available).
- **Apples-to-apples constraint.** If we update the body Table~2 row
  for an open LLM to a non-vanilla template, we should disclose the
  template choice in methodology and either (a) update the closed-API
  Table~2 row to the same template family for fairness, or (b) keep
  the closed-API row at its current vanilla setting and explicitly
  flag the asymmetry.

### Quick re-run plan once the sweep results land

1. Aggregate: `python -m afriscience_mt aggregate --out experiments/summary.csv`.
2. Pick winners: a small script (TBD: `scripts/select_best_template_shot.py`)
   reads the new rows, applies the decision rules above, and emits a
   per-model JSON like
   `{"llama3_8b": {"template": "5", "shots": 10, "avg_comet": 64.x}}`.
3. **If a winner beats the current vanilla 5-shot row by $> 1$ COMET avg**:
   replace the corresponding row in Table~2 (rebuild via
   `python scripts/build_paper_tables.py` after extending
   `build_open_vs_closed_icl` to honor the per-model strategy choice).
4. **Otherwise**: leave Table~2 alone and put the sweep summary in an
   appendix table + paragraph titled "Effect of prompt engineering on
   open-LLM in-context learning."

### Discussion / appendix prose (to use depending on outcome)

If body is unchanged (open LLMs stay at vanilla 5-shot):

> \paragraph{Prompt engineering at scale.}
> Closed APIs gain about $5$--$6$ COMET on \texttt{eng}$\to$\texttt{xxx}
> when their vanilla 5-shot ICL prompt is replaced by a more carefully
> structured template (templates~3 and~5 in our sweep), at the same
> shot count; further shots add at most $0.7$ COMET on top of that
> (Table~\ref{tab:closed_ablations_comet}). A matching sweep on the
> open LLMs yields broadly the same picture
> (Table~\ref{tab:open_template_shot_sweep}): prompt format is the
> dominant lever and 10-shot is enough --- $20$ and $50$-shot returns
> are within metric noise. We retain the vanilla 5-shot prompt in
> Table~\ref{tab:open_vs_closed_icl_comet} for cross-group
> comparability and report the per-model best configurations in the
> appendix.

If body switches to per-model best template (cap 10 shots):

> \paragraph{Per-model prompt selection.}
> Following our prompt-engineering sweep
> (Table~\ref{tab:open_template_shot_sweep}), each open LLM in
> Table~\ref{tab:open_vs_closed_icl_comet} now reports its highest-
> scoring (template, shot-count) configuration capped at 10 shots
> rather than the vanilla 5-shot baseline; closed APIs are likewise
> reported under their per-pair best template. Beyond 10 shots we
> see $< 0.7$ COMET improvement across all models, so we stop there.

### Still needs writing

- The `scripts/select_best_template_shot.py` selection script.
- The appendix Table `tab:open_template_shot_sweep` and its caption.
- The closed-API counterpart (already partly there in
  `figures/new/gpt-vs-gemini/...pdf` and the closed_ablations table) ---
  decide whether it stays as-is or gets folded into a unified open+closed
  template/shot table.

---

## How to add a new bookmark

1. Add a dated `## YYYY-MM-DD — short title` heading.
2. Open with **Status:** + a one-line "what we're waiting on".
3. Paste the proposed wording verbatim under a `### Proposed text` header.
4. Note **What's still missing** so future-you knows the unblocker.
