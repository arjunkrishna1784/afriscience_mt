# Paper writing — pending decisions and bookmarks

Items captured during drafting that need either user input or specific data
before they can be finalized. Each entry has the proposed wording verbatim
so it can be edited later without re-deriving the argument.

## Style rules (ALWAYS follow when drafting paper text)

- **Never use the word "backbone"** in any paper text or table caption.
  Say "encoder--decoder model" / "seq2seq model" / "base model" /
  the explicit model name instead.
- **Never use the LaTeX `.\` construct** (period followed by backslash-space)
  in captions or prose. Plain `. ` is correct -- the spacing is handled
  by `\frenchspacing` or normal LaTeX. This rule was set 2026-05-08;
  earlier captions with `vs.\ ` etc. have been swept and replaced with
  plain `vs. `.

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

## 2026-05-08 — Table 2 ICL row: swap vanilla -> templated (DEFERRED)

**Status:** WAITING. Don't change Table 2 yet --- wait for the open-LLM
(Llama3-8B, Gemma2-9B, and any peers) template/shot sweep to land in
`summary.csv` first. Once it does, pick the per-model best template at
$\le 10$ shots and swap the rows uniformly across closed and open
groups so the table stays apples-to-apples.

### Why this is deferred

Two pieces of evidence have to be in hand before we touch Table 2:

1. **Closed-API templated runs already qualify** for replacement.
   Vanilla `icl_5shot` at $T{=}0$ for Gemini-1.5-Flash sits at COMET
   58.3 (eng->xxx avg); the same model with `template3_5shot` /
   `template5_5shot` lands at 64.6 / 64.8 --- a $\sim 6$ COMET gap that
   easily clears the "$> 1$ COMET" decision rule. GPT-4o has the same
   gap (`icl_5shot_semantic` 58.2 vs `semantic_5shot` 64.6).

2. **Open-LLM templated runs are not yet imported.** As of 2026-05-08
   `experiments/llm/open/icl/<model>/` only has `0shot/` and `5shot/`
   directories using the vanilla `few_shot` template (config field
   confirms). If we swap closed-API rows now, the table compares
   open-LLM vanilla against closed-API templated --- which is exactly
   the asymmetry we are trying to eliminate.

### Action plan once the open sweep lands

1. Run `python -m afriscience_mt aggregate --out experiments/summary.csv`.
2. Use a small selection script (TBD: `scripts/select_best_template_shot.py`,
   already mentioned in the earlier template-vs-shot bookmark) to find
   the per-model best `(template, shots)` configuration with `shots
   <= 10` and emit per-model JSON, e.g.
   `{"llama3_8b": {"template": "5", "shots": 10, "comet_avg": 64.x}}`.
3. Extend `build_open_vs_closed_icl` in
   `scripts/build_paper_tables.py` to read that per-model choice when
   filling the 5-shot ICL block (or rename the row label to
   "ICL (best template, $\le$10 shots)" if shot counts vary
   across models).
4. Regenerate paper tables and verify the row labels in
   `tables/new/open_vs_closed_icl_{comet,chrf}.tex` reflect the
   actual per-model setup.
5. Update Table 2's caption and the body prose: explicitly say which
   template each row uses, and reference
   `Appendix~\ref{tab:closed_ablations_comet}` for the
   prompt-engineering ablation.

### Headline-shift risk

Swapping vanilla -> templated lifts closed-API ICL avg by $\sim 6$
COMET, putting Gemini-1.5 (~65) and GPT-4o (~64.5) within $\sim 2$
COMET of fine-tuned NLLB-1.3B (67.3). That changes the body story
from "fine-tuning still wins by a wide margin against ICL" to
"fine-tuned NLLB barely edges templated closed-API ICL". The body
prose currently sells the wider margin --- be ready to soften that
language when the swap lands.

### Reminder: 50-shot is still off the table

The shot cap is 10 for the open-LLM sweep (decision rule above). The
project-wide 50-shot drop (2026-05-08) means we will not consider any
configuration above 20 shots even if a row scores marginally higher
there.

---

## 2026-05-08 — DECISION: template5 + 10-shot is the universal ICL config

**Status:** LOCKED. Use `template5_10shot` for every non-fine-tuned LLM
in the body tables. This decision supersedes earlier "5-shot vs 10-shot"
deliberations.

**Evidence (bidirectional avg COMET, T=0, eng->xxx + xxx->eng):**

| model              | template5_10shot | per-model best                | penalty |
|--------------------|-----------------:|-------------------------------|--------:|
| Gemini-1.5-Flash   | **65.96**        | template5_10shot (65.96)      |   0.00  |
| GPT-4o             | 65.77            | template3_10shot (65.87)      |   0.10  |
| Llama3-8B          | **46.94**        | template5_10shot (46.94)      |   0.00  |
| Gemma2-9B-IT       | **51.44**        | template5_10shot (51.44)      |   0.00  |

template5_10shot is best for 3/4 reference models and only $0.10$ COMET
below GPT-4o's peak. Picking template3_10shot instead would penalize
Llama3 by $-0.62$ and Gemma2 by $-0.90$.

**Application:**

- All non-fine-tuned LLM rows in Table 2 / closed_ablations / appendix
  ICL tables use `template5_10shot` at $T{=}0$.
- Fine-tuned LLMs stay locked to their training-time prompt
  (vanilla `few_shot`); they are not re-prompted with template5.
- 50-shot remains banned project-wide; 20-shot is allowed in appendix
  ablations only.

**Script defaults:** `scripts/rerun_open_llm_template.py` ships with
`--template 5 --shots 10` so the 5 remaining open LLMs
(AfriqueLlama-8B, AfriqueQwen-8B, Tiny-Aya-Earth, Tiny-Aya-Global,
TranslateGemma-12B) inherit the locked config without further
decisions.

---

## 2026-05-09 — Open-LLM templated rerun: 3 of 5 models degenerated

**Status:** RE-RUN PENDING. Same fix family as Tiny-Aya-G (2026-05-07);
re-running the 3 broken models with anti-degeneration controls
(`--anti-degeneration`, default `rep_penalty=1.08`, `freq_penalty=0.0`,
explicit chat-template stop tokens).

**What we found.** After running
`scripts/rerun_open_llm_template.py` (default config: `template5_10shot`,
random retrieval, T=0) on the 5 missing open LLMs, three of them
showed the COMET-much-greater-than-ChrF degeneration signal:

| model              | ChrF  | COMET | gap pattern       |
|--------------------|------:|------:|-------------------|
| AfriqueLlama-8B    |  6.91 | 47.22 | wrong-language    |
| AfriqueQwen-8B     | 11.44 | 41.63 | echo-the-prompt   |
| TranslateGemma-12B | 12.43 | 44.46 | empty/partial out |
| Llama3-8B          | 27.71 | 46.94 | normal            |
| Gemma2-9B-IT       | 34.94 | 51.44 | normal            |
| Tiny-Aya-Earth     | 34.01 | 55.32 | normal            |
| Tiny-Aya-Global    | 33.78 | 55.14 | normal            |
| Gemini-1.5         | 49.47 | 65.96 | normal            |
| GPT-4o             | 51.34 | 65.77 | normal            |

**Per-model failure modes (eyeballed from predictions.json):**

- **AfriqueLlama-8B** keeps generating English continuations of the
  source (paper-header style: ``a mathematical modelling study\n  •
  1International Research Centre of Health, University of
  Florence...``) instead of translating to the target.
- **AfriqueQwen-8B** echoes the prompt itself back, sometimes with a
  DOI prefix; on Amharic targets it outputs the *instruction* in
  Amharic and then the example list verbatim. Classic
  off-distribution behaviour.
- **TranslateGemma-12B** truncates after a partial Amharic
  translation and emits an empty hyp on many rows.

The COMET-vs-ChrF gap is the same diagnostic that flagged Tiny-Aya-G's
sentence-doubling on 2026-05-07 (`docs/PAPER_TODOS.md` "2026-05-07 --
Tiny-Aya-G xxx->eng degeneration"). COMET tolerates fluent text in
*any* language; ChrF requires lexical overlap with the reference. When
they diverge by tens of points, the model is generating something
fluent that isn't actually a translation.

**Remediation (in flight):**

`scripts/rerun_open_llm_template.py` now has `--anti-degeneration`
which routes vLLM through a custom `SamplingParams` with
`repetition_penalty=1.08`, `frequency_penalty=0.0`, and explicit
chat-template end-of-turn stop tokens (`<|END_OF_TURN_TOKEN|>`,
`<|end_of_turn|>`, `<end_of_turn>`, `<|im_end|>`, `<|eot_id|>`).
The metadata fields `anti_degeneration`, `rep_penalty`,
`freq_penalty` are recorded in each `results.json` so we can later
distinguish the two runs.

**Server command to fill the gap:**

    python scripts/rerun_open_llm_template.py \
        --models afriquellama_8b afriqueqwen_8b translategemma_12b \
        --anti-degeneration

(no `--skip-existing`: we *want* to overwrite the broken results.)

**What's still missing.** After the 3-model rerun lands, decide:
- Are the new ChrF values in the 25-50 range (matching Llama3 / Gemma2 /
  Tiny-Aya)? If yes, ship; if not, try `--rep-penalty 1.15` (the
  v1 Tiny-Aya-G setting) and re-evaluate.
- Whether to also re-run the 2 already-clean Aya models (Earth and
  Global) with `--anti-degeneration` for a uniform decoding setup
  across the 5 open LLMs we filled in. They have no degeneration so
  it's optional.

**Lesson for the paper (locked in the Tiny-Aya-G appendix paragraph):**
the COMET >> ChrF gap is a cheap, free degeneration signal --- worth
flagging in the methodology paragraph as a recommendation.

---

## 2026-05-09 — chrF strategic-inclusion policy

**Status:** Locked policy for the paper. Not all chrF mirrors of
COMET tables are inputted; we add chrF only where it earns the page.

**The rule.** chrF appears in the paper only when:

1. The metric-divergence story actually changes the conclusion in
   that table (e.g. ranking flips, COMET~$\gg$~chrF degeneration
   visible in the row), OR
2. A claim in the prose nearby explicitly compares the two metrics.

**Where chrF is currently in the paper (keep these):**

- `tab:prompt_engineering_open` --- COMET and chrF are inline,
  side-by-side. The COMET-vs-chrF gap is the story (degeneration on
  3 of 7 open LLMs).
- `tab:mafand_nllb` --- COMET on top, chrF on bottom in the same
  table for the cross-domain transfer story.
- The Tiny-Aya-G appendix paragraph (Decoding controls and manual
  inspection) cites concrete chrF numbers inline (lug-eng:
  $47.0 \to 44.9$ chrF vs.\ $63.9 \to 21.9$ COMET).

**Where chrF mirrors are *generated but not inputted*:**

- `tables/new/main_zs_ft_chrf.tex`
- `tables/new/open_vs_closed_icl_chrf.tex`
- `tables/new/doc_chrf.tex`
- `tables/new/closed_ablations_chrf.tex`
- `tables/new/template_shot_ablation_chrf.tex`
- `tables/new/lora_chrf.tex`
- `tables/new/seq2seq_chrf.tex`, `seq2seq_smaller_chrf.tex`
- `tables/new/llm_sentence_chrf.tex`

These exist for ad-hoc inspection while writing. If reading the body
prose surfaces a metric-divergence claim that needs a full chrF table
to back it up, the file is one `\input{tables/new/<name>_chrf}` away
in the appendix.

**Trigger conditions for adding a chrF mirror later:**

- A reviewer asks "does the seq2seq winner change on chrF?"
- A specific language pair flips between metrics in a body claim.
- We argue NLLB-1.3B leads on COMET but a different system tops chrF
  on $X$ pairs.

If we add a new chrF mirror, document the *prose claim* it supports
in this bookmark.

---

## 2026-05-09 — Removing "50-shot" from the paper (post-cap decision)

**Status:** ACTION REQUIRED in Overleaf. The
2026-05-08 decision capped shot counts at 20; rendered tables and
build script are clean, but the paper text still references 50-shot
in three places. List below shows exactly what to edit.

### `acl_latex.tex`

**Line 459 (results prose):** four mentions of "50-shot" in the
sentence-level ICL paragraph; the sentence cites stale numbers from
\Cref{tab:models_performances_templates_eng_xxx,tab:models_performances_templates_xxx_eng}.
Replace the entire paragraph with a tighter version that cites the
new \Cref{tab:open_vs_closed_icl_comet} (template5\_10shot) numbers.

**Line 461 (results prose):** three mentions of "50-shot" in the
"few-shot learning provides limited and inconsistent gains"
paragraph. Replace with the
\Cref{tab:template_shot_ablation_comet} 10-vs-20 plateau finding
(gain $\le 0.2$ COMET past 10 shots).

**Line 463:** references the now-stale figure
`fig:template_shots_gpt_gemini_eng_xxx` and stale tables
`tab:models_performances_templates_*`. Replace with
\Cref{tab:prompt_engineering_open} for the prompt-engineering claim
and \Cref{fig:gpt_gemini_eng_xxx,fig:gpt_gemini_xxx_eng} for the
template$\times$shot curves.

### `appendix_plots_and_tables.tex`

**Line 130 (caption of `fig:gpt_gemini_eng_xxx`):**
`shot counts $\in\{0,5,10,20,50\}$` $\to$ `$\{0,5,10,20\}$`.

**Line 151 (caption of `fig:gpt_gemini_xxx_eng`):** same change.

**Line 351 (`\input{table_icl_llm}`):** *remove* this line. The
inputted file (`table_icl_llm.tex`) contains the stale Tables 11/12
with 50-shot rows + older models (Llama-3.2-3B, Llama-3.1-8B). It is
fully superseded by `tab:template_shot_ablation_comet` (which is
already inputted nearby).

**Line 377 (`\input{table_icl_llm_doc}`):** review whether the doc-
level template/shot tables in `table_icl_llm_doc.tex` still add
information beyond `tab:doc_comet`. If they only repeat what the
new doc table covers, *remove* this `\input` too.

### `figures/new/gpt-vs-gemini/`

The .pdf files were generated when `SHOTS=[0,5,10,20,50]` in
`scripts/build_figure4.py`. The script now defaults to
`SHOTS=[0,5,10,20]`; rerunning it on the box that has matplotlib +
`experiments/summary.csv` will regenerate the four .pdfs at the
correct shot range. After regeneration, replace the .pdfs in the
overleaf project so the rendered figure matches the new caption.

### Spider plot (`fig:spider_per_pair`) -- GPT-4o config drift

**Status:** ACTION REQUIRED. The spider PDFs at
`figures/new/spider_per_pair/{comet,chrf}/<pair>.pdf` were rendered
from GPT-4o's *per-domain* SSA-COMET data, which only exists for
`template3_5shot` (xxx-eng at T=0; eng-xxx at T=0.6 because the
T=0 eng-xxx templated runs landed only after the spider rescore).
The legend (line 194) and caption (line 197) of
`appendix_plots_and_tables.tex` accurately describe the rendered
data ("GPT-4o (5-shot ICL, template 3)") but disagree with the
body Table~2 config (`template5_10shot`) for that model.

Pick one fix:

1. **Text-only fix (cheaper).** Leave the spider data as-is; clarify
   in the caption that the GPT-4o panel uses an earlier
   `template3_5shot` configuration retained for the per-domain
   rescore. One sentence in the caption:
   *"The GPT-4o panel uses the earlier `template3_5shot` setup
   that was rescored per-domain; differs from the headline
   `template5_10shot` in Table~\ref{tab:open_vs_closed_icl_comet}
   by $\le$ 0.2 COMET on average."*
2. **Data fix (cleaner, needs GPU).** Run
   `python scripts/rescore_per_domain_ssa.py --src
   experiments/llm/closed/t0/icl/gpt4o/template5_10shot` on a box
   with the SSA-COMET model. Then re-render the spider PDFs (the
   spider build script reads from the per-domain field). Update the
   legend/caption accordingly:

   - Line 194: `\textcolor{spcGPT}{\textbf{$\cdot\cdot\cdot$}}
     GPT-4o (5-shot ICL)` $\to$ `... GPT-4o (10-shot ICL, template5)`
   - Line 197: `GPT-4o (5-shot ICL, template 3)` $\to$
     `GPT-4o (10-shot ICL, template5)`

   The same per-domain rescore should ideally cover Gemini-1.5
   `template5_10shot` too if Gemini shows up in any future spider
   variant.

We have per-domain data already for:
- NLLB-1.3B FT (`experiments/seq2seq/finetune/nllb_200_1_3b/dcs`)
- Gemma2-9B-IT LoRA r=64 (`experiments/llm/open/finetune/gemma2_9b_it/r64`)
- GPT-4o `template3_5shot` (T=0 xxx-eng + T=0.6 eng-xxx)

Missing per-domain data: GPT-4o `template5_10shot` (the locked body
config).

---

## 2026-05-09 — Multilingual vs bilingual: original claim is too strong

**Status:** new appendix table ready
(`tables/new/multilingual_vs_bilingual.tex`,
`\label{tab:multilingual_vs_bilingual}`). The body needs one
sentence rewritten -- the old "bilingual outperforms multilingual
across all model architectures" claim is only true for the seq2seq
model, not for the LoRA LLMs.

### What the new table shows (eng$\to$xxx avg)

| Family | Bilingual best | Multilingual best | Verdict |
|---|---|---|---|
| NLLB-600M | 66.7 COMET / 51.7 chrF (FT) | 39.8 COMET / 30.2 chrF (400k MMT) | Bilingual dramatically better; MMT ckpts are short/early-stage |
| Llama-3.1-8B-Instruct LoRA | $\sim$57.3 COMET (best non-r64 rank) | $\sim$57.4 COMET (r=16) | **Essentially tied -- the outlier** |
| Gemma-2-9B (non-instruct) LoRA | 62.5 COMET / 50.4 chrF (r=128) | 58.8 COMET / 45.5 chrF (r=32-64) | Bilingual leads by $\sim$3.7 COMET / $\sim$5 chrF |

### Where to edit the body in `acl_latex.tex`

Find the sentence in the LoRA / multilingual training discussion that
asserts bilingual > multilingual "across all model architectures"
(the old claim was around line 419 in an earlier draft -- search for
"outperforms" near a bilingual/multilingual comparison). Replace with
something like:

> Bilingual fine-tuning outperforms multilingual fine-tuning for
> NLLB and for Gemma-2-9B LoRA, both at SSA-COMET and chrF
> (Table~\ref{tab:multilingual_vs_bilingual}); the only exception
> is Llama-3.1-8B-Instruct LoRA, where bilingual and multilingual
> training end up essentially tied at SSA-COMET ($\sim$57) across
> matching ranks.

### Caveats baked into the caption (do not lose in the body)

1. NLLB MMT checkpoints are short/early-stage runs -- the gap is
   partly a steps-of-training confound, not purely a
   bilingual-vs-multilingual difference.
2. **Llama r=64 bilingual is anomalous** (chrF 49.0, the highest in
   the bilingual block, while COMET is 53.8, the lowest). The other
   bilingual ranks {8, 16, 32, 128} cluster at chrF~46 / COMET~57.
   This is a chrF-up/COMET-down degeneration signature -- worth
   flagging if Llama r=64 numbers appear elsewhere in the paper.

### Why is Llama the outlier? (open question)

Two leads worth noting if a reviewer or co-author asks:

- **Instruct vs base model.** Llama-3.1-8B-*Instruct* (RLHF'd) is
  on both sides; Gemma-2-9B is the *non-instruct* base on both
  sides. The instruction-tuned starting point may be flexible
  enough that the multilingual shared parameters do not cost
  anything per pair, whereas the raw base benefits from per-pair
  specialization.
- **Llama r=64 bilingual is the degenerate outlier** (caveat #2
  above) -- without it, the Llama bilingual cluster is also a few
  COMET below the Gemma bilingual gain. Hard to rule out that the
  full Llama bilingual ceiling is just lower than what a clean run
  would produce.

### What's still missing

- Decision on whether to re-train the NLLB MMT checkpoints longer
  (so the seq2seq comparison is apples-to-apples in updates), or
  just keep the disclaimer.
- Overleaf-side edit of the body sentence (above) once the user
  confirms the wording.

---

## 2026-05-12 — Closed-API rerun asymmetry: GPT-5.4 will have ~half the templated configs Gemini-3.1-flash-lite has

**Status:** observation surfaced while monitoring the four parallel closed-API
reruns (`run_everything.sh`). User to confirm whether the asymmetry is acceptable
as-is or whether the GPT gap should be filled with newly-built (non-replay) prompts.

### Observation

The reruns replay byte-identical prompts from the legacy TSV dumps under
`/ext_data/idris/DCS-MT-Updated/old_experiments/LLMS/<model>_temp_<T>/`. Counted
TSVs:

| Legacy dump | xxx→eng templated | eng→xxx templated | total TSVs |
|---|---|---|---|
| `gpt-4o_temp_0.0`            | ✅ 6 pairs × 11 conds = 66 | ❌ (only semantic-similarity strategies exist) | 66 |
| `gpt-4o_temp_0.6`            | ❌                          | ✅ 6 × 12 = 72                  | 72 |
| `gemini-1.5-flash-002_temp_0.0` | ✅ 6 × 11 = 66          | ✅ 6 × 11 = 66                  | 132 |
| `gemini-1.5-flash-002_temp_0.6` | ✅ 6 × 11 ≈ 66          | ✅ 6 × 11 ≈ 66                  | ~132 |

This matches the structural gap documented at `scripts/rerun_gpt4o_templated.py:8-14`:
> The closed-API audit (2026-05-08) found one structural gap: GPT-4o
> ``eng->xxx`` at T=0 only has the semantic-similarity strategies; the
> templated runs (templates 1..5) live exclusively at T=0.6.

GPT-4o was originally run **split**: templates at T=0 only for xxx→eng, templates
at T=0.6 only for eng→xxx. Gemini-1.5 was run **symmetric**: both directions at
both temperatures.

### Effect on the in-flight reruns

After all four sweeps land (`gpt54` × T={0, 0.6}, `gemini_31_flash_lite` × T={0, 0.6}):

- **gpt54** ≈ **126 templated configs** total — each (pair, template, shots) cell
  appears at **exactly one** temperature (xxx→eng at T=0, eng→xxx at T=0.6).
- **gemini_31_flash_lite** ≈ **216 templated configs** total — every cell at **both**
  temperatures.

This is **not duplication on Gemini's side** — it's missing coverage on GPT's side
inherited verbatim from the legacy design. Comparable cells *do* exist between the
two models (every cell appears under at least one temperature), but the temperature
ablation only exists for Gemini.

### Implications for the paper

- **For any plot/table that compares closed APIs at a fixed temperature**: only the
  cells that exist for *both* models are comparable. Concretely:
    - xxx→eng templated, T=0 — GPT-5.4 ✅, Gemini-3.1-flash-lite ✅
    - eng→xxx templated, T=0.6 — GPT-5.4 ✅, Gemini-3.1-flash-lite ✅
    - The other two quadrants exist only for Gemini.
- **For closed-API temperature ablations**: only Gemini supports a head-to-head
  T=0 vs T=0.6 comparison across every cell. GPT can only be temperature-compared
  by switching direction simultaneously, which confounds the ablation.

### Options

1. **Accept the asymmetry** (cheap). Note it explicitly in the closed-API table
   captions and skip GPT-5.4 from any temperature-ablation plot. Same caveat as
   the legacy gpt-4o/gemini-1.5 comparison the paper already lives with.
2. **Fill the GPT gap** (~$$, ~hours). Build new prompts (using the current
   `build_translation_prompt()` templates) for the two missing GPT quadrants —
   xxx→eng at T=0.6 and eng→xxx at T=0 — and run them against gpt-5.4. Would
   require a new `scripts/rerun_gpt54_fillgaps.py` (no source TSVs to replay).
   Resulting prompts would not be byte-identical to the original gpt-4o run, so
   comparison to legacy gpt-4o becomes prompt-not-identical for those cells.

### What's still missing

- User decision: option 1 vs option 2.
- If option 2, agreement on whether to use template5 + 10-shot only (the paper's
  canonical config per the 2026-05-08 decision) or the full template/shot sweep.

---

## 2026-05-13 — Closed-API rerun audit: ICL done; doc has real gap; vanilla-template question open

**Status:** ICL sweeps for `gpt54` and `gemini_31_flash_lite` are complete or
on-track. Audited `summary.csv` (gpt4o keys) against `gpt54` outputs on disk
applying current scope policy (no 50-shot, no `template5_0shot`, T=0.6 +
semantic = ablation eng→xxx only). Two unresolved items below.

### What's done

- **ICL T=0 (`gpt54`)** — 126 dirs on disk; matches scope-filtered plan exactly.
  All 24 "missing vs gpt4o" cells are out-of-scope (50-shot policy + invalid
  `template5_0shot` + xxx→eng excluded for semantic).
- **ICL T=0.6 (`gpt54`)** — 54 dirs on disk; all 18 "missing vs gpt4o" cells
  are out-of-scope (50-shot policy + xxx→eng excluded for ablation +
  `template5_0shot` invalid).
- **ICL T=0 (`gemini_31_flash_lite`)** — running, 32/122 done at audit time.
  Plan is the same 126 cells as `gpt54` t0 → 1-to-1 head-to-head guaranteed
  at T=0.

### Doc audit: real gap

`gpt54` doc currently has only `doc_{0,1}shot_template1` (24 dirs,
bidirectional). With the **5-shot drop now extended to doc** (`doc_5shot_*`
out per project policy), the remaining doc work for `gpt54` is:

- `doc_{0,1}shot_template2` — 12 dirs each
- `doc_{0,1}shot_template3` — 12 dirs each
- **Total: 4 strategies × 12 dirs = 48 cells**

The `doc_{0,1}shot_legacy` rows in gpt4o summary are gpt-4o-only artifacts
(legacy prompts that can't be replayed) — not part of the rerun plan.

### Vanilla-template (open-vs-closed) question

Open models (`llama3_8b`, `gemma2_9b_it`, `tiny_aya_*`, `afriquellama_8b`,
`afriqueqwen_8b`, `translategemma_12b`) use **vanilla** strategies that the
closed-API runs do NOT have:

| Slice | Open vanilla strategy | Closed counterpart |
|---|---|---|
| ICL | `0shot`, `5shot` | `template{1,2,4}_0shot` (≈ 0shot), `template{3,5}_5shot` (5-shot) |
| Doc | `doc_0shot`, `doc_1shot` | `doc_{0,1}shot_template{1,2,3}` |

Closed APIs have only the templated variants. Open models have only the
vanilla variants. **Effect:** any direct closed-vs-open table at the doc or
ICL level requires picking one templated cell as "the" closed proxy for the
open vanilla, since they aren't byte-identical prompts.

**Open question:** do we need to also run `gpt54` (and
`gemini_31_flash_lite`) on the vanilla `0shot`/`5shot` ICL prompts and
`doc_0shot`/`doc_1shot` doc prompts to enable direct closed-vs-open
comparison, or is the templated proxy acceptable?

### What's still missing

- User decision on vanilla-template question (run vanillas on closed APIs, or
  document the templated-vs-vanilla mismatch as a known caveat).
- If running vanillas: a script that emits the vanilla prompts (no legacy TSV
  to replay) and writes to `experiments/llm/closed/{icl,doc}/<short>/<vanilla
  strategy>/<pair>/`.
- Launch `doc_{0,1}shot_template{2,3}` for `gpt54` (48 cells) — confirmed
  in-scope.

### Cross-refs

- Scope policy: this doc, sections "2026-05-08" (50-shot drop) and
  "2026-05-12" (closed-API asymmetry).

---

## 2026-05-13 09:15 — Closed-API rerun execution snapshot (mid-run)

**Status:** All audit-flagged jobs were launched ~05:08; gemini side
finished its T=0 stage cleanly and chained into the T=0.6 e2x ablation
at 08:59. gpt54 doc + vanilla still in-flight on OpenAI batch (latency-
bound, not stalled). gpt54 T=0.6 ablation was already complete from
2026-05-12 (writes to `t6/`, not `t06/`, due to `temp_setting()` quirk
in `rerun_gpt4o_templated.py:221`).

### Active processes

| PID | Job | Started | Notes |
|---|---|---|---|
| 3308418 | wrapper `run_rerun_gemini_31_flash_lite.sh` | May 12 | parent of 3396573 |
| 3367683 | `rerun_openai_doc.py --temperature 0.0` (gpt54) | 05:08 | 3 OpenAI batches in-flight |
| 3367688 | `rerun_openai_vanilla.py --temperature 0.0` (gpt54) | 05:08 | 3 OpenAI batches in-flight |
| 3396573 | `rerun_gemini_newer.py --temperature 0.6 --direction e2x` (gemini) | 08:59 | 6 Gemini batches in-flight |

### Completed configs (on disk)

| Model  | ICL T=0 | doc T=0 (templ 1+2+3) | vanilla sent (0+5) | vanilla doc (0+1) | T=0.6 e2x ablation |
|---|---|---|---|---|---|
| **gpt54**  | 126 / 126 ✅ | 34 / 72 | 8 / 24 | 0 / 24 | **54 / 54 ✅** |
| **gemini** | 126 / 126 ✅ | 72 / 72 ✅ | 24 / 24 ✅ | 24 / 24 ✅ | 1 / 54 |

**Total remaining: 131 cells** = 38 (gpt54 doc) + 16 (gpt54 vsent) + 24
(gpt54 vdoc) + 53 (gemini t6 e2x).

### Output-path gotcha

`temp_setting(0.6) = "t6"` not `"t06"` — both `rerun_gpt4o_templated.py`
and `rerun_gemini_newer.py` use this. So all T=0.6 outputs land under
`experiments/llm/closed/t6/icl/<short>/`. Anyone looking for `t06/gpt54`
will see no results; the data is at `t6/icl/gpt54/` (54 cells, untracked
in git as of this snapshot).

### Quality verification

Sampled 6 gemini configs across ICL / doc / vanilla / sent. All show
real translations (avg hyp len 148–4786 chars, 0–1 empty samples per
config out of 28–970). Healthy — not a fail mode.

### What's still missing

- gpt54 doc + vanilla completion (OpenAI batch wall time ~ hours-to-day).
- gemini t6 e2x ablation completion (~3–4 h based on stage 1 pace).
- Commit + push the `t6/` and `doc/`/`t0/icl/` new outputs once jobs
  drain (currently untracked in git).
- After all 5 remaining jobs settle: run `python -m afriscience_mt
  aggregate --out experiments/summary.csv` then SSA-COMET rescore
  (`scripts/rescore_per_domain_ssa.py`) on the new closed configs.

### Cross-refs

- Scope policy: see prior 2026-05-13 audit bookmark above.
- Output-path memory: [[closed-api-rerun-scope]] (also notes 50-shot,
  template5_0shot, semantic xxx→eng exclusions).

---

## 2026-05-13 13:00 — GPT-5 silent-empty-hyp lesson (advice for the paper)

**Status:** During gpt-5.4 reruns we discovered the model silently returns
empty completions when `max_completion_tokens` is too small relative to
the (reasoning + visible-output) budget. This costs API credits, breaks
chrF/BLEU/COMET, and is **not flagged as an error** by the API. Worth a
short methodological warning in the paper so others avoid the trap.

### What happened

| Cell | empty/n |
|---|---|
| `gpt54 doc_0shot_template1 eng-amh` | 26/28 (93%) |
| `gpt54 template5_5shot yor-eng` | 231/970 (24%) |
| `gpt54 semantic_20shot eng-amh` | 355/970 (37%) |

Default `max_completion_tokens` was 4096 (doc) / 2048 (sentence). Combined
with gpt-5's reasoning tokens consuming the same budget and Amharic's
~3-4 tokens/character cost in Ethiopic script, the model exhausted its
output budget on reasoning alone and returned empty `content` for many
prompts. The OpenAI batch API reported these as `completed` (no failure).

### Fix applied (verified in this run)

1. `max_completion_tokens=16384` for all gpt-5 translation calls.
2. `reasoning_effort=low` (translation rarely benefits from heavy CoT,
   and low effort frees ~3-5× more visible-output budget for the same
   token cap, **and is meaningfully cheaper per call**).
3. Explicit per-row empty-hyp audit before trusting metrics.

### Proposed paper text (Methods or Limitations)

> "We observed that GPT-5 series models can silently return empty
> completions when `max_completion_tokens` is set too low: reasoning
> tokens consume the same budget as visible output, and translations
> into morphologically rich or non-Latin-script targets (Amharic, Yorùbá,
> Northern Sotho) routinely exceed the GPT-4-era 4096 default. The API
> returns no error; the empties surface only as zero metric scores.
> We set `max_completion_tokens=16384` and `reasoning_effort=low` for
> all GPT-5 calls. We also recommend that future closed-API translation
> work explicitly audit prediction files for empty hypotheses before
> aggregating metrics."

### What's still missing

- Confirm the reasoning_effort=low + max_completion_tokens=16384 fix
  produces 0/970 empties in the rerun currently in flight (gpt54 ICL +
  doc + vanilla, kicked off 2026-05-13 13:00).
- Cells found bad (201 total: 136 t0 ICL + 49 t6 ICL + 16 doc) were
  *moved* to `experiments/llm/closed/discarded/<original_path>/` so the
  failure data is preserved for diagnostic / paper figures.
- Cite the OpenAI docs page that documents reasoning-token budget
  semantics (search needed).

### Cross-refs

- Memory: [[closed-api-max-tokens-lesson]] (operational summary).
- Code: `afriscience_mt/api/openai_client.py` reads
  `OPENAI_REASONING_EFFORT` env var (default "medium"); wrappers set
  it to "low".
- Audit: 201 bad cells under `experiments/llm/closed/discarded/`.

---

## How to add a new bookmark

1. Add a dated `## YYYY-MM-DD — short title` heading.
2. Open with **Status:** + a one-line "what we're waiting on".
3. Paste the proposed wording verbatim under a `### Proposed text` header.
4. Note **What's still missing** so future-you knows the unblocker.
