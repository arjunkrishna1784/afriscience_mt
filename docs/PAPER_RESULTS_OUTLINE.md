# Paper results section — proposed structure

This is a planning aid for the results section, written from the data
currently on disk plus the in-flight domain-balance ablation. Numbers are
derived per `experiments/summary.csv` and the per-domain breakdown that the
test set's `category` column makes free.

Reporting follows the project metric hierarchy (`docs/EXPERIMENTS.md`):
**SSA-COMET primary, chrF in any two-metric table, BLEU only when a prior
paper we cite quoted it.**

The narrative arc moves from broad to narrow: a benchmark headline, then
architectural comparison, then targeted ablations on data composition and
methodology, closing with a per-domain analysis that ties back to the
small-domain claim.

## §1 Headline benchmark (the table everyone reads)

**One table, four blocks:** zero-shot seq2seq, fine-tuned seq2seq, LLMs
(zero-shot / ICL / LoRA), closed APIs. Rows = best system per block (or top
2). Columns = the 12 directional pairs grouped as `eng→X` then `X→eng`.
Cell = SSA-COMET. Best per column bolded.

Use `afriscience-mt table --metric ssa_comet --layout model_x_pair --bold-best --format tex`.

**Headline claims to support from this single table:**
- Best overall system per pair across all settings (likely a fine-tuned NLLB or one of the LoRA llama3_8b settings, depending on language).
- The fine-tuning gain over pre-trained baselines is substantial on every African pair (single sentence with the average delta in COMET).
- Direction asymmetry: `X→eng` typically wins over `eng→X` on most systems — quantify and explain (English-rich pretraining).

## §2 Pretrained vs fine-tuned (the "does fine-tuning help" question)

Side-by-side pre-trained zero-shot vs DCS fine-tune for each NLLB / M2M-100
size. Same metric, same pairs. One sentence per row in the prose; the table
carries the rest.

Two intended takeaways:
- DCS fine-tuning helps every pair we have tested (state magnitude).
- Diminishing returns on the larger NLLB-1.3B vs distilled-600M only show
  up where pretraining was already strong — call this out for the paper's
  "when do you need a bigger model?" subsection.

## §3 Data composition: MAFAND vs MAFAND++

This is the cleanest ablation in the paper. Same model
(`nllb_200_1_3b`), same pairs (the 4 MAFAND African languages × 2
directions = 8 pairs), only the training corpus differs:

| condition | training data | role |
|---|---|---|
| MAFAND | `dataset = "mafand"` | baseline |
| MAFAND++ | `dataset = "mafand_pp"` | + DCS scientific data |

Use `afriscience-mt table --filter-type seq2seq_finetune --filter-model nllb_200_1_3b --layout dataset_compare --metric ssa_comet --include-both --format tex`.

**Claim shape:** "Adding the scientific DCS data on top of MAFAND
yields +X COMET on average across the 8 MAFAND-supported pairs, with the
largest gain on Y." Pair the COMET delta with a chrF delta in a footnote.

If the gain is small or negative on some pair, note it explicitly — that
is the most credible kind of ablation report.

## §4 Domain composition: temperature-sampled balance (ablation)

Three points along the τ axis on the same training corpus and the same
per-pair test set (NLLB-1.3B):

| τ | per-example weight | source |
|---|---|---|
| 1 (proportional) | natural | existing `seq2seq/finetune/nllb_200_1_3b/dcs/` |
| 2 (sqrt) | `(1/N_d)^(1/2)` | `balance_t2/` runs |
| ∞ (uniform) | `1/N_d` | `balance_tinf/` runs |

### Result (computed 2026-05-07): balance hurts uniformly, no small-domain win

The hypothesis was "small domains gain, large domains lose under
balancing." **The data contradicts that.** Every domain loses ChrF when
moving from τ=1 → τ=∞, including the smallest (chemistry, N=46,
−1.5 ChrF). τ=2 sits in the middle, also worse than τ=1 everywhere.

| metric | τ=1 best cells | τ=2 best | τ=∞ best |
|---|---|---|---|
| SSA-COMET (12 pairs) | 11 / 12 | 1 / 12 (zul→eng tie) | 0 / 12 |
| Avg ChrF (avg over 11 domains) | 57.5 | 56.4 | 56.0 |

Per-domain Δ ChrF (τ=∞ − τ=1) ranges from **−0.6 (CS) to −1.9 (Biology)**;
no domain is positive. Assets:
- [tables/new/balance_chrf.tex](../tables/new/balance_chrf.tex), [tables/new/balance_comet.tex](../tables/new/balance_comet.tex)
- [figures/new/balance_heatmap.pdf](../_ACL2026__Decolonizing_Science_in_Africa_through_Text_Translation_/figures/new/balance_heatmap.pdf) — 3 × 11 heatmap
- [figures/new/balance_delta_by_size.pdf](../_ACL2026__Decolonizing_Science_in_Africa_through_Text_Translation_/figures/new/balance_delta_by_size.pdf) — Δ ChrF per domain

### Revised claim shape

> "Re-weighting the training distribution toward smaller domains via
> temperature sampling (τ ∈ {2, ∞}) hurts performance across **every**
> domain, not just the dominant one. Even the smallest domain
> (chemistry, N = 46) loses 1.5 ChrF under uniform sampling. Volume
> dominates balance: under data-constrained fine-tuning, the model
> benefits more from the quantity of (any-domain) examples than from
> equalised per-domain proportions."

This is actually a more useful negative result than the original
hypothesised "trade-off" framing — it's a clean finding the paper can
make a contribution out of, and aligns with the existing
`acl_latex.tex` line 438 claim that "training on all available data,
despite natural domain imbalances, yields superior performance compared
to artificially balanced datasets" (which earlier in the draft was
based on a *truncation* experiment, not temperature sampling — now we
have the cleaner temperature-sampled evidence).

## §5 LoRA rank ablation

Llama-3.1-8B-Instruct has the full sweep (r4, r8, r16, r32, r64, r128) on
DCS for both eng↔African directions. r4 and r64 are the focus runs (vanilla
base); r8/r16/r32/r128 are imported (Unsloth 4-bit base — flag this in a
footnote).

A single 6×12 table with rank as rows and pairs as columns shows whether
the curve is monotone. Footnote: r8/r16/r32/r128 used a different base
(Unsloth 4-bit), so the absolute numbers are not directly comparable to
r4/r64 — what they tell you is the *shape* of the rank curve, not the
absolute level.

**Claim shape:** "Performance saturates at r ≈ 16-32 for most pairs;
beyond that, additional rank capacity offers <X COMET points improvement
and approaches noise." If saturation is around r=64 instead, say so —
that's the more interesting finding given r4 and r64 were the planned
"main focus" ranks.

## §6 Bilingual vs multilingual

Two parallel sub-experiments, one for seq2seq, one for LoRA:

- **Seq2seq (NLLB-1.3B):** per-pair fine-tune (`dcs/`) vs joint multilingual
  (`multilingual/<checkpoint>/`). 400k checkpoint is the comparable
  endpoint; 40k/80k/200k are training-curve points for the supplementary.
- **LoRA (llama3_8b):** per-pair (`finetune/r64/`) vs joint
  (`multilingual/r64/`). Caveat: mmt only covers `eng→African`, so the
  comparison is asymmetric and only for those 6 pairs.

**Claim shape:** "Joint multilingual training matches per-pair NLLB on
high-resource pairs (eng-hau, eng-zul) but loses on the very lowest-data
target (eng-amh). On LLMs, the joint LoRA is competitive with per-pair r64
on `eng→African` despite using a single adapter for all 6 pairs."

## §7 Open vs closed LLMs

GPT-4o and Gemini-1.5-Flash at default temperature (`closed/t0/`) versus
the best open LLMs (zero-shot, ICL-5shot). Add the t06 ablation in a
single subsection if the temperature delta is large enough to mention.

**Claim shape:** Closed APIs are competitive only at zero-shot; once you
allow ICL on the open side, parity is reached. Fine-tuning closes any
remaining gap on the African pairs.

## §8 Per-domain analysis (ties together §4 and the benchmark)

A grouped bar chart: domains on the x-axis ordered by N_d, COMET (averaged
over the 12 pairs) on the y-axis, three bars per domain (zero-shot,
fine-tune-DCS, balanced-τ=∞). Lets readers see the small-domain claim
without doing arithmetic on a table.

Single paragraph following the chart spelling out which domains are hardest
universally (chemistry will probably top the difficulty list across all
systems), and which domains see the largest fine-tuning gain.

## §9 Discussion / takeaways

Three crisp takeaways the paper is set up to support:
1. Scientific-text MT for African languages benefits substantially from
   in-domain fine-tuning even at small data scales (§2, §3, §8).
2. Joint multilingual training is a viable substitute for per-pair models
   when you have to ship one model, with a small cost on the lowest-data
   pair (§6).
3. Per-domain data composition matters: balanced (τ=∞) sampling helps
   very low-data domains but hurts the dominant one. Sqrt smoothing
   (τ=2) is the practical sweet spot if you need to pick one (§4, §8).

## §10 Threats to validity (footnote / appendix)

- Test set is drawn from the same DCS distribution; out-of-distribution
  generalization is not measured here.
- Closed-API runs (gpt4o, gemini_15_flash) used checkpoints frozen at the
  collection date; results may shift with subsequent model releases.
- LoRA rank ablation mixes vanilla and Unsloth 4-bit bases (§5
  footnote); we report the rank *shape* but caveat the absolute levels.
- Multilingual LoRA (mmt) trained `eng→African` only; reverse direction is
  not in this paper.
