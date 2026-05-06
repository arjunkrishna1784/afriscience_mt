# Experiments — paper-writing reference

This document is the single source of truth for what's in `experiments/`,
where each piece came from, and how the metrics were computed. Refresh
counts with `afriscience-mt aggregate` and check `experiments/summary.csv`
when you need numbers in a table.

## Languages, splits, metrics

- **Languages.** `eng` + 6 African: `amh, hau, lug, nso, yor, zul`. 12 directional pairs (eng↔X for each X).
- **Test split.** `data/processed_files/test/<lang>.tsv` — 970 sentences per language, columns `paper, segment, category, text`. Position-aligned: `predictions.json` row *i* corresponds to `test/<lang>.tsv` row *i* (text column).
- **Train split.** `data/processed_files/train/<lang>.tsv` — 5 792 sentences, 177 papers, 11 domains. **Highly imbalanced** (57×: health 2 605 vs chemistry 46). Domain breakdown:

  | domain | sents | papers | % |
  |---|---|---|---|
  | chemistry | 46 | 2 | 0.8% |
  | Indigenous knowledge | 92 | 3 | 1.6% |
  | engineering | 93 | 3 | 1.6% |
  | statistics | 184 | 6 | 3.2% |
  | computer science | 213 | 7 | 3.7% |
  | geography | 218 | 8 | 3.8% |
  | agriculture | 388 | 12 | 6.7% |
  | biology | 632 | 19 | 10.9% |
  | biochemistry | 651 | 20 | 11.2% |
  | sociology | 670 | 21 | 11.6% |
  | health | 2 605 | 76 | 45.0% |

- **Metrics.** Three metrics computed on every prediction set, **but reporting hierarchy is strict:**
  - **SSA-COMET (primary)** via `McGill-NLP/ssa-comet-mtl`. Headline tables, main figures, and any single-metric claim use SSA-COMET. The model is trained specifically for sub-Saharan African languages; this is what we trust most.
  - **chrF (secondary, ablation)** via `sacrebleu` (chrF++ with `word_order=2`). Used for two-metric tables and as a confirmation/cross-check on COMET. Robust on morphologically rich African languages where BLEU's word-level matching is harsh.
  - **BLEU (tertiary)** via `sacrebleu` (default tokeniser `13a`). Reported only when absolutely necessary — e.g., direct comparison with a prior paper that quotes BLEU, or in the supplementary. Do not lead with BLEU in the main results.

  Recompute any of them with `afriscience-mt evaluate --mode rescore --src <dir> --metric {bleu|chrf|ssa_comet}` — it reads `predictions.json`, looks up `src`/`ref` from the test split by index, scores, and updates the sibling `results.json`. The default for `afriscience-mt table` is `--metric ssa_comet`, and the `pair_x_metric` layout lists the three in `ssa_comet, chrf, bleu` order.

## Layout (arch-first, ablation-flagged)

```
experiments/
├── seq2seq/
│   ├── zero_shot/<model>/<pair>/
│   ├── finetune/<model>/<dataset>/<pair>/        # dataset: dcs | mafand | mafand_pp
│   └── multilingual/<model>/<checkpoint>/<pair>/ # NLLB N-to-N fairseq run
└── llm/
    ├── open/
    │   ├── zero_shot/<model>/<pair>/
    │   ├── icl/<model>/<strategy>/<pair>/        # 0shot | 5shot | 5shot_semantic
    │   ├── doc/<model>/<strategy>/<pair>/        # 0shot | 1shot
    │   ├── finetune/<model>/r<N>/<pair>/         # per-pair LoRA
    │   └── multilingual/<model>/r<N>/<pair>/     # joint multilingual LoRA (mmt)
    └── closed/
        ├── t0/{zero_shot,icl,doc}/<model>/.../<pair>/    # default temperature: "actual"
        └── t06/{zero_shot,icl,doc}/<model>/.../<pair>/   # temperature 0.6: ablation
```

**Hyp-only predictions.** `predictions.json` contains `samples: [{hyp: "..."}, ...]` only — no `src` or `ref`. These come from `data/processed_files/test/<lang>.tsv` via index lookup at rescore time.

## What counts as ablation

`metadata.is_ablation = true` is set in every `results.json` for runs in any of these buckets. `aggregate` writes it as a column in `summary.csv`; `afriscience-mt table` excludes them by default and accepts `--ablation` (only) or `--include-both` (combined).

- **`dataset == "mafand_pp"`** — DCS-augmented MAFAND fine-tune (was `dcs_mafand`); ablates against the MAFAND baseline.
- **`llm/closed/t06/...`** — closed-LLM (gpt4o, gemini_15_flash) at temperature 0.6; ablates default-temperature `t0`.
- **All `seq2seq/multilingual/...`** — joint multilingual NLLB. 400k checkpoint is the natural "main" run; promote it later if you want.
- **All `llm/open/multilingual/...`** — joint multilingual LoRA on `Meta-Llama-3.1-8B-Instruct` (mmt). r64 is the natural "main" rank.
- **All `llm/open/finetune/llama3_8b/r{8,16,32,128}/...`** — non-`{r4,r64}` LoRA ranks (rank ablation).

## Inventory (post-2026-05-06, 1 000 rows in `summary.csv`)

The active experiment matrix:

### seq2seq

| setting | model | dataset / config | pairs | comet ✓ | source |
|---|---|---|---|---|---|
| zero_shot | m2m100_418m | — | 30 | ✓ | original repo |
| zero_shot | m2m100_1_2b | — | 30 | ✓ | original repo |
| zero_shot | nllb_200_distilled_600m | — | 30 | ✓ | original repo |
| zero_shot | nllb_200_1_3b | — | 30 | ✓ | original repo |
| finetune | m2m100_418m | dcs | 12 | ✓ | original |
| finetune | m2m100_1_2b | dcs | 12 | ✓ | original |
| finetune | nllb_200_distilled_600m | dcs | 12 | ✓ | original |
| finetune | nllb_200_1_3b | dcs | 12 | ✓ | original |
| finetune | nllb_200_1_3b | mafand | 8 | ✓ | original |
| finetune | nllb_200_1_3b | mafand_pp | 8 | ✓ ablation | original |
| multilingual | nllb_200_distilled_600m | 40k / 80k / 200k / 400k updates | 4 × 12 = 48 | ✓ all-ablation | imported from `DCS-MT-Old/results/nllb/multilingual/` (was raw fairseq output; predictions extracted from `processed/<src>_<tgt>.txt`, metrics recomputed from canonical refs) |

### llm — open

| setting | model | config | pairs | comet ✓ | source |
|---|---|---|---|---|---|
| zero_shot | 7 open LLMs | — | 12 each = 84 | ✓ | original |
| icl | 4 open LLMs | 5shot | 12 each = 48 | ✓ | original |
| icl | gpt4o-style 5shot_semantic for some | — | … | ✓ | original |
| icl | afriquellama_8b, afriqueqwen_8b, tiny_aya_global | 0shot | 12 each = 36 | ✓ | **predictions recovered** from `github.com/abumafrim/AfriScience-MT@5e72e4f` (under `experiments/llm-old/icl_0shot/<model>/<pair>/predictions.json`); originals were missing predictions.json so SSA-COMET was un-recomputable until restore. BLEU/chrF in original results matched the recomputed values. |
| doc | 7 open LLMs | 0shot, 1shot | 12 × 2 = 24 each | ✓ | original |
| finetune | llama3_8b | r4, r64 | 12 each = 24 | ✓ | original (per-pair LoRA, vanilla `meta-llama/Llama-3.1-8B-Instruct`) |
| finetune | other 5 open LLMs | r4, r64 | 12 each = 60 | ✓ | original (per-pair LoRA) |
| finetune | llama3_8b | r8, r16, r32, r128 | 12 each = 48 (ablation) | ✓ | **imported** from `DCS-MT-Updated/old_experiments/lora/Meta-Llama-3.1-8B-Instruct/rank-{8,16,32,128}/`. Base = `unsloth/meta-llama-3.1-8b-instruct-bnb-4bit` (4-bit Unsloth, **not** vanilla — explains why some imported COMETs run higher than r4/r64 vanilla). xx-eng pairs trained 10 epochs (`predictions_10e.txt`); eng-xx pairs trained 3 epochs (`predictions.txt`). All ablation. |
| multilingual | llama3_8b | r8, r16, r32, r64, r128 (mmt) × eng-X only | 5 × 6 = 30 | ✓ all-ablation | **imported** from `DCS-MT-Old/models/unsloth/lora/Meta-Llama-3.1-8B-Instruct/rank-{N}/mmt/lora_model/`. Single LoRA adapter trained jointly on all eng→african pairs (no xx-eng direction). |

### llm — closed

| setting | model | t0 (actual) | t06 (ablation) | comet ✓ |
|---|---|---|---|---|
| zero_shot | gpt4o | 12 | 12 | ✓ |
| zero_shot | gemini_15_flash | 12 | 12 (incl. previously-duplicated `__alt` removed) | ✓ |
| icl | gpt4o | 12 (5shot, 5shot_semantic) | 12 (5shot) | ✓ |
| icl | gemini_15_flash | 12 (5shot, 5shot_semantic) | 12 (5shot) | ✓ |
| doc | gpt4o, gemini_15_flash | 12 (0shot, 1shot) each | – | ✓ |

## Schema additions in `results.json`

The canonical fields are documented in the top-level README. This session
added the following metadata-only conventions; aggregator and table CLI
already pick them up.

| field | type | when set |
|---|---|---|
| `metadata.is_ablation` | bool | true for all rows in the buckets listed above |
| `metadata.temp_setting` | "t0" / "t06" | closed-LLM only |
| `metadata.original_path` | string | source path before the restructure (chains via `original_path_chain` if multiple migrations) |
| `metadata.restructured_to` | string | new path after the 2026-04 restructure |
| `metadata.source` | string | `migrated`, `imported_from_old_experiments`, `imported_from_dcs_mt_old_multilingual`, `imported_from_dcs_mt_old_mmt`, etc. |
| `metadata.import_note` | string | free-text caveat (e.g. epoch mismatches, alt naming, base-model differences) |
| `metadata.rescored` | list of `{metric, value, timestamp}` | appended each time `evaluate --mode rescore` runs |

## Changes from the original DCS-MT layout

| Renamed / moved | From | To |
|---|---|---|
| dataset | `dcs_mafand` | `mafand_pp` |
| closed-LLM models | `gpt4o_t06`, `gemini_15_flash_t06`, `gemini_15_flash_temp_0_6` | normalized to canonical `gpt4o`/`gemini_15_flash` under `closed/{t0,t06}/`; the `_temp_0_6` variant was a duplicate of `_t06` (identical metrics, different timestamps) and was removed |
| ICL strategies | `icl_0shot`, `icl_5shot`, `icl_5shot_semantic` | strategy subdir under `llm/.../icl/<model>/<strategy>/<pair>/` as `0shot` / `5shot` / `5shot_semantic` |
| Doc strategies | `doc_0shot`, `doc_1shot`, `doc_0shot_legacy`, `doc_1shot_legacy` | normalized to `0shot` / `1shot` (`_legacy` suffix dropped — those were the only doc runs for gpt4o and gemini_15_flash; not duplicates) |
| Mafand 600M stub | `seq2seq_finetune/nllb_200_distilled_600m/mafand/eng-hau/` | **removed** — single-pair stub, doesn't represent a real comparison |

## Known caveats for paper-writing

These are worth a footnote in the paper or supplementary material:

1. **Imported llama3_8b LoRA r{8,16,32,128} use a different base.** They were trained with `unsloth/meta-llama-3.1-8b-instruct-bnb-4bit` (Unsloth's 4-bit quantised stack), while the original r4 and r64 used vanilla `meta-llama/Llama-3.1-8B-Instruct`. This is the main reason imported COMETs run higher than r4/r64 vanilla on the same pairs. Recorded in `config.base_model` per-row.
2. **Imported llama3_8b `xx-eng` pairs are 10-epoch** (only `predictions_10e.txt` was available), while `eng-xx` pairs are 3-epoch. Recorded in `config.num_epochs`.
3. **mmt LoRA is eng→african only.** No xx-eng direction was trained for the multilingual setup, so the comparison against per-pair LoRAs is asymmetric.
4. **Multilingual NLLB checkpoints** are training-update counts (40k / 80k / 200k / 400k), not epochs. They are different snapshots of the same training run; 400k is the natural main result to promote.
5. **`afriquellama_8b/0shot`, `afriqueqwen_8b/0shot`, `tiny_aya_global/0shot`** ICL predictions were recovered from a separate older repo (`AfriScience-MT@5e72e4f`); they had not been in `DCS-MT-Updated`. SSA-COMET was recomputed; BLEU and chrF in our results match the originally-reported values.
6. **`gpt4o/eng-{amh,hau,lug,nso,yor,zul}`** had broken metadata before this session (model.name was the literal string `"unknown"`, metrics were nested in `automatic_metrics` instead of top-level). Repaired in place — model.name set to `gpt4o`, metrics hoisted to canonical location, experiment_id renamed.
7. **The `restructure` from `experiments/llm_zero_shot/...` to `experiments/llm/{open,closed}/...`** preserved every file; original paths are recorded in `metadata.original_path` and `metadata.original_path_chain` for traceability.

## Planned: per-domain balance ablation (NLLB-1.3B)

**Question.** Do low-data domains benefit from being trained alongside high-data domains, and what is the effect (positive or negative) of balanced representation?

### Why temperature sampling and not hard truncation

A previous internal script (`DCS-MT-Old/scripts/test.ipynb` cell 38) picks 1–3 papers per domain to match the smallest domain's count. Applied to our train split this *truncates* the corpus from **5 792 → ≈ 506 sentences (a 91% reduction)** before any model sees it. We chose temperature sampling instead, for four reasons that are each strong enough to mention in the paper:

1. **Truncation conflates the hypothesis with a confound.** The research question is whether *re-weighting* domains helps. Truncating to a uniform-but-tiny corpus changes both the per-domain weighting **and** the total training volume. A drop in BLEU under truncation cannot be attributed to balance — it could just as well be data starvation. Temperature sampling keeps every example available; only the per-example sampling probability changes, so the comparison isolates the balance variable.

2. **NLLB-1.3B cannot be meaningfully fine-tuned on 506 sentences.** Empirically on this corpus, a fine-tune at that scale will under-train against a 1.3 B-parameter base model and the result will look uniformly bad across domains. That is not the per-domain balance question we set out to answer.

3. **Temperature sampling is the established multilingual-NMT methodology.** NLLB (Costa-jussà et al., 2022) trains with temperature τ ≈ 5; M2M-100 (Fan et al., 2021) uses an analogous temperature-based smoothing. Picking the same family of methods aligns the experimental design with the literature reviewers will compare against and avoids defending a strawman setting.

4. **It exposes a continuous trade-off, not a binary split.** Sweeping τ ∈ {1, 2, ∞} produces three points along the proportional-to-uniform axis. Reporting the per-domain BLEU as τ moves from natural to fully balanced is a richer story (and a more useful figure) than a single "balanced vs not" comparison would yield.

Implementation note: weights are `w_i = (1/N_d)^(1/τ)` per example and consumed via `torch.utils.data.WeightedRandomSampler`. Total optimizer steps are held constant across τ values (same `num_samples = len(dataset)` per epoch, same number of epochs), so the only thing changing between conditions is the sampling distribution. The `multilingual_comparison.yaml` config in `DCS-MT-Updated` already names this strategy (`temperature_sampling: 5.0`), so we are aligning with prior in-house conventions as well.

**Settings (3-way comparison, all 12 lang pairs, all 11 domains).**

| τ | per-example weight | name | role |
|---|---|---|---|
| 1 | proportional (`P(d) ∝ N_d`) | `proportional` | natural distribution — existing `seq2seq/finetune/nllb_200_1_3b/dcs/` |
| 2 | sqrt smoothing (`P(d) ∝ √N_d`) | `sqrt` | moderate balance |
| ∞ | uniform (`P(d) = 1/11`) | `uniform` | full balance via `WeightedRandomSampler` upsampling (no truncation) |

**Implementation.** Drop-in `torch.utils.data.WeightedRandomSampler` with `weights[i] = (1/N_d)^(1/τ)` per example. Same total optimizer steps across τ to isolate the sampling effect. Same hyperparams otherwise (5 epochs equivalent budget, batch 8, FP16).

**Per-domain reporting comes free.** Same eval predictions, group by the test set's `category` column. So one fine-tune per pair → per-domain BLEU/chrF/SSA-COMET breakdown.

**How this answers the question.**
- *Cross-domain transfer:* compare τ=1 (which contains 45% health) to zero-shot per-domain on small domains. If τ=1 ≫ zero-shot on chemistry/Indigenous/engineering despite their tiny share of train, the gain is cross-domain transfer.
- *Effect of balance:* τ=1 vs τ=2 vs τ=∞ per-domain. If τ→∞ recovers small-domain performance at a cost on health, the trade-off is visible. If it loses everywhere, volume dominated balance.

**Storage.** Outputs land at `experiments/seq2seq/finetune/nllb_200_1_3b/balance_t{1,2,inf}/<pair>/`. All runs flagged `metadata.is_ablation = true` (τ=1 IS the existing `dcs/` non-ablation baseline; we don't re-train it).

**Compute.** ≈ 12 fine-tunes × 2 new τ values = 24 NLLB-1.3B runs. ~10–15 min/run on a single A6000; GPU 1 idle (48 GB), so 1–3 h wall time running pairs in parallel.

## Commands you'll want during paper-writing

```bash
# refresh CSV after any change
afriscience-mt aggregate --src experiments --out experiments/summary.csv

# main-results table (excludes ablation rows by default)
afriscience-mt table --metric ssa_comet --layout model_x_pair --bold-best --format tex

# ablation only
afriscience-mt table --ablation --metric ssa_comet --layout model_x_pair --format tex

# dataset comparison (mafand vs mafand_pp), needs --include-both
afriscience-mt table --filter-type seq2seq_finetune \
                     --filter-model nllb_200_1_3b \
                     --layout dataset_compare --metric chrf \
                     --include-both --format tex --out tables/mafand_compare.tex

# re-score with a different metric without re-running inference
afriscience-mt evaluate --mode rescore --src experiments/<scope> --metric ssa_comet
```
