---
license: apache-2.0
language:
  - eng
  - amh
  - hau
  - lug
  - nso
  - yor
  - zul
task_categories:
  - translation
tags:
  - machine-translation
  - african-languages
  - scientific-translation
  - low-resource
  - parallel-corpus
pretty_name: AfriScience-MT
size_categories:
  - 1K<n<10K
configs:
  - config_name: corpus
    default: true
    data_files:
      - split: train
        path: corpus/train-*
      - split: dev
        path: corpus/dev-*
      - split: test
        path: corpus/test-*
  - config_name: predictions
    data_files:
      - split: outputs
        path: predictions/outputs-*
  - config_name: metrics
    data_files:
      - split: summary
        path: predictions/metrics-*
  - config_name: glossary
    data_files:
      - split: terms
        path: glossary/terms-*
---

# AfriScience-MT

A parallel scientific machine-translation corpus for **English + six African languages** (Amharic, Hausa, Luganda, Northern Sotho, Yorùbá, isiZulu), co-developed with expert science communicators and professional translators across **11 scientific domains** (Agriculture, Biochemistry, Biology, Chemistry, Computer Science, Engineering, Geography, Health, Indigenous Knowledge, Sociology, Statistics).

Alongside the corpus we release **every model prediction and per-run metric** behind the accompanying paper, so the benchmark can be reproduced and extended without re-running any experiments.

- **Source code:** the `afriscience-mt` CLI at https://github.com/dsfsi/afriscience_mt.
- **License:** Apache-2.0.
- **Paper:** [AfriScience-MT: Towards Decolonizing Science in Africa through Text Translation](https://arxiv.org/abs/2605.29741) (arXiv:2605.29741).

## Configurations

This repo exposes four configurations: `corpus` (default), `predictions`, `metrics`, and `glossary`.

### `corpus` (default)
The parallel scientific corpus. 230 papers, 7,605 English source sentences, each translated into the six African target languages and aligned at both the sentence and document level. Splits are made at the *document* level within each domain so every split contains documents from every domain and no within-paper sentences leak across splits.

Fields per row:

| field | type | description |
|---|---|---|
| `paper_id` | string | source paper identifier |
| `domain` | string | one of 11 scientific domains |
| `sentence_id` | int | position within the paper (0-indexed) |
| `lang_pair` | string | e.g. `eng-amh` |
| `source_lang` | string | ISO 639-3 |
| `target_lang` | string | ISO 639-3 |
| `source` | string | source sentence (English) |
| `target` | string | target translation (African language) |

Splits: `train` (5,792 sentences / 177 papers) · `dev` (843 / 25) · `test` (970 / 28).

### `predictions`
Per-sentence model outputs for every system evaluated in the paper: four seq2seq models (M2M100-418M/1.2B, NLLB-600M/1.3B), seven open-weight LLMs (Llama3-8B, Gemma2-9B-IT, AfriqueLlama-8B, AfriqueQwen-8B, Tiny-Aya-Earth, Tiny-Aya-Global, TranslateGemma-12B), and four closed models (GPT-4o, Gemini-1.5-Flash, GPT-5.4, Gemini-3.1-Flash-Lite), across zero-shot, in-context-learning, and document-level configurations. One row per (model, configuration, language pair, sentence). Carries enough metadata to join back to the test split of the `corpus` config via (`lang_pair`, `split`, `sentence_id`).

Single split: `outputs` (1,982,558 rows).

| field | type | description |
|---|---|---|
| `model_short` | string | e.g. `nllb_200_1_3b`, `gpt54`, `gemini_31_flash_lite` |
| `experiment_type` | string | `seq2seq_zero_shot`, `seq2seq_finetune`, `llm_zero_shot`, `llm_icl`, `llm_doc`, `lora_finetune`, … |
| `prompt_strategy` | string | e.g. `template5_10shot`, `doc_0shot`, `semantic_10shot` (null for fine-tuned) |
| `lora_rank` | int | LoRA rank, or null |
| `temp_setting` | string | `t0` (default) or `t06` (temperature ablation) |
| `dataset` | string | `dcs` (AfriScience-MT), `mafand`, `mafand_pp` |
| `lang_pair` | string | e.g. `eng-amh`, `amh-eng` |
| `source_lang` | string | ISO 639-3 |
| `target_lang` | string | ISO 639-3 |
| `split` | string | `test` (default) |
| `sentence_id` | int | position within the test set; joins to `corpus[test]` |
| `source` | string | source sentence |
| `reference` | string | reference translation |
| `prediction` | string | model output |
| `is_ablation` | bool | excluded from the main results when true |

### `metrics`
Per-run aggregated metrics: one row per (model, configuration, language pair). This is the canonical source for `experiments/summary.csv`, which the paper builders read.

| field | type | description |
|---|---|---|
| `model_short` / `experiment_type` / `prompt_strategy` / `lora_rank` / `temp_setting` / `dataset` / `lang_pair` / `split` / `is_ablation` | — | same join keys as `predictions` |
| `bleu` | float | BLEU |
| `chrf` | float | chrF |
| `ssa_comet` | float | SSA-COMET (the paper's primary metric) |
| `num_samples` | int | sentences scored |
| `val_bleu` / `val_chrf` / `val_ssa_comet` | float | validation-split metrics where available |

Single split: `summary`.

### `glossary`
The co-developed bilingual scientific glossaries built during translation, one row per (English term, target-language translation) pair, stacked across all six target languages.

| field | type | description |
|---|---|---|
| `target_lang` | string | ISO 639-3 of the African target language |
| `eng` | string | English scientific term |
| `target` | string | translation in the target language |

Single split: `terms`. Counts per language: Amharic 333 · Hausa 130 · Luganda 181 · Northern Sotho 263 · Yorùbá 587 · isiZulu 385. The largest glossaries (Yorùbá, isiZulu) correspond to the languages for which standardised scientific terminology was scarcest.

## Loading

```python
from datasets import load_dataset

# Parallel corpus, default config.
corpus = load_dataset("dsfsi/afriscience_mt", "corpus")
corpus["train"][0]
# {'paper_id': '...', 'domain': 'Health', 'sentence_id': 0,
#  'lang_pair': 'eng-amh', 'source_lang': 'eng', 'target_lang': 'amh',
#  'source': '...', 'target': '...'}

# Per-sentence model outputs.
preds = load_dataset("dsfsi/afriscience_mt", "predictions")
preds["outputs"][0]   # one model output per row

# Per-run aggregated metrics.
metrics = load_dataset("dsfsi/afriscience_mt", "metrics")
metrics["summary"][0] # one (model, config, lang_pair) row

# Bilingual scientific glossaries.
gloss = load_dataset("dsfsi/afriscience_mt", "glossary")
gloss["terms"][0]
# {'target_lang': 'yor', 'eng': '...', 'target': '...'}
```

Common joins:

```python
import pandas as pd
test_corpus = corpus["test"].to_pandas()
outputs     = preds["outputs"].to_pandas()
metrics_df  = metrics["summary"].to_pandas()

# Predictions paired with the source/reference from the corpus test split.
joined = outputs.merge(
    test_corpus[["lang_pair", "sentence_id", "domain", "paper_id"]],
    on=["lang_pair", "sentence_id"], how="left",
)

# All NLLB-1.3B fine-tuned scores on the AfriScience-MT (dcs) test split.
metrics.query(
    "model_short == 'nllb_200_1_3b' and "
    "experiment_type == 'seq2seq_finetune' and "
    "dataset == 'dcs' and not is_ablation"
)
```

## Languages

| code | language | family | region |
|---|---|---|---|
| `eng` | English | Germanic | — |
| `amh` | Amharic | Afro-Asiatic | East Africa |
| `hau` | Hausa | Afro-Asiatic | West Africa |
| `lug` | Luganda | Niger-Congo | East Africa |
| `nso` | Northern Sotho (Pedi) | Niger-Congo | Southern Africa |
| `yor` | Yorùbá | Niger-Congo | West Africa |
| `zul` | isiZulu | Niger-Congo | Southern Africa |

Collectively, these six target languages are spoken by over 200 million people across 15 countries.

## Construction

The corpus follows a two-stage process: a domain-expert science communicator first produces a 250–350-word lay summary of each paper (preserving key contributions and scientific terminology), and a professional translator with per-language lead-translator review then renders the summary, abstract, terminology, and definitions into each target language. Bilingual scientific glossaries are co-developed during translation to fill gaps where no standardised terminology existed.

Full preprocessing details (untranslated-sentence removal, single-word filtering, sentence-level deduplication, document-level preservation, glossary normalisation) are documented in the paper appendix.

## Citation

```bibtex
@misc{abdulmumin2026afriscience,
  title         = {AfriScience-MT: Towards Decolonizing Science in Africa through Text Translation},
  author        = {Idris Abdulmumin and Tajuddeen Gwadabe and Shamsuddeen Hassan Muhammad and David Ifeoluwa Adelani and Nomonde Khalo and Ibrahim Said Ahmad and Abiodun Modupe and Anina Mumm and Sibusiso Biyela and Michelle Rabie and Johanna Havemann and Marek Rei and Jade Abbott and Vukosi Marivate},
  year          = {2026},
  eprint        = {2605.29741},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2605.29741}
}
```

## License

Released under the Apache License 2.0. See [`LICENSE`](LICENSE).
