# AfriScience-MT

Single-CLI machine-translation framework for African scientific text.
Harmonizes result schemas across all experiment types (zero-shot, seq2seq
fine-tune, LoRA fine-tune, LLM zero-shot / ICL / document-level), and
reproduces every table and figure in the accompanying paper from a single
aggregated CSV.

- **Code (this repo):** the `afriscience-mt` CLI, training/inference/eval pipelines, and the `afriscience_mt.paper` modules that rebuild every paper table and figure.
- **Data and predictions:** Hugging Face, [`dsfsi/afriscience_mt`](https://huggingface.co/datasets/dsfsi/afriscience_mt).
- **Paper:** to appear (citation forthcoming).
- **License:** Apache-2.0 (see [`LICENSE`](LICENSE)).

## Hugging Face dataset

The parallel corpus and every evaluation prediction live in one HF dataset
repo, exposed as two configurations:

```python
from datasets import load_dataset

# (1) The parallel scientific corpus -- English + six African targets
#     (amh, hau, lug, nso, yor, zul), 11 domains, sentence- and
#     document-aligned. Default configuration.
corpus = load_dataset("dsfsi/afriscience_mt", "corpus")
corpus["train"], corpus["dev"], corpus["test"]

# (2) Model outputs for every system we evaluate
#     (four seq2seq, seven open-weight LLMs, four closed models) across
#     zero-shot, in-context-learning, and document-level configurations,
#     plus the per-run aggregate metrics that drive every paper table.
preds = load_dataset("dsfsi/afriscience_mt", "predictions")
preds["outputs"]   # one row per (model, config, lang_pair, sentence_id)
preds["metrics"]   # one row per (model, config, lang_pair): chrF, COMET, BLEU
```

Every `outputs` row carries enough metadata to join back to the test split
of the corpus (`lang_pair`, `split`, `sentence_id`) and to filter by
`model_short`, `experiment_type`, `prompt_strategy`, `lora_rank`,
`temp_setting`, `dataset`, `is_ablation`. The `metrics` split is the
canonical source for `experiments/summary.csv`, which the
`afriscience-mt paper` builders read.

## Reproducing the paper

Rebuild any paper asset from the aggregated metrics:

```bash
afriscience-mt paper tables       # LaTeX tables (sentence, doc, ablations)
afriscience-mt paper plots        # paper plots
afriscience-mt paper doc-errors   # document-level accuracy-error pies
afriscience-mt paper balance      # domain-balance heatmap + delta-by-size
afriscience-mt paper figure4      # Figure 4
afriscience-mt paper assets       # rebuild everything
```

## Quickstart on the released dataset

End-to-end loop on the corpus published at [`dsfsi/afriscience_mt`](https://huggingface.co/datasets/dsfsi/afriscience_mt):

```bash
# 1. Install
pip install -e .

# 2. Pull the corpus from Hugging Face into the local cache used by the
#    training/eval pipelines.
python -c "from datasets import load_dataset; \
           load_dataset('dsfsi/afriscience_mt', 'corpus')"

# 3. Fine-tune NLLB-200 on (eng, hau) using the train/dev splits.
afriscience-mt train \
    --mode seq2seq \
    --model facebook/nllb-200-distilled-600M \
    --langs hau --directions eng2x \
    --num-epochs 5 --batch-size 16

# 4. Score every system you have under experiments/ on the test split.
afriscience-mt evaluate --mode finetuned

# 5. Aggregate per-run metrics into one CSV ready for the paper builders.
afriscience-mt aggregate --src experiments --out experiments/summary.csv

# 6. Regenerate paper-style tables and figures from that CSV.
afriscience-mt paper tables
afriscience-mt paper plots
```

For LLM evaluation (no fine-tuning), use `afriscience-mt evaluate --mode zero_shot` or the in-context-learning runners in `afriscience_mt/inference/`; the prompt templates are configurable per language pair.

## Using your own dataset

The same pipelines work on any parallel corpus formatted as one CSV per source document (the format used by `data/raw_files/`):

```
Paper#, Seg.#, Source, <Target1>, <Target2>, ...
1, 1, "English sentence ...", "Translation 1 ...", "Translation 2 ...", ...
1, 2, ...
```

- `Paper#` is a document identifier (used for paper-level train/dev/test splits).
- `Seg.#` is the sentence index within the document.
- `Source` is the source-language sentence (English in our setup; can be any language).
- One column per target language; the column header is the language name and is mapped to its ISO 639-3 code via [`afriscience_mt/utils/language_utils.py`](afriscience_mt/utils/language_utils.py).

Point the pipelines at your own data with the path environment variable:

```bash
export DCSMT_DATA_DIR=/path/to/your/data        # holds raw_files/, glossary/
export DCSMT_EXPERIMENTS_DIR=/path/to/runs      # where results.json/predictions.json land

# Train on your CSVs the same way:
afriscience-mt train --mode seq2seq \
    --model facebook/nllb-200-distilled-600M \
    --langs <your-targets> --directions both

# Evaluate, aggregate, and build paper-style tables on your own runs:
afriscience-mt evaluate --mode finetuned
afriscience-mt aggregate --src $DCSMT_EXPERIMENTS_DIR
afriscience-mt paper tables
```

If your domain doesn't match the 11 scientific domains used here, that's fine -- the splits are made at the *document* level so the framework only cares about `Paper#` and `Seg.#`. Glossaries are optional; drop TSVs into `<your-data-dir>/glossary/<lang>.tsv` (`eng\t<target>` per row) to enable terminology-aware evaluation.

## CLI reference

```bash
afriscience-mt train      --mode {seq2seq|lora|multi|mafand}    # train MT model
afriscience-mt evaluate   --mode {zero_shot|finetuned|rescore}  # eval / re-score
afriscience-mt aggregate  --src experiments --out experiments/summary.csv
afriscience-mt table      --metric ssa_comet --layout model_x_pair --bold-best --format tex
                          # --ablation       show only ablation runs
                          # --include-both   combine main + ablation runs
afriscience-mt plot       --kind heatmap --metric ssa_comet --out figs/zs_heat.pdf
afriscience-mt paper <target>   # tables | plots | doc-errors | balance | figure4 | assets
afriscience-mt migrate    --src OLD --dst NEW                   # one-time
```

Run any subcommand with `--help` for full options.

## Repository layout

```
AfriScience-MT/
├── afriscience_mt/              python package + CLI (afriscience-mt)
│   ├── commands/                CLI subcommands (train, evaluate, paper, ...)
│   ├── inference/               vLLM / API clients, ICL prompt builders
│   ├── training/                seq2seq + LoRA + multilingual trainers
│   ├── evaluation/              metrics (BLEU, chrF, SSA-COMET) + LLM judges
│   ├── paper/                   paper-asset builders (tables, plots, pies)
│   └── ...
├── scripts/                     one-off and orchestration scripts
├── configs/                     YAML configs (models, prompts, evaluation)
├── templates/                   ICL prompt templates
├── data/                        glossaries + raw parallel CSVs (corpus and
│                                predictions live on HF, not here)
├── docs/EXPERIMENTS.md          experiment-layout reference
└── pyproject.toml
```

Local development outputs that **stay on disk and are git-ignored**:
`checkpoints/`, `runs/`, `wandb/`, `outputs/`, `*.bin` / `*.safetensors` / `*.pt`, `hf_cache/`, `experiments/` (predictions and results), `data/processed_files/` (regenerable).

## Canonical result schema

Every experiment writes one `results.json` under
`experiments/<experiment_type>/<model_short>/[<dataset>/][<rN>/][<strategy>/]<src-tgt>/`,
e.g. `experiments/seq2seq_finetune/nllb_200_distilled_600m/mafand/eng-hau/`.

Fields:

```json
{
  "schema_version": "1.0",
  "experiment_id": "...",
  "experiment_type": "zero_shot | seq2seq_finetune | lora_finetune | llm_zero_shot | llm_icl | llm_doc",
  "timestamp": "ISO8601",
  "model": {"name": "...", "type": "seq2seq | llm | lora", "short_name": "...", "base_model": "..."},
  "source_lang": "eng", "target_lang": "hau", "lang_pair": "eng-hau",
  "split": "test", "dataset": "dcs | mafand | mafand_pp",
  "config": { ... hyperparams or prompt config ... },
  "metrics": {"bleu": ..., "chrf": ..., "ssa_comet": ..., "num_samples": ...},
  "val_metrics": { ... },        // optional
  "train_metrics": { ... },      // optional
  "metadata": {"source": "new | migrated", "comet_metric": "ssa-comet", ...},
  "environment": { ... }, "git": { ... }
}
```

`predictions.json` (sibling) is a `PredictionFile`: list of
`{src, ref, hyp, doc_id?, domain?}`.

## Setup

Requires Python ≥ 3.9.

```bash
git clone https://github.com/dsfsi/afriscience_mt.git
cd afriscience_mt
pip install -e .
# Or, if dependencies are already satisfied in your env:
pip install -e . --no-deps
```

After install, the CLI is available as `afriscience-mt` (and equivalently
as `python -m afriscience_mt`).

Environment variable overrides (optional):
- `DCSMT_DATA_DIR` — default `./data`
- `DCSMT_EXPERIMENTS_DIR` — default `./experiments`
- `DCSMT_HF_CACHE` — default `./hf_cache` (set this when running training so
  HF downloads land outside the repo and stay out of git)

When training, set the HF cache to a path outside the repo, e.g.:

```bash
export DCSMT_HF_CACHE=/ext_data/idris/DCS-MT-Updated/hf_cache
afriscience-mt train --mode mafand --setup all --langs hau lug yor zul
```

The new training script writes intermediate checkpoints to `./checkpoints/`
(git-ignored) and deletes them after evaluation unless you pass
`--keep-checkpoints`. Final results.json and predictions.json land under
`experiments/seq2seq_finetune/<model>/<setup>/<pair>/` and DO get committed.

## Citation

```bibtex
@misc{afriscience-mt-2026,
  title  = {AfriScience-MT: Towards Decolonizing Science in Africa through Text Translation},
  author = {Idris Abdulmumin and Tajuddeen Gwadabe and Shamsuddeen Hassan Muhammad and David Ifeoluwa Adelani and Nomonde Khalo and Ibrahim Said Ahmad and Abiodun Modupe and Anina Mumm and Sibusiso Biyela and Michelle Rabie and Johanna Havemann and Marek Rei and Jade Abbott and Vukosi Marivate},
  year   = {2026}
}
```