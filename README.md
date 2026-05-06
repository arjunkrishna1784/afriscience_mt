# AfriScience-MT

Single-CLI machine-translation framework for African scientific text.
Harmonizes result schemas across all experiment types (zero-shot, seq2seq
fine-tune, LoRA fine-tune, LLM zero-shot / ICL / document-level).

## Layout

```
AfriScience-MT-Latest/
├── afriscience_mt/             python package (CLI: afriscience-mt)
├── scripts/
├── experiments/
│   ├── seq2seq/
│   │   ├── zero_shot/<model>/<pair>/
│   │   └── finetune/<model>/<dataset>/<pair>/    # dataset: dcs | mafand | mafand_pp
│   └── llm/
│       ├── open/
│       │   ├── zero_shot/<model>/<pair>/
│       │   ├── icl/<model>/<strategy>/<pair>/    # strategy: 0shot | 5shot | 5shot_semantic
│       │   ├── doc/<model>/<strategy>/<pair>/    # strategy: 0shot | 1shot
│       │   └── finetune/<model>/r<N>/<pair>/     # LoRA ranks
│       └── closed/
│           ├── t0/                               # default-temperature: "actual"
│           │   ├── zero_shot/<model>/<pair>/
│           │   ├── icl/<model>/<strategy>/<pair>/
│           │   └── doc/<model>/<strategy>/<pair>/
│           └── t06/                              # temperature 0.6: ablation
│               └── (same shape as t0)
└── data/                       processed_files, glossary, raw_files

# Ablation runs are flagged in each results.json via metadata.is_ablation = true:
#   - dataset == "mafand_pp"   (MAFAND++ DCS-augmented finetune)
#   - closed-LLM with t06      (temperature 0.6)
# `afriscience-mt table` excludes these by default; pass --ablation for ablation
# only or --include-both for combined.

# Model files are NEVER shipped. .gitignore excludes:
#   - checkpoints/, runs/, wandb/, outputs/  (local training output)
#   - *.bin / *.safetensors / *.pt / *.gguf  (any weight format)
#   - hf_cache/, adapter/                    (HF cache, adapter weight dirs)
#   - experiments_legacy/                    (pre-migration backup; safe to delete
#                                             once you've verified the migration)
# Only code, configs, predictions, results, and data ship.
```

## CLI

```bash
afriscience-mt train      --mode {mafand|seq2seq|lora|multi}    # train
afriscience-mt evaluate   --mode {zero_shot|finetuned|rescore}  # eval / re-score
afriscience-mt aggregate  --src experiments --out experiments/summary.csv
afriscience-mt table      --metric ssa_comet --layout model_x_pair --bold-best --format tex
                          # add --ablation to show only ablation runs
                          # add --include-both to combine actual + ablation
afriscience-mt plot       --kind heatmap --metric ssa_comet --out figs/zs_heat.pdf
afriscience-mt migrate    --src OLD --dst NEW                   # one-time
```

Run any subcommand with `--help` for details.

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

## Filling the MAFAND gap

The single-script entry that completes the "with vs without AfriScience"
ablation is:

```bash
# Full comparison: 3 setups × 4 langs × 2 directions × 1 model = 24 runs
python scripts/train_mafand_comparison.py \
    --model facebook/nllb-200-distilled-600M \
    --setup all --langs hau lug yor zul --directions both \
    --num-epochs 5 --batch-size 16
```

After that, regenerate the comparison table (uses `--include-both` so the
mafand baseline and the mafand_pp ablation appear side by side):

```bash
afriscience-mt aggregate --src experiments
afriscience-mt table --filter-type seq2seq_finetune \
            --filter-model nllb_200_distilled_600m \
            --layout dataset_compare --metric chrf \
            --include-both \
            --format tex --bold-best --out tables/mafand_compare.tex
```

## Migration from legacy

The migration is already complete; `experiments_legacy/` holds the old layout
for reference. To re-run:

```bash
afriscience-mt migrate --src experiments_legacy --dst experiments
```

The migrator fans out aggregated zero-shot files, preserves LoRA ranks under
`r{N}/`, and disambiguates LLM prompt strategies (doc_0shot, doc_1shot,
icl_0shot, icl_5shot, icl_5shot_semantic) under `<strategy>/`. Original
file paths are recorded in `metadata.original_path`.

## Setup

Requires Python ≥ 3.9.

```bash
git clone https://github.com/abumafrim/AfriScience-MT-Latest.git
cd AfriScience-MT-Latest
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
