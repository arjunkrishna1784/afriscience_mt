# Session handoff — 2026-05-06

This file captures the state of long-running background work so a future
Claude Code session (or the user manually) can pick up where this session
left off.

## What's running in background (will survive terminal close)

Two NLLB-1.3B fine-tune sweeps for the per-domain temperature-balance
ablation. Both processes were launched as session leaders and reparent to
PID 1 (systemd) when the harness exits. Confirmed independent of the
parent terminal session.

| process | GPU | τ | output | status as of handoff |
|---|---|---|---|---|
| `bl77lbztb` (Bash bg task id) | 0 | 2 (sqrt) | `/ext_data/idris/tmp/claude-1015/-ext-data-idris-DCS-MT-Framework/050152b2-4fe8-4367-b502-be0b49af2009/tasks/bl77lbztb.output` | eng-amh ✓, eng-hau in flight |
| `buceaejre` (Bash bg task id) | 1 | ∞ (uniform) | `/ext_data/idris/tmp/claude-1015/-ext-data-idris-DCS-MT-Framework/050152b2-4fe8-4367-b502-be0b49af2009/tasks/buceaejre.output` | eng-amh ✓, eng-hau in flight |

Each sweep covers all 12 lang pairs (`eng↔{amh, hau, lug, nso, yor, zul}`).
~30-40 min per pair, so ~6-8 hours wall time per sweep, both running in
parallel. Outputs to `experiments/seq2seq/finetune/nllb_200_1_3b/balance_t{2,inf}/<pair>/`.

The launching commands (recoverable if the processes die):

```bash
cd /ext_data/idris/DCS-MT-Framework

# τ=2 on GPU 0
CUDA_VISIBLE_DEVICES=0 \
  DCSMT_HF_CACHE=/ext_data/idris/DCS-MT-Updated/hf_cache \
  HF_HOME=/ext_data/idris/DCS-MT-Updated/hf_cache \
  /home/idris/miniconda3/envs/dcs/bin/python \
    scripts/train_balance_comparison.py \
    --temperatures 2 --num-epochs 5 --batch-size 8 --grad-accum 2

# τ=∞ on GPU 1
CUDA_VISIBLE_DEVICES=1 \
  DCSMT_HF_CACHE=/ext_data/idris/DCS-MT-Updated/hf_cache \
  HF_HOME=/ext_data/idris/DCS-MT-Updated/hf_cache \
  /home/idris/miniconda3/envs/dcs/bin/python \
    scripts/train_balance_comparison.py \
    --temperatures inf --num-epochs 5 --batch-size 8 --grad-accum 2
```

Both scripts skip pairs whose `results.json` already exists, so re-running
is idempotent and recoverable from where they left off.

## How to check progress (works in any new shell)

```bash
# count completed pairs per τ (out of 12 each)
ls experiments/seq2seq/finetune/nllb_200_1_3b/balance_t2/  | wc -l
ls experiments/seq2seq/finetune/nllb_200_1_3b/balance_tinf/ | wc -l

# trainers alive?
pgrep -af train_balance_comparison

# most recent progress line for each
grep -E "done in|=====" /ext_data/idris/tmp/claude-1015/-ext-data-idris-DCS-MT-Framework/050152b2-4fe8-4367-b502-be0b49af2009/tasks/bl77lbztb.output | tail -5
grep -E "done in|=====" /ext_data/idris/tmp/claude-1015/-ext-data-idris-DCS-MT-Framework/050152b2-4fe8-4367-b502-be0b49af2009/tasks/buceaejre.output | tail -5

# disk room (got tight earlier; 128 GB free at handoff)
df -h /ext_data | tail -1
```

## What to do once both sweeps finish

1. **Sanity-check counts.** Both `balance_t2/` and `balance_tinf/` should
   each contain 12 pair-subdirs with `results.json` and `predictions.json`.
2. **Run SSA-COMET on the new ablation** (BLEU/chrF should already be in
   results.json from the trainer):

   ```bash
   export DCSMT_HF_CACHE=/ext_data/idris/DCS-MT-Updated/hf_cache
   /home/idris/miniconda3/envs/dcs/bin/afriscience-mt evaluate --mode rescore \
       --src experiments/seq2seq/finetune/nllb_200_1_3b/balance_t2 --metric ssa_comet
   /home/idris/miniconda3/envs/dcs/bin/afriscience-mt evaluate --mode rescore \
       --src experiments/seq2seq/finetune/nllb_200_1_3b/balance_tinf --metric ssa_comet
   ```

   (The trainer already attempts SSA-COMET inline, but if a run skipped it
   for any reason the rescore will fill it in.)

3. **Run SSA-COMET on the deferred Gemma mmt block** (imported but only
   BLEU/chrF rescored so far — COMET deferred so it didn't compete with
   the GPU sweeps):

   ```bash
   export DCSMT_HF_CACHE=/ext_data/idris/DCS-MT-Updated/hf_cache
   /home/idris/miniconda3/envs/dcs/bin/afriscience-mt evaluate --mode rescore \
       --src experiments/llm/open/multilingual/gemma2_9b --metric ssa_comet
   ```

4. **Refresh aggregate:**

   ```bash
   /home/idris/miniconda3/envs/dcs/bin/afriscience-mt aggregate \
       --src experiments --out experiments/summary.csv
   ```

5. **Commit + push.** Suggested message:

   ```
   Domain-balance ablation results + Gemma mmt LoRA SSA-COMET

   - 24 NLLB-1.3B fine-tunes under temperature sampling
     (τ=2 sqrt, τ=∞ uniform) × 12 lang pairs.
     Sampling via WeightedRandomSampler with weight (1/N_d)^(1/τ).
     Same total step budget across τ values; τ=1 baseline is the existing
     experiments/seq2seq/finetune/nllb_200_1_3b/dcs/ block.
   - SSA-COMET filled in for the imported Gemma-2-9B (base) mmt LoRA block.
   - Refreshed summary.csv to ~1100 rows.
   ```

## What's already in this session's commits (on origin/main)

Latest pushed commit: see `git log -1`. Recent history (newest first):

- `2fbf180` Add domain-balance trainer + paper results-section outline
- `bf06c15` Establish metric hierarchy: SSA-COMET primary, chrF secondary, BLEU tertiary
- `97baf7a` Add docs/EXPERIMENTS.md (paper-writing reference)
- `b3a58f0` Force-rewrite history to remove `.claude/` (was force-push)
- `86850b9` Big import session (multilingual NLLB, mmt LoRA, recovered 0shot ICL preds, gpt4o fixes)
- `59af722` Predictions store only hyp; rescore looks up src/ref from data/processed_files/test/
- `b3090ec` Import llama3_8b LoRA r{8,16,32,128} ablations
- earlier commits: rename dcsmt → afriscience_mt, restructure to seq2seq/llm{open,closed}, etc.

## Local state (uncommitted at handoff)

| path | what | will be committed by |
|---|---|---|
| `experiments/llm/open/multilingual/gemma2_9b/r{8,16,32,64,128}/<pair>/` | 30 gemma mmt LoRA results, BLEU/chrF only (COMET deferred) | this handoff commit |
| `experiments/seq2seq/finetune/nllb_200_1_3b/balance_t2/<pair>/` | growing as τ=2 sweep finishes pairs | this handoff commit (snapshot of partial state); next commit (full state once sweep finishes) |
| `experiments/seq2seq/finetune/nllb_200_1_3b/balance_tinf/<pair>/` | growing as τ=∞ sweep finishes pairs | this handoff commit (snapshot of partial state); next commit (full state once sweep finishes) |
| `scripts/import_gemma_mmt_lora.py` | one-shot import script for the gemma mmt block | this handoff commit |
| `docs/SESSION_HANDOFF.md` | this file | this handoff commit |

## Other context worth knowing about

- **Disk room.** `/ext_data` has 128 GB free at handoff (was 0 GB earlier
  when we hit a disk crisis on `/`; user removed 18 stale conda envs and
  three idle base-model HF caches plus Gemma-3 model cache). The active
  τ-sweep checkpoints live in `checkpoints/balance/` and self-prune as
  each pair finishes; peak usage during the sweep is ~50 GB. Should be
  safe.

- **Metric hierarchy** (`docs/EXPERIMENTS.md`): SSA-COMET primary, chrF
  for ablation tables, BLEU only when a baseline paper quoted it. Default
  `afriscience-mt table --metric` is already `ssa_comet`. The
  `pair_x_metric` layout lists the three in priority order.

- **Memory** (in `~/.claude/projects/-ext-data-idris-DCS-MT-Framework/memory/`):
  metric_hierarchy.md is saved as a project memory so future sessions
  inherit the convention.

- **Gemma mmt caveat** for the paper: the imported `gemma2_9b` mmt block
  uses `unsloth/gemma-2-9b-bnb-4bit` (base, NOT instruct), while the
  llama3_8b mmt block uses the instruct base. The gemma-vs-llama
  comparison is therefore not strictly apples-to-apples. Recorded in each
  result's `config.is_instruct_base = False` and in `metadata.import_note`.
  Recommended in the paper: footnote it; or train a matching
  `gemma-2-9b-it` mmt LoRA (~1-2 GPU-hours) for parity if the comparison
  is a headline claim.
