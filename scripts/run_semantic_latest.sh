#!/usr/bin/env bash
# Re-run semantic-similarity ICL (5/10/20-shot) for the latest two
# closed APIs (GPT-5.4, Gemini-3.1-Flash-Lite) at T=0 on eng->xxx.
#
# Background (root cause of the bad existing predictions):
# Legacy DCS-MT semantic TSVs save the prompt template WITHOUT the
# source text — the source lives in a separate `text` column and was
# injected at API call time as a LangChain Human message. The previous
# rerun pass replayed the bare templates, so:
#   * GPT-5.4 returned ~60% empty hyps and "please send the English
#     text" boilerplate (COMET ~16); reasoning-mode refuses when there
#     is nothing to translate.
#   * Gemini-3.1 returned non-empty but degraded translations driven
#     entirely by the in-context examples (COMET ~55 vs templated ~68).
#
# The fix is in scripts/rerun_gpt4o_templated.py + rerun_gemini_newer.py:
# read_prompts() now appends the source text from the TSV's `text` /
# `src_text` column when no `\nHuman: ` marker is present, so semantic
# prompts arrive at the API with the source attached.
#
# This script:
#   1. Backs up the existing broken predictions to a timestamped
#      sibling directory under experiments/llm/closed/discarded/...
#      (so --skip-existing won't preserve the bad data).
#   2. Runs gpt54 + gemini_31_flash_lite semantic_{5,10,20}shot at T=0
#      via the OpenAI / Google Batch APIs (50% cheaper, ~24h SLA).
#   3. Brevo emails fire if BREVO_API_KEY + BREVO_SENDER_EMAIL are set.
#
# Run:        bash scripts/run_semantic_latest.sh
# Background: nohup bash scripts/run_semantic_latest.sh > /dev/null 2>&1 &
#             tail -f logs/semantic_latest_*.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$REPO_ROOT/.env"
    set +a
fi

# Activate the project conda env so afriscience_mt + omegaconf resolve.
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate dcs
fi

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

# Reasoning-effort low: cheaper credits + bigger output budget for the
# same max_completion_tokens (less chance of empty hyps from reasoning
# starvation on GPT-5.4).
export OPENAI_REASONING_EFFORT=low

# --- Step 1: back up the existing broken semantic predictions -----------
backup_broken() {
    local model_short="$1"
    local backup_root="experiments/llm/closed/discarded/t0/icl/${model_short}_broken_semantic_${TS}"
    for k in 5 10 20; do
        local src="experiments/llm/closed/t0/icl/${model_short}/semantic_${k}shot"
        if [ -d "$src" ]; then
            mkdir -p "$backup_root"
            mv "$src" "$backup_root/semantic_${k}shot"
            echo "[$(date '+%F %T')] backed up $src -> $backup_root/semantic_${k}shot"
        fi
    done
}

backup_broken gpt54
backup_broken gemini_31_flash_lite

# --- Step 2: re-run with the patched read_prompts -----------------------
# templates=0 filters to semantic-only; direction=e2x because the legacy
# semantic dump only has eng->xxx.

GPT_LOG="logs/semantic_latest_gpt54_${TS}.log"
echo "[$(date '+%F %T')] starting gpt-5.4 semantic_{5,10,20}shot @ T=0  ->  $GPT_LOG"
python scripts/rerun_gpt4o_templated.py \
    --model gpt-5.4 \
    --short-name gpt54 \
    --direction e2x \
    --templates 0 \
    --shot-counts 5 10 20 \
    --temperature 0.0 \
    --use-batch \
    --skip-existing \
    --notify-every 25 \
    --max-tokens 16384 \
    2>&1 | tee "$GPT_LOG"
echo "[$(date '+%F %T')] finished gpt-5.4 semantic"

GEM_LOG="logs/semantic_latest_gemini31_${TS}.log"
echo "[$(date '+%F %T')] starting gemini-3.1-flash-lite semantic_{5,10,20}shot @ T=0  ->  $GEM_LOG"
python scripts/rerun_gemini_newer.py \
    --model gemini-3.1-flash-lite \
    --short-name gemini_31_flash_lite \
    --direction e2x \
    --templates 0 \
    --shot-counts 5 10 20 \
    --temperature 0.0 \
    --use-batch \
    --skip-existing \
    --notify-every 25 \
    --max-output-tokens 16384 \
    2>&1 | tee "$GEM_LOG"
echo "[$(date '+%F %T')] finished gemini-3.1-flash-lite semantic"

echo "[$(date '+%F %T')] all semantic reruns done."
echo
echo "Next steps after batch jobs complete:"
echo "  1. python scripts/aggregate_summary.py    # refresh summary.csv"
echo "  2. python scripts/build_paper_tables.py   # regenerate tab:closed_ablations_*"
