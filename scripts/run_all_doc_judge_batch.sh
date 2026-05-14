#!/usr/bin/env bash
# Submit both LLM-judges (GPT-5.5 + Gemini-3.1-flash-lite) as batched
# jobs over the doc-level closed-API cells for the new models
# (gpt54 + gemini_31_flash_lite). Each judge runs in its own process,
# so submission + polling proceed in parallel. State files at
# experiments/_judge_<model>_batch_state.json keep both jobs resumable.
#
# Modes:
#   bash scripts/run_all_doc_judge_batch.sh                    # both
#   bash scripts/run_all_doc_judge_batch.sh gpt                # gpt-5.5 only
#   bash scripts/run_all_doc_judge_batch.sh gemini             # gemini only
#
# Background:
#   nohup bash scripts/run_all_doc_judge_batch.sh > /dev/null 2>&1 &
#   tail -f logs/judge_batch_*.log

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

MODE="${1:-both}"
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

# Scope B (locked): new closed-API doc cells only.
SRCS=(
    --src experiments/llm/closed/doc/gpt54
    --src experiments/llm/closed/doc/gemini_31_flash_lite
)

run_judge() {
    local model="$1" out_name="$2" tag="$3"
    local log="logs/judge_batch_${tag}_${TS}.log"
    echo "[$(date '+%F %T')] starting batched judge=$model  ->  $log"
    python scripts/run_doc_llm_judge_batch.py \
        "${SRCS[@]}" \
        --judge-model "$model" \
        --out-name "$out_name" \
        --batch-poll-seconds 60 \
        2>&1 | tee "$log"
    echo "[$(date '+%F %T')] finished batched judge=$model"
}

case "$MODE" in
    gpt)
        run_judge "gpt-5.5" "judge" "gpt55"
        ;;
    gemini)
        run_judge "gemini-3.1-flash-lite" "judge_gemini" "gemini_31_flash_lite"
        ;;
    both)
        # Fire each in the background so both submissions go out
        # immediately and both pollers run in parallel. Wait so the
        # script blocks until both finish (cleaner for nohup).
        (run_judge "gpt-5.5" "judge" "gpt55") &
        gpt_pid=$!
        (run_judge "gemini-3.1-flash-lite" "judge_gemini" "gemini_31_flash_lite") &
        gem_pid=$!
        echo "launched gpt55 (pid $gpt_pid) and gemini (pid $gem_pid); waiting..."
        wait "$gpt_pid" "$gem_pid"
        ;;
    *)
        echo "usage: $0 [both|gpt|gemini]" >&2
        exit 2
        ;;
esac

echo "[$(date '+%F %T')] all judges done."
