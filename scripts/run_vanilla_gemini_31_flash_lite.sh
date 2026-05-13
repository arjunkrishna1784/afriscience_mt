#!/usr/bin/env bash
# Run gemini-3.1-flash-lite on the vanilla (open-LLM-comparable) prompts:
#   sent 0shot + sent 5shot + doc 0shot + doc 1shot, all at T=0.
#
# See scripts/run_vanilla_gpt54.sh for context.
#
# Run:        bash scripts/run_vanilla_gemini_31_flash_lite.sh
# Background: nohup bash scripts/run_vanilla_gemini_31_flash_lite.sh > /dev/null 2>&1 &
#             tail -f logs/gemini_31_flash_lite_vanilla_*.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$REPO_ROOT/.env"
    set +a
fi

MODEL="gemini-3.1-flash-lite"
SHORT="gemini_31_flash_lite"
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

LOG="logs/${SHORT}_vanilla_${TS}.log"
echo "[$(date '+%F %T')] starting $MODEL vanilla (sent+doc) @ T=0  ->  $LOG"
python scripts/rerun_google_vanilla.py \
    --model "$MODEL" \
    --short-name "$SHORT" \
    --temperature 0.0 \
    --use-batch \
    --skip-existing \
    --notify-every 10 \
    2>&1 | tee "$LOG"
echo "[$(date '+%F %T')] finished $MODEL vanilla."
