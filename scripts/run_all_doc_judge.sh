#!/usr/bin/env bash
# Run the GPT-5.5 LLM judge over every doc-level closed-API result
# (new + legacy) in one pass. The judge processes one cell at a time
# but sends progress emails every 20 cells via Brevo.
#
# To also produce a second judge pass with Gemini for averaging, run
# both judges in parallel:
#
#     bash scripts/run_all_doc_judge.sh                  # gpt-5.5 (default)
#     bash scripts/run_all_doc_judge.sh gemini           # gemini-3.1-flash-lite
#
# Or fire both at once with separate logs:
#
#     nohup bash scripts/run_all_doc_judge.sh > /dev/null 2>&1 &
#     nohup bash scripts/run_all_doc_judge.sh gemini > /dev/null 2>&1 &
#
# Run:
#     bash scripts/run_all_doc_judge.sh
# Background:
#     nohup bash scripts/run_all_doc_judge.sh > /dev/null 2>&1 &
#     tail -f logs/judge_*.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$REPO_ROOT/.env"
    set +a
fi

# Mode: gpt (default) or gemini.
MODE="${1:-gpt}"
case "$MODE" in
    gpt)
        JUDGE_MODEL="gpt-5.5"
        OUT_NAME="judge"
        TAG="gpt55"
        ;;
    gemini)
        JUDGE_MODEL="gemini-3.1-flash-lite"
        OUT_NAME="judge_gemini"
        TAG="gemini_31_flash_lite"
        ;;
    *)
        echo "usage: $0 [gpt|gemini]" >&2
        exit 2
        ;;
esac

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
LOG="logs/judge_${TAG}_${TS}.log"

echo "[$(date '+%F %T')] starting LLM judge with $JUDGE_MODEL  ->  $LOG"

python scripts/run_doc_llm_judge.py \
    --src experiments/llm/closed/doc/gpt54 \
    --src experiments/llm/closed/doc/gemini_31_flash_lite \
    --src experiments/llm/closed/doc/gpt4o \
    --src experiments/llm/closed/doc/gemini_15_flash \
    --judge-model "$JUDGE_MODEL" \
    --out-name "$OUT_NAME" \
    --notify-every 20 \
    2>&1 | tee "$LOG"

echo "[$(date '+%F %T')] finished LLM judge ($JUDGE_MODEL)."
