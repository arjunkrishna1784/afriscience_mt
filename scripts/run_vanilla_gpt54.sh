#!/usr/bin/env bash
# Run gpt-5.4 on the vanilla (open-LLM-comparable) prompts:
#   sent 0shot + sent 5shot + doc 0shot + doc 1shot, all at T=0.
#
# Uses configs/prompts/sentence_templates.yaml (explicit / few_shot) and
# configs/prompts/document_templates.yaml (basic) so the closed-API
# vanilla cells are byte-comparable to the open-LLM vanilla cells
# already on disk under experiments/llm/open/icl/<model>/{0,5}shot/
# and .../open/doc/<model>/{0,1}shot/.
#
# Brevo emails fire if BREVO_API_KEY + BREVO_SENDER_EMAIL are set in .env.
#
# Run:        bash scripts/run_vanilla_gpt54.sh
# Background: nohup bash scripts/run_vanilla_gpt54.sh > /dev/null 2>&1 &
#             tail -f logs/gpt54_vanilla_*.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$REPO_ROOT/.env"
    set +a
fi

MODEL="gpt-5.4"
SHORT="gpt54"
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

LOG="logs/${SHORT}_vanilla_${TS}.log"
echo "[$(date '+%F %T')] starting $MODEL vanilla (sent+doc) @ T=0  ->  $LOG"
python scripts/rerun_openai_vanilla.py \
    --model "$MODEL" \
    --short-name "$SHORT" \
    --temperature 0.0 \
    --use-batch \
    --skip-existing \
    --notify-every 10 \
    2>&1 | tee "$LOG"
echo "[$(date '+%F %T')] finished $MODEL vanilla."
