#!/usr/bin/env bash
# Generate doc-level translations on gpt-5.4 across the full 3-template x
# 3-shot grid (T=0 only — there is no doc-level temperature ablation in
# the paper plan; the closed-API temperature ablation lives at sentence
# level).
#
# Brevo emails are sent automatically if BREVO_API_KEY + BREVO_SENDER_EMAIL
# are set in .env.
#
# Run:        bash scripts/run_doc_gpt54.sh
# Background: nohup bash scripts/run_doc_gpt54.sh > /dev/null 2>&1 &
#             tail -f logs/gpt54_doc_*.log

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

LOG="logs/${SHORT}_doc_${TS}.log"
echo "[$(date '+%F %T')] starting $MODEL doc-level @ T=0  ->  $LOG"
python scripts/rerun_openai_doc.py \
    --model "$MODEL" \
    --short-name "$SHORT" \
    --temperature 0.0 \
    --use-batch \
    --skip-existing \
    --notify-every 10 \
    2>&1 | tee "$LOG"
echo "[$(date '+%F %T')] finished $MODEL doc-level."
