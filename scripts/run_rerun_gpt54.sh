#!/usr/bin/env bash
# Rerun gpt-4o legacy prompts on gpt-5.4. T=0 first, then T=0.6.
# Stops on the first failure (set -e) — T=0.6 only starts if T=0 finishes cleanly.
# Brevo emails are sent automatically if BREVO_API_KEY + BREVO_SENDER_EMAIL are set in .env.
#
# Run:        bash scripts/run_rerun_gpt54.sh
# Background: nohup bash scripts/run_rerun_gpt54.sh > /dev/null 2>&1 &
#             tail -f logs/gpt54_*.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Load .env so OPENAI_API_KEY, BREVO_API_KEY, BREVO_SENDER_EMAIL, etc.
# are exported into the child python process. `set -a` auto-exports
# every assignment until `set +a`.
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

run_one() {
    local temp="$1"
    local tag="$2"
    local log="logs/${SHORT}_${tag}_${TS}.log"
    echo "[$(date '+%F %T')] starting $MODEL @ T=$temp  →  $log"
    python scripts/rerun_gpt4o_templated.py \
        --model "$MODEL" \
        --short-name "$SHORT" \
        --temperature "$temp" \
        --use-batch \
        --notify-every 25 \
        2>&1 | tee "$log"
    echo "[$(date '+%F %T')] finished $MODEL @ T=$temp"
}

run_one 0.0 t0
run_one 0.6 t06

echo "[$(date '+%F %T')] all $MODEL reruns done."
