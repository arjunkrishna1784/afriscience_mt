#!/usr/bin/env bash
# Rerun gemini-1.5-flash legacy prompts on gemini-3.1-flash-lite.
#
# Two stages, stop-on-failure (set -e):
#   1. T=0   from legacy_temp 0.0  (both directions: templated + semantic)
#   2. T=0.6 from legacy_temp 0.6  with --direction e2x   (eng->xxx only;
#      we explicitly skip xxx->eng at T=0.6 per the user decision on
#      2026-05-12 — only the T=0.6 eng->xxx quadrant is comparable to
#      GPT-5.4's T=0.6 eng->xxx; xxx->eng at T=0.6 has no GPT match)
#
# Brevo emails are sent automatically if BREVO_API_KEY + BREVO_SENDER_EMAIL
# are set in .env.
#
# Run:        bash scripts/run_rerun_gemini_31_flash_lite.sh
# Background: nohup bash scripts/run_rerun_gemini_31_flash_lite.sh > /dev/null 2>&1 &
#             tail -f logs/gemini_31_flash_lite_*.log

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

run_stage() {
    local tag="$1"; shift
    local log="logs/${SHORT}_${tag}_${TS}.log"
    echo "[$(date '+%F %T')] starting $MODEL stage=$tag  ->  $log"
    python scripts/rerun_gemini_newer.py \
        --model "$MODEL" \
        --short-name "$SHORT" \
        --use-batch \
        --skip-existing \
        --notify-every 25 \
        "$@" \
        2>&1 | tee "$log"
    echo "[$(date '+%F %T')] finished $MODEL stage=$tag"
}

# Stage 1: T=0, both directions (matches the legacy gemini-1.5 design).
run_stage t0       --temperature 0.0

# Stage 2: T=0.6 only for eng->xxx — skip xxx->eng since GPT-5.4 has no
# T=0.6 xxx->eng coverage to compare against (legacy gpt-4o was run
# split by direction).
run_stage t06_e2x  --temperature 0.6 --direction e2x

echo "[$(date '+%F %T')] all $MODEL reruns done."
