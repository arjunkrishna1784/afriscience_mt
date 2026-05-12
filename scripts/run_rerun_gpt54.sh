#!/usr/bin/env bash
# Rerun gpt-4o legacy prompts on gpt-5.4.
#
# Four stages, all stop-on-failure (set -e):
#   1. T=0   from legacy_temp 0.0  (xxx->eng templated + eng->xxx semantic)
#   2. T=0.6 from legacy_temp 0.6  (eng->xxx templated)
#   3. T=0   from legacy_temp 0.6  with --direction e2x   (FILL GAP:
#      eng->xxx templated at T=0; legacy temp 0.0 doesn't have these
#      prompts so we replay the T=0.6 templated prompts at T=0)
#   4. T=0   semantic from legacy_temp 0.0 with --templates 0
#      (no-op if stage 1 already picked them up; kept as an explicit
#      "semantic only" pass for clarity / re-runs)
#
# Stage 3 is what the docs/PAPER_TODOS.md 2026-05-12 "Closed-API rerun
# asymmetry" note recommended (user picked option 2: fill the GPT gap).
#
# Brevo emails are sent automatically if BREVO_API_KEY + BREVO_SENDER_EMAIL
# are set in .env.
#
# Run:        bash scripts/run_rerun_gpt54.sh
# Background: nohup bash scripts/run_rerun_gpt54.sh > /dev/null 2>&1 &
#             tail -f logs/gpt54_*.log

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

# Args: <tag> <python-args...>
run_stage() {
    local tag="$1"; shift
    local log="logs/${SHORT}_${tag}_${TS}.log"
    echo "[$(date '+%F %T')] starting $MODEL stage=$tag  ->  $log"
    python scripts/rerun_gpt4o_templated.py \
        --model "$MODEL" \
        --short-name "$SHORT" \
        --use-batch \
        --skip-existing \
        --notify-every 25 \
        "$@" \
        2>&1 | tee "$log"
    echo "[$(date '+%F %T')] finished $MODEL stage=$tag"
}

# Stage 1: T=0 reading T=0 dump (xxx->eng templated + eng->xxx semantic)
run_stage t0       --temperature 0.0 --legacy-temp 0.0

# Stage 2: T=0.6 reading T=0.6 dump (eng->xxx templated)
run_stage t06      --temperature 0.6 --legacy-temp 0.6

# Stage 3: GAP FILL — gpt-5.4 eng->xxx templated at T=0 using T=0.6 prompts
run_stage t0_fill  --temperature 0.0 --legacy-temp 0.6 --direction e2x

echo "[$(date '+%F %T')] all $MODEL reruns done."
