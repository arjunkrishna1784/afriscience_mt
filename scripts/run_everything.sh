#!/usr/bin/env bash
# Fire ALL FOUR closed-API reruns in parallel so we get results urgently:
#
#   1. sentence-level  gpt-5.4                (T=0, then T=0.6)
#   2. sentence-level  gemini-3.1-flash-lite  (T=0, then T=0.6)
#   3. document-level  gpt-5.4                (T=0 only, 1 template x 2 shots = 24 cells)
#   4. document-level  gemini-3.1-flash-lite  (T=0 only, 24 cells)
#
# Each job runs as an independent nohup'd process with its own log file
# and its own Brevo email progress notifications (start / every-N / done /
# failed). Closing the shell does NOT kill the jobs.
#
# Sentence-level wrappers handle T=0 first, then T=0.6 sequentially within
# their own process — the parallelism here is across the four MODELS, not
# across temperatures.
#
# Run:
#     bash scripts/run_everything.sh
#
# Tail all four logs at once:
#     tail -f logs/gpt54_*.log logs/gemini_31_flash_lite_*.log
#
# AFTER all four "Run complete" emails arrive, fire the judge:
#     bash scripts/run_all_doc_judge.sh
#
# Or judge in parallel with both gpt-5.5 + gemini-3.1-flash-lite (for averaging):
#     nohup bash scripts/run_all_doc_judge.sh        > /dev/null 2>&1 &
#     nohup bash scripts/run_all_doc_judge.sh gemini > /dev/null 2>&1 &

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs

launch() {
    local label="$1"
    local script="$2"
    echo "[$(date '+%F %T')] launching $label  ($script)"
    nohup bash "$script" > /dev/null 2>&1 &
    echo "  PID: $!"
}

launch "sent / gpt-5.4 (templated)"            scripts/run_rerun_gpt54.sh
launch "sent / gemini-3.1-flash-lite (templated)" scripts/run_rerun_gemini_31_flash_lite.sh
launch "doc  / gpt-5.4 (templates 1-3)"        scripts/run_doc_gpt54.sh
launch "doc  / gemini-3.1-flash-lite (templates 1-3)" scripts/run_doc_gemini_31_flash_lite.sh
launch "vanilla / gpt-5.4 (sent+doc, open-LLM-comparable)"     scripts/run_vanilla_gpt54.sh
launch "vanilla / gemini-3.1-flash-lite (sent+doc, open-LLM-comparable)" scripts/run_vanilla_gemini_31_flash_lite.sh

echo
echo "All six jobs running in the background."
echo "Each will email progress independently via Brevo (set BREVO_API_KEY"
echo "+ BREVO_SENDER_EMAIL in .env)."
echo
echo "Tail logs:"
echo "  tail -f logs/gpt54_*.log logs/gemini_31_flash_lite_*.log"
echo
echo "Check process status:"
echo "  ps -fp \$(pgrep -f 'rerun_(gpt4o_templated|gemini_newer|openai_doc|google_doc)')"
echo
echo "When all four send 'Run complete' emails, kick off the judge:"
echo "  bash scripts/run_all_doc_judge.sh"
