#!/usr/bin/env bash
# Fire both doc-level translation reruns in parallel so we get results
# fast. Each invocation is fully independent: its own log file, its own
# Brevo emails (the underlying scripts/rerun_*_doc.py send progress
# emails every --notify-every cells, plus a "started" and "complete"
# email, plus a "failed" email on unhandled exceptions).
#
# Each job is launched with nohup so closing the shell does not kill it.
#
# After both translation jobs finish (you will get two "Run complete"
# emails), run the judge separately:
#
#     bash scripts/run_all_doc_judge.sh
#
# Run:
#     bash scripts/run_all_doc.sh
#
# Tail logs while running:
#     tail -f logs/gpt54_doc_*.log logs/gemini_31_flash_lite_doc_*.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs

echo "[$(date '+%F %T')] launching gpt-5.4 doc rerun in background ..."
nohup bash scripts/run_doc_gpt54.sh > /dev/null 2>&1 &
echo "  PID: $!"

echo "[$(date '+%F %T')] launching gemini-3.1-flash-lite doc rerun in background ..."
nohup bash scripts/run_doc_gemini_31_flash_lite.sh > /dev/null 2>&1 &
echo "  PID: $!"

echo
echo "Both translation jobs running in the background."
echo "Tail logs:"
echo "  tail -f logs/gpt54_doc_*.log"
echo "  tail -f logs/gemini_31_flash_lite_doc_*.log"
echo
echo "Each job sends its own Brevo progress emails (set BREVO_API_KEY"
echo "+ BREVO_SENDER_EMAIL in .env). After both 'Run complete' emails,"
echo "kick off the judge:"
echo "  bash scripts/run_all_doc_judge.sh"
