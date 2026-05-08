#!/bin/bash
# Inference + eval for Gemma-2-9B-IT across the llama3_8b template/shot sweep.
#
# Phase 1: vLLM venv  -> generate translations, write predictions.json (+ stub results.json)
# Phase 2: dcs env    -> rescore BLEU, chrF, SSA-COMET into results.json
#
# Resumable: phase 1 skips configs whose predictions.json exists; phase 2
# skips results whose metric is already set (rescore_missing_ssa.py behavior;
# bleu/chrf are recomputed in-place each run, which is cheap).
#
# Usage:
#   bash scripts/run_gemma2_9b_it_templates.sh                      # full sweep, 1 GPU
#   bash scripts/run_gemma2_9b_it_templates.sh --tp 2               # 2-GPU TP
#   bash scripts/run_gemma2_9b_it_templates.sh --templates 1 2 4    # subset
#
# All extra args are forwarded to the python inference script.

set -euo pipefail

REPO=/ext_data/idris/DCS-MT-Framework
VLLM_PY=/home/idris/ext_data/venvs/vllm/bin/python
DCS_PY=/home/idris/miniconda3/envs/dcs/bin/python
OUT_SUBDIR=experiments/llm/open/icl/gemma2_9b_it

cd "$REPO"

echo "=========================================="
echo "Phase 1: vLLM inference (Gemma-2-9B-IT)"
echo "=========================================="
"$VLLM_PY" scripts/evaluate_gemma2_9b_it_templates.py "$@"

echo
echo "=========================================="
echo "Phase 2: dcs eval (BLEU, chrF, SSA-COMET)"
echo "=========================================="

echo "[2/3] BLEU..."
"$DCS_PY" -m afriscience_mt.commands.evaluate \
    --mode rescore --src "$OUT_SUBDIR" --metric bleu

echo "[2/3] chrF..."
"$DCS_PY" -m afriscience_mt.commands.evaluate \
    --mode rescore --src "$OUT_SUBDIR" --metric chrf

echo "[3/3] SSA-COMET..."
"$DCS_PY" scripts/rescore_missing_ssa.py --src "$OUT_SUBDIR"

echo
echo "✓ all done. results in: $OUT_SUBDIR/template*/<pair>/results.json"
