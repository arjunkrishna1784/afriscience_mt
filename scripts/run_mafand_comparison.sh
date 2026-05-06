#!/usr/bin/env bash
# Train NLLB on MAFAND, DCS, and DCS+MAFAND for the 4 supported African
# languages (hau, lug, yor, zul) in both directions, then aggregate the
# results into a comparison table. Run inside tmux:
#
#   tmux new -s mafand
#   bash scripts/run_mafand_comparison.sh
#   # detach: Ctrl-b d   re-attach: tmux a -t mafand
#
# Override defaults via env, e.g.:
#   GPU=1 EPOCHS=10 MODEL=facebook/nllb-200-1.3B \
#       bash scripts/run_mafand_comparison.sh

set -euo pipefail

# --- Config (override via env) -----------------------------------------------
GPU="${GPU:-0}"
MODEL="${MODEL:-facebook/nllb-200-distilled-600M}"
# SETUPS = whitespace-separated list, runs each in turn.
# Defaults to mafand + dcs_mafand (the two needed for the AfriScience ablation).
# DCS-only is omitted because the existing DCS seq2seq results already exist
# in experiments/seq2seq_finetune/, and data/processed_files/nllb/ has been
# pruned. Add 'dcs' here only if you re-add data/processed_files/nllb/.
SETUPS="${SETUPS:-mafand dcs_mafand}"
LANGS="${LANGS:-hau lug yor zul}"
DIRECTIONS="${DIRECTIONS:-both}"   # from-eng | to-eng | both
EPOCHS="${EPOCHS:-5}"
BATCH="${BATCH:-8}"            # 1.3B fits at 8 with grad accum; bump for 600M
GRAD_ACCUM="${GRAD_ACCUM:-2}"  # effective batch = BATCH * GRAD_ACCUM = 16
LR="${LR:-2e-5}"
PRECISION="${PRECISION:-fp16}"
SEED="${SEED:-42}"
TEST_SET="${TEST_SET:-dcs}"    # dcs | mafand (default dcs for paper-comparable scores)
KEEP_CKPT="${KEEP_CKPT:-0}"    # 1 = pass --keep-checkpoints

# --- Paths -------------------------------------------------------------------
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="/ext_data/idris/DCS-MT-Updated/.venv/bin/python"
HF_HOME_DIR="/ext_data/idris/DCS-MT-Updated/hf_cache"
LOG_DIR="$REPO/logs"
LOG_FILE="$LOG_DIR/mafand_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

# --- Pre-flight --------------------------------------------------------------
if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: venv python not found at $VENV_PY" >&2
    echo "       Edit VENV_PY in this script to point at your env." >&2
    exit 1
fi
if [[ ! -d "$HF_HOME_DIR" ]]; then
    echo "WARN: HF cache dir $HF_HOME_DIR not found; HF will use ~/.cache."
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARN: nvidia-smi not found; cannot verify GPU $GPU." >&2
fi

# --- Banner ------------------------------------------------------------------
{
echo "============================================================"
echo " MAFAND ± AfriScience comparison"
echo "============================================================"
echo " repo:       $REPO"
echo " gpu:        $GPU"
echo " model:      $MODEL"
echo " setups:     $SETUPS"
echo " langs:      $LANGS"
echo " test_set:   $TEST_SET"
echo " directions: $DIRECTIONS"
echo " epochs:     $EPOCHS  batch: $BATCH x grad_accum $GRAD_ACCUM  lr: $LR  precision: $PRECISION"
echo " log:        $LOG_FILE"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader 2>/dev/null \
    | awk -F, -v g="$GPU" '$1+0==g {print " GPU '$GPU' free:" $3}'
echo
} | tee -a "$LOG_FILE"

# --- Build common args -------------------------------------------------------
COMMON_ARGS=(
    --model "$MODEL"
    --langs $LANGS
    --directions "$DIRECTIONS"
    --num-epochs "$EPOCHS"
    --batch-size "$BATCH"
    --gradient-accumulation-steps "$GRAD_ACCUM"
    --learning-rate "$LR"
    --precision "$PRECISION"
    --seed "$SEED"
    --test-set "$TEST_SET"
)
if [[ "$KEEP_CKPT" == "1" ]]; then
    COMMON_ARGS+=(--keep-checkpoints)
fi

run_python () {
    CUDA_VISIBLE_DEVICES="$GPU" \
    HF_HOME="$HF_HOME_DIR" \
    PYTHONPATH="$REPO" \
    "$VENV_PY" "$@"
}

# --- Show plan first ---------------------------------------------------------
echo "[plan]" | tee -a "$LOG_FILE"
for SETUP in $SETUPS; do
    run_python "$REPO/scripts/train_mafand_comparison.py" \
        "${COMMON_ARGS[@]}" --setup "$SETUP" --dry-run \
        2>&1 | tee -a "$LOG_FILE"
done

# --- Run ---------------------------------------------------------------------
echo | tee -a "$LOG_FILE"
echo "[run] starting at $(date)" | tee -a "$LOG_FILE"
START_TS=$(date +%s)
TRAIN_RC=0
for SETUP in $SETUPS; do
    echo | tee -a "$LOG_FILE"
    echo "[run] === setup=$SETUP ===" | tee -a "$LOG_FILE"
    SETUP_T0=$(date +%s)
    set +e
    run_python "$REPO/scripts/train_mafand_comparison.py" \
        "${COMMON_ARGS[@]}" --setup "$SETUP" \
        2>&1 | tee -a "$LOG_FILE"
    SETUP_RC=${PIPESTATUS[0]}
    set -e
    SETUP_DT=$(($(date +%s) - SETUP_T0))
    echo "[run] setup=$SETUP rc=$SETUP_RC time=${SETUP_DT}s" | tee -a "$LOG_FILE"
    if [[ $SETUP_RC -ne 0 ]]; then
        TRAIN_RC=$SETUP_RC
        echo "[run] FAILED on setup=$SETUP — continuing to next setup" | tee -a "$LOG_FILE"
    fi
done
END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

if [[ $TRAIN_RC -ne 0 ]]; then
    echo "[run] completed with failures after ${ELAPSED}s ($((ELAPSED/60))m)" | tee -a "$LOG_FILE"
else
    echo "[run] all setups done in ${ELAPSED}s ($((ELAPSED/60))m)" | tee -a "$LOG_FILE"
fi

# --- Aggregate + emit comparison table --------------------------------------
echo | tee -a "$LOG_FILE"
echo "[post] aggregating results" | tee -a "$LOG_FILE"
PYTHONPATH="$REPO" "$VENV_PY" -m afriscience_mt.cli aggregate \
    --src "$REPO/experiments" \
    --out "$REPO/experiments/summary.csv" \
    2>&1 | tee -a "$LOG_FILE"

MODEL_SHORT=$("$VENV_PY" -c "import sys; sys.path.insert(0,'$REPO'); from afriscience_mt.core.registry import short_name; print(short_name('$MODEL'))")

echo "[post] generating comparison table (chrf, ssa_comet)" | tee -a "$LOG_FILE"
mkdir -p "$REPO/tables"
for METRIC in chrf ssa_comet bleu; do
    PYTHONPATH="$REPO" "$VENV_PY" -m afriscience_mt.cli table \
        --filter-type seq2seq_finetune \
        --filter-model "$MODEL_SHORT" \
        --layout dataset_compare \
        --metric "$METRIC" \
        --bold-best --format tex \
        --out "$REPO/tables/mafand_compare_${MODEL_SHORT}_${METRIC}.tex" \
        2>&1 | tee -a "$LOG_FILE"
done

echo | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " DONE. Logs: $LOG_FILE" | tee -a "$LOG_FILE"
echo " Tables:    $REPO/tables/mafand_compare_${MODEL_SHORT}_*.tex" | tee -a "$LOG_FILE"
echo " Results:   $REPO/experiments/seq2seq_finetune/$MODEL_SHORT/" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
