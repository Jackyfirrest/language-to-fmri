#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_finetuned_lora_parallel.sh — SLURM batch script for finetuned_lora ridge
#
# Submit a parallel array job (splits voxels across N tasks):
#   sbatch --array=0-9 run_finetuned_lora_parallel.sh
#
# Optional overrides via environment variables:
#   SUBJECT=subject3 SUBJECT_ID=s3 TASKS=10 CHUNK_SIZE=500 N_ALPHAS=10 \
#   CV_FRAC=0.2 SAVE_WEIGHTS=1 sbatch --array=0-9 run_finetuned_lora_parallel.sh
#
# After the array completes, merge partial results with:
#   python merge_finetuned_lora_results.py --subject-id s3
#
# Then run top-5% extraction and SHAP:
#   python extract_top_voxels.py --subject-id s3 --method finetuned_lora
#   python shap_analysis.py --subject-id s3 --method finetuned_lora
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J lora_ridge
#SBATCH --partition=RM-shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000M
#SBATCH --time=08:00:00

set -euo pipefail

PYTHON="${PYTHON:-}"

# Prefer the user's conda 'stat214' python if present, else fall back to system python3/python
if [[ -z "${PYTHON}" ]]; then
    if [[ -x "${HOME}/.conda/envs/stat214/bin/python" ]]; then
        PYTHON="${HOME}/.conda/envs/stat214/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON="$(command -v python)"
    else
        echo "ERROR: no Python interpreter found. Set PYTHON env or install Python." >&2
        exit 1
    fi
fi

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

SUBJECT="${SUBJECT:-subject3}"
SUBJECT_ID="${SUBJECT_ID:-s3}"
METHOD="${METHOD:-finetuned_lora}"
CHUNK_SIZE="${CHUNK_SIZE:-500}"
N_ALPHAS="${N_ALPHAS:-10}"
ALPHA_MIN_LOG10="${ALPHA_MIN_LOG10:-0.0}"
ALPHA_MAX_LOG10="${ALPHA_MAX_LOG10:-3.0}"
CV_FRAC="${CV_FRAC:-0.2}"
SAVE_WEIGHTS="${SAVE_WEIGHTS:-1}"
DATA_PATH="${DATA_PATH:-/ocean/projects/mth250011p/shared/215a/final_project/data}"
EMBEDDING_DIR="${EMBEDDING_DIR:-../../../data/embeddings}"
SPLIT_PATH="${SPLIT_PATH:-../../../data/train_test_split.json}"
RESULT_DIR="${RESULT_DIR:-../../../results}"
TASK_COUNT="${TASK_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"
TASK_ID="${TASK_ID:-${SLURM_ARRAY_TASK_ID:-0}}"

LOG_DIR="$WORKDIR/logs"
mkdir -p "$LOG_DIR"

JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
LOG_FILE="$LOG_DIR/lora_ridge_${JOB_ID}_${TASK_ID}.out"
exec > "$LOG_FILE" 2>&1

echo "[$(date)] Starting finetuned_lora ridge task ${TASK_ID}/${TASK_COUNT}"
echo "Args: subject=${SUBJECT} subject-id=${SUBJECT_ID} method=${METHOD}"
echo "Args: chunk-size=${CHUNK_SIZE} n-alphas=${N_ALPHAS} cv-frac=${CV_FRAC}"

cd "$WORKDIR"

CMD=(
    "$PYTHON" ../run_ridge.py
    --subject "$SUBJECT"
    --subject-id "$SUBJECT_ID"
    --method "$METHOD"
    --data-path "$DATA_PATH"
    --embedding-dir "$EMBEDDING_DIR"
    --split-path "$SPLIT_PATH"
    --result-dir "$RESULT_DIR"
    --chunk-size "$CHUNK_SIZE"
    --n-alphas "$N_ALPHAS"
    --alpha-min-log10 "$ALPHA_MIN_LOG10"
    --alpha-max-log10 "$ALPHA_MAX_LOG10"
    --cv-frac "$CV_FRAC"
    --task-id "$TASK_ID"
    --task-count "$TASK_COUNT"
)

if [[ "$SAVE_WEIGHTS" != "0" && "$SAVE_WEIGHTS" != "false" && "$SAVE_WEIGHTS" != "False" ]]; then
    CMD+=(--save-weights)
fi

"${CMD[@]}"

echo "[$(date)] Done: task ${TASK_ID}"
