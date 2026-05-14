#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_ridge.sh — SLURM batch script for run_ridge.py
#
# Submit a single run (all voxels, one task):
#   sbatch run_ridge.sh --subject subject2 --subject-id s2 --method bow
#
# Submit a parallel array job (splits voxels across N tasks):
#   sbatch --array=0-9 run_ridge.sh --subject subject2 --subject-id s2 --method glove
#
# After array completes, merge partial results with:
#   python merge_ridge_results.py --subject-id s2 --method glove
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J run_ridge
#SBATCH --partition=low
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}/code/ridge_regression"
PYTHON="${PYTHON:-python}"

LOG_DIR="$WORKDIR/logs"
mkdir -p "$LOG_DIR"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
LOG_FILE="$LOG_DIR/run_ridge_${JOB_ID}_${TASK_ID}.out"

exec > "$LOG_FILE" 2>&1

echo "[$(date)] Starting: task ${TASK_ID}/${SLURM_ARRAY_TASK_COUNT:-1}"
echo "Args: $*"

cd "$WORKDIR"
$PYTHON run_ridge.py \
    --chunk-size 500 \
    --n-alphas 10 \
    --n-alpha-voxels 200 \
    "$@"

echo "[$(date)] Done: task ${TASK_ID}"
