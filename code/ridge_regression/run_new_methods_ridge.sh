#!/bin/bash
# Array ridge job for new embedding methods.
# Splits voxels across 10 tasks (same as original finetuned_lora runs).
#
# Called by submit_new_methods.sh — do not run directly.
#SBATCH -J new_ridge
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

WORKDIR="/accounts/masters/shizhe_zhang/personal/lab-3-group-08"
PYTHON="/scratch/users/shizhe_zhang/conda/envs/lab3/bin/python"
LOG_DIR="$WORKDIR/code/ridge_regression/logs"
mkdir -p "$LOG_DIR"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
LOG_FILE="$LOG_DIR/new_ridge_${JOB_ID}_task${TASK_ID}.out"
exec > "$LOG_FILE" 2>&1

echo "[$(date)] task $TASK_ID / method=$METHOD subj=$SUBJECT_ID"

cd "$WORKDIR"
$PYTHON code/ridge_regression/run_ridge.py \
    --subject    "$SUBJECT" \
    --subject-id "$SUBJECT_ID" \
    --method     "$METHOD" \
    --data-path  /scratch/users/s214/lab3 \
    --embedding-dir "$EMBEDDING_DIR" \
    --split-path    "$WORKDIR/data/train_test_split.json" \
    --result-dir    "$WORKDIR/results" \
    --n-alphas 10 --alpha-min-log10 0.0 --alpha-max-log10 3.0 \
    --chunk-size 500

echo "[$(date)] done task $TASK_ID"
