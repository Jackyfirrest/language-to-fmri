#!/bin/bash
# Rerun LoRA embedding extraction (clean base model) + ridge regression.
# Submit from the repo root:
#   sbatch code/embeddings/rerun_fixed_lora.sh
#SBATCH -J fixed_lora_pipeline
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A100:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -euo pipefail

WORKDIR="/accounts/masters/shizhe_zhang/personal/lab-3-group-08"
PYTHON="/scratch/users/shizhe_zhang/conda/envs/lab3/bin/python"

LOG_DIR="$WORKDIR/code/embeddings/logs"
mkdir -p "$LOG_DIR"
JOB_ID="${SLURM_JOB_ID:-local}"
LOG_FILE="$LOG_DIR/fixed_lora_pipeline_${JOB_ID}.out"
exec > "$LOG_FILE" 2>&1

echo "[$(date)] ── Step 1: re-extract embeddings with clean LoRA reload ──"
cd "$WORKDIR"

$PYTHON code/embeddings/fine_tune_bert.py \
    --data-path  /scratch/users/s214/lab3 \
    --split-json data/train_test_split.json \
    --output-dir data/embeddings \
    --layer      last4_mean \
    --pooling    mean \
    --window-size 256 \
    --stride     128 \
    --batch-size 4 \
    --skip-training \
    --checkpoint-dir checkpoints/lora_best

echo "[$(date)] ── Step 2: ridge regression s2 ──"
$PYTHON code/ridge_regression/run_ridge.py \
    --subject    subject2 \
    --subject-id s2 \
    --method     finetuned_lora \
    --data-path  /scratch/users/s214/lab3 \
    --embedding-dir "$WORKDIR/data/embeddings" \
    --split-path    "$WORKDIR/data/train_test_split.json" \
    --result-dir    "$WORKDIR/results" \
    --n-alphas 10 \
    --alpha-min-log10 0.0 \
    --alpha-max-log10 3.0 \
    --chunk-size 500

echo "[$(date)] ── Step 3: ridge regression s3 ──"
$PYTHON code/ridge_regression/run_ridge.py \
    --subject    subject3 \
    --subject-id s3 \
    --method     finetuned_lora \
    --data-path  /scratch/users/s214/lab3 \
    --embedding-dir "$WORKDIR/data/embeddings" \
    --split-path    "$WORKDIR/data/train_test_split.json" \
    --result-dir    "$WORKDIR/results" \
    --n-alphas 10 \
    --alpha-min-log10 0.0 \
    --alpha-max-log10 3.0 \
    --chunk-size 500

echo "[$(date)] ── All done ──"
