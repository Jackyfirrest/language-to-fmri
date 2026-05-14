#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_extract_finetuned.sh — re-extract finetuned_lora embeddings using
# sliding window (no re-training; loads from saved LoRA checkpoint).
#
# Submit from the repo root on SCF:
#   sbatch code/embeddings/run_extract_finetuned.sh
#
# Requires: checkpoints/lora_epoch8/ to exist in the repo root (saved by the
# previous fine-tuning job). After the job completes, upload embeddings to
# HuggingFace and re-run ridge regression.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J extract_finetuned_lora
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A100:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PYTHON="${PYTHON:-python}"

LOG_DIR="$WORKDIR/code/embeddings/logs"
mkdir -p "$LOG_DIR"

JOB_ID="${SLURM_JOB_ID:-local}"
LOG_FILE="$LOG_DIR/extract_finetuned_${JOB_ID}.out"

exec > "$LOG_FILE" 2>&1

echo "[$(date)] Starting finetuned_lora sliding-window extraction: ${JOB_ID}"
echo "Working directory: $WORKDIR"

cd "$WORKDIR"

$PYTHON code/embeddings/fine_tune_bert.py \
    --data-path /scratch/users/s214/lab3 \
    --split-json data/train_test_split.json \
    --output-dir data/embeddings \
    --layer last4_mean \
    --pooling mean \
    --window-size 256 \
    --stride 128 \
    --batch-size 4 \
    --skip-training \
    --checkpoint-dir checkpoints/lora_epoch8

echo "[$(date)] Done."
