#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_finetune_bert.sh — SLURM batch script for fine_tune_bert.py
#
# Submit from the repo root on SCF:
#   sbatch code/embeddings/run_finetune_bert.sh
#
# Check GPU availability first:
#   sinfo -s
#   sinfo -o "%P %G" | grep gpu
#
# After job completes, upload embeddings to HuggingFace with hf_sync.py
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J finetune_bert_lora
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A100:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PYTHON="${PYTHON:-/scratch/users/shizhe_zhang/conda/envs/lab3/bin/python}"

LOG_DIR="$WORKDIR/code/embeddings/logs"
mkdir -p "$LOG_DIR"

JOB_ID="${SLURM_JOB_ID:-local}"
LOG_FILE="$LOG_DIR/finetune_bert_${JOB_ID}.out"

exec > "$LOG_FILE" 2>&1

echo "[$(date)] Starting fine-tune BERT LoRA job: ${JOB_ID}"
echo "Working directory: $WORKDIR"

cd "$WORKDIR"

$PYTHON code/embeddings/fine_tune_bert.py \
    --data-path /scratch/users/s214/lab3 \
    --split-json data/train_test_split.json \
    --output-dir data/embeddings \
    --epochs 8 \
    --lr 2e-5 \
    --batch-size 2 \
    --max-words-per-chunk 128 \
    --layer last4_mean \
    --pooling mean

echo "[$(date)] Done."
