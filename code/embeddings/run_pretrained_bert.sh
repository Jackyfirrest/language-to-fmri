#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_pretrained_bert.sh — SLURM batch script for pretrained_bert.py
#
# Generates sliding-window BERT embeddings (last4_mean, window=256, stride=128)
# and saves them as:
#   data/embeddings/s{2,3}_{train,test}_pretrained_bert_embeddings.npz
#
# Submit from the repo root on SCF:
#   sbatch code/embeddings/run_pretrained_bert.sh
#
# After job completes, upload embeddings to HuggingFace with hf_sync.py:
#   python hf_sync.py upload data/embeddings/s2_train_pretrained_bert_embeddings.npz ...
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J pretrained_bert_embed
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
LOG_FILE="$LOG_DIR/pretrained_bert_${JOB_ID}.out"

exec > "$LOG_FILE" 2>&1

echo "[$(date)] Starting pretrained BERT embedding extraction: ${JOB_ID}"
echo "Working directory: $WORKDIR"

cd "$WORKDIR"

$PYTHON code/embeddings/pretrained_bert.py \
    --data-path /scratch/users/s214/lab3 \
    --split-json data/train_test_split.json \
    --output-dir data/embeddings \
    --layer last4_mean \
    --pooling mean \
    --window-size 256 \
    --stride 128 \
    --batch-size 4

echo "[$(date)] Done."
