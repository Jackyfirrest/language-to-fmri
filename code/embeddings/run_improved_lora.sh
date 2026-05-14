#!/bin/bash
# Retrain LoRA with three fixes applied together:
#   1. 256-word sliding window training chunks (matches inference context)
#   2. Whole-word masking (mask all subwords of a word together)
#   3. LoRA also applied to FFN intermediate/output layers
# Then extract embeddings and run ridge regression for both subjects.
#
# Submit from repo root:
#   sbatch code/embeddings/run_improved_lora.sh
#SBATCH -J improved_lora
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A100:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

WORKDIR="/accounts/masters/shizhe_zhang/personal/lab-3-group-08"
PYTHON="/scratch/users/shizhe_zhang/conda/envs/lab3/bin/python"
LOG_DIR="$WORKDIR/code/embeddings/logs"
CKPT_DIR="$WORKDIR/checkpoints/lora_improved"
mkdir -p "$LOG_DIR" "$CKPT_DIR"
JOB_ID="${SLURM_JOB_ID:-local}"
exec > "$LOG_DIR/improved_lora_${JOB_ID}.out" 2>&1

cd "$WORKDIR"

echo "[$(date)] ── Step 1: retrain with sliding-window + whole-word masking + FFN LoRA ──"
$PYTHON code/embeddings/fine_tune_bert.py \
    --data-path   /scratch/users/s214/lab3 \
    --split-json  data/train_test_split.json \
    --output-dir  data/embeddings_improved \
    --epochs      8 \
    --lr          2e-5 \
    --batch-size  2 \
    --layer       last4_mean \
    --pooling     mean \
    --window-size 256 \
    --stride      128 \
    --train-window-size 256 \
    --train-stride      128 \
    --whole-word-masking \
    --mlm-prob    0.15 \
    --lora-r      32 \
    --lora-alpha  64 \
    --lora-target-ffn \
    --checkpoint-dir "$CKPT_DIR" \
    --method-name    finetuned_lora_improved

echo "[$(date)] ── Step 2 & 3: ridge regression s2 and s3 ──"
for SID in s2 s3; do
    case $SID in s2) SUBJ=subject2 ;; s3) SUBJ=subject3 ;; esac
    echo "[$(date)] Ridge: $SUBJ"
    $PYTHON code/ridge_regression/run_ridge.py \
        --subject    "$SUBJ" \
        --subject-id "$SID" \
        --method     finetuned_lora_improved \
        --data-path  /scratch/users/s214/lab3 \
        --embedding-dir "$WORKDIR/data/embeddings_improved" \
        --split-path    "$WORKDIR/data/train_test_split.json" \
        --result-dir    "$WORKDIR/results" \
        --n-alphas 10 --alpha-min-log10 0.0 --alpha-max-log10 3.0 \
        --chunk-size 500
done

echo "[$(date)] ── All done ──"
