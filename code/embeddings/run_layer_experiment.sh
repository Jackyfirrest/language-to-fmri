#!/bin/bash
# Test mid4_mean and all_mean layer extractions using the existing lora_best checkpoint.
# No retraining — just re-extracts embeddings with different layer aggregations and
# runs ridge regression for both subjects.
#
# Submit from repo root:
#   sbatch code/embeddings/run_layer_experiment.sh
#SBATCH -J lora_layer_experiment
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
exec > "$LOG_DIR/layer_experiment_${JOB_ID}.out" 2>&1

cd "$WORKDIR"

for LAYER in mid4_mean all_mean; do
    echo "[$(date)] ── Extracting layer=$LAYER ──"
    $PYTHON code/embeddings/fine_tune_bert.py \
        --data-path  /scratch/users/s214/lab3 \
        --split-json data/train_test_split.json \
        --output-dir "data/embeddings_${LAYER}" \
        --layer      "$LAYER" \
        --pooling    mean \
        --window-size 256 \
        --stride     128 \
        --batch-size 4 \
        --skip-training \
        --checkpoint-dir checkpoints/lora_best \
        --method-name    "finetuned_lora_${LAYER}"

    for SID in s2 s3; do
        case $SID in s2) SUBJ=subject2 ;; s3) SUBJ=subject3 ;; esac
        echo "[$(date)] ── Ridge: $SUBJ  layer=$LAYER ──"
        $PYTHON code/ridge_regression/run_ridge.py \
            --subject    "$SUBJ" \
            --subject-id "$SID" \
            --method     "finetuned_lora_${LAYER}" \
            --data-path  /scratch/users/s214/lab3 \
            --embedding-dir "$WORKDIR/data/embeddings_${LAYER}" \
            --split-path    "$WORKDIR/data/train_test_split.json" \
            --result-dir    "$WORKDIR/results" \
            --n-alphas 10 --alpha-min-log10 0.0 --alpha-max-log10 3.0 \
            --chunk-size 500
    done
done

echo "[$(date)] ── Layer experiment done ──"
