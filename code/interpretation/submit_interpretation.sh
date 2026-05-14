#!/bin/bash
#SBATCH --job-name=lab3-interp
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/shizhe_zhang/lab3/logs/interp_%j_%x.out
#SBATCH --error=/scratch/users/shizhe_zhang/lab3/logs/interp_%j_%x.err

# Usage:
#   sbatch submit_interpretation.sh              # run both s3 and s2
#   sbatch submit_interpretation.sh --subject-id s3 --subject subject3
#   sbatch --export=ALL,SID=s2 submit_interpretation.sh

set -euo pipefail

PYTHON=/scratch/users/shizhe_zhang/conda/envs/lab3/bin/python
REPO=/accounts/masters/shizhe_zhang/personal/lab-3-group-08
SCRIPT=$REPO/code/interpretation/run_interpretation.py

mkdir -p /scratch/users/shizhe_zhang/lab3/logs

# Check that lime is available; install if missing
$PYTHON -c "import lime" 2>/dev/null || {
    echo "Installing lime into lab3 env …"
    /scratch/users/shizhe_zhang/conda/envs/lab3/bin/pip install lime --quiet
}

DATA=/scratch/users/s214/lab3

run_subject() {
    local SID=$1
    local SUBJ=$2
    echo "=============================="
    echo " Subject: $SUBJ ($SID)"
    echo "=============================="
    $PYTHON $SCRIPT \
        --subject       "$SUBJ" \
        --subject-id    "$SID" \
        --method        finetuned_lora_mid4_mean \
        --data-path     "$DATA" \
        --embedding-dir "$REPO/data/embeddings_mid4_mean" \
        --split-path    "$REPO/data/train_test_split.json" \
        --result-dir    "$REPO/results" \
        --output-dir    "$REPO/results/interpretation" \
        --stories onapproachtopluto marryamanwholoveshismother birthofanation \
        --n-candidate  256 \
        --top-k-voxels 3 \
        --word-window  48 \
        --peak-trs     5 \
        --lime-samples 300 \
        --n-features   12
}

# If called with explicit --subject-id, pass all args through
if [[ $# -gt 0 ]]; then
    $PYTHON $SCRIPT \
        --method        finetuned_lora_mid4_mean \
        --data-path     "$DATA" \
        --embedding-dir "$REPO/data/embeddings_mid4_mean" \
        --split-path    "$REPO/data/train_test_split.json" \
        --result-dir    "$REPO/results" \
        --output-dir    "$REPO/results/interpretation" \
        --stories onapproachtopluto marryamanwholoveshismother birthofanation \
        --n-candidate  256 \
        --top-k-voxels 3 \
        --word-window  48 \
        --peak-trs     5 \
        --lime-samples 300 \
        --n-features   12 \
        "$@"
else
    # Default: run s3 first (higher CCs), then s2
    run_subject s3 subject3
    run_subject s2 subject2
fi

echo "Done."
