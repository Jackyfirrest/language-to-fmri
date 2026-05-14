#!/bin/bash
#SBATCH --job-name=lime
#SBATCH --output=logs/lime_%x_%j.out
#SBATCH --error=logs/lime_%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail

SUBJECT="$1"
SID="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RAW_DATA_ROOT="/scratch/users/s214/lab3"

cd "${PROJECT_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lab3

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "RAW_DATA_ROOT=${RAW_DATA_ROOT}"
echo "SUBJECT=${SUBJECT}"
echo "SID=${SID}"

python code/lime/lime_story_voxel_analysis.py \
  --subject "${SUBJECT}" \
  --subject-id "${SID}" \
  --stories onapproachtopluto marryamanwholoveshismother \
  --data-path "${RAW_DATA_ROOT}" \
  --embedding-dir "${PROJECT_ROOT}/data/embeddings" \
  --split-path "${PROJECT_ROOT}/data/train_test_split.json" \
  --result-dir "${PROJECT_ROOT}/results" \
  --figure-dir "${PROJECT_ROOT}/results/figures/lime" \
  --candidate-voxels 256 \
  --lime-samples 300 \
  --top-k-voxels 3
