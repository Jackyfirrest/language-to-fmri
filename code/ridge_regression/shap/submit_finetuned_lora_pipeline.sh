#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_finetuned_lora_pipeline.sh — one-command launcher for the full flow
#
# What it does:
#   1. Submits the finetuned_lora ridge regression as a SLURM array job
#   2. Waits for the array job to finish
#   3. Merges the partial ridge outputs
#   4. Extracts the top 5% voxels
#   5. Runs SHAP analysis on those voxels
#
# Example:
#   ./submit_finetuned_lora_pipeline.sh --subject-id s3 --array-size 10
#
# Environment overrides:
#   SUBJECT=subject3 SUBJECT_ID=s3 ARRAY_SIZE=10 CHUNK_SIZE=500 N_ALPHAS=10 \
#   CV_FRAC=0.2 SAVE_WEIGHTS=1 ./submit_finetuned_lora_pipeline.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Determine Python interpreter: prefer user's conda env if available
PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
    if [[ -x "${HOME}/.conda/envs/stat214/bin/python" ]]; then
        PYTHON="${HOME}/.conda/envs/stat214/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON="$(command -v python)"
    else
        echo "ERROR: no Python interpreter found. Set PYTHON env or install Python." >&2
        exit 1
    fi
fi

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

SUBJECT="${SUBJECT:-subject3}"
SUBJECT_ID="${SUBJECT_ID:-s3}"
METHOD="${METHOD:-finetuned_lora}"
ARRAY_SIZE="${ARRAY_SIZE:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-500}"
N_ALPHAS="${N_ALPHAS:-10}"
ALPHA_MIN_LOG10="${ALPHA_MIN_LOG10:-0.0}"
ALPHA_MAX_LOG10="${ALPHA_MAX_LOG10:-3.0}"
CV_FRAC="${CV_FRAC:-0.2}"
SAVE_WEIGHTS="${SAVE_WEIGHTS:-1}"
DATA_PATH="${DATA_PATH:-/ocean/projects/mth250011p/shared/215a/final_project/data}"
EMBEDDING_DIR="${EMBEDDING_DIR:-../../../data/embeddings}"
SPLIT_PATH="${SPLIT_PATH:-../../../data/train_test_split.json}"
RESULT_DIR="${RESULT_DIR:-../../../results}"
TOP_PERCENTILE="${TOP_PERCENTILE:-5}"
N_TOP_VOXELS="${N_TOP_VOXELS:-}"
TASK_COUNT="${TASK_COUNT:-$ARRAY_SIZE}"

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --subject SUBJECT        Full subject name (default: ${SUBJECT})
  --subject-id SUBJECT_ID  Subject ID (default: ${SUBJECT_ID})
  --array-size N           Number of SLURM array tasks (default: ${ARRAY_SIZE})
  --chunk-size N           Voxels per chunk (default: ${CHUNK_SIZE})
  --n-alphas N             Number of alphas to try (default: ${N_ALPHAS})
  --cv-frac F              Validation fraction for dense ridge (default: ${CV_FRAC})
  --top-percentile P       Top percentile to keep (default: ${TOP_PERCENTILE})
  --n-top-voxels N         Limit SHAP analysis to N voxels
  --no-save-weights        Disable saving weights for SHAP
  --help                   Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subject)
            SUBJECT="$2"
            shift 2
            ;;
        --subject-id)
            SUBJECT_ID="$2"
            shift 2
            ;;
        --array-size)
            ARRAY_SIZE="$2"
            TASK_COUNT="$2"
            shift 2
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --n-alphas)
            N_ALPHAS="$2"
            shift 2
            ;;
        --cv-frac)
            CV_FRAC="$2"
            shift 2
            ;;
        --top-percentile)
            TOP_PERCENTILE="$2"
            shift 2
            ;;
        --n-top-voxels)
            N_TOP_VOXELS="$2"
            shift 2
            ;;
        --no-save-weights)
            SAVE_WEIGHTS=0
            shift 1
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

LOG_DIR="$WORKDIR/logs"
mkdir -p "$LOG_DIR"

cd "$WORKDIR"

echo "Submitting finetuned_lora ridge array job..."
SBATCH_CMD=(
    sbatch
    --parsable
    --array=0-$((ARRAY_SIZE - 1))
    "$WORKDIR/run_finetuned_lora_parallel.sh"
)

export SUBJECT SUBJECT_ID METHOD ARRAY_SIZE CHUNK_SIZE N_ALPHAS ALPHA_MIN_LOG10 ALPHA_MAX_LOG10 CV_FRAC SAVE_WEIGHTS DATA_PATH EMBEDDING_DIR SPLIT_PATH RESULT_DIR TASK_COUNT
JOB_ID="$(${SBATCH_CMD[@]})"
echo "Submitted array job: ${JOB_ID}"

echo "Waiting for SLURM array job to finish..."
WAIT_CMD="sacct -j ${JOB_ID} --format=JobID,State --noheader"
if ! command -v sacct >/dev/null 2>&1; then
    WAIT_CMD="squeue -j ${JOB_ID}"
fi

while true; do
    if command -v sacct >/dev/null 2>&1; then
        STATES="$(${WAIT_CMD} | awk 'NF {print $2}' | tr '\n' ' ' )"
        if [[ -n "$STATES" ]] && [[ "$STATES" != *"RUNNING"* ]] && [[ "$STATES" != *"PENDING"* ]]; then
            break
        fi
    else
        if ! squeue -j "$JOB_ID" >/dev/null 2>&1; then
            break
        fi
    fi
    sleep 60
done

echo "Array job complete, merging results..."
"$PYTHON" "$WORKDIR/merge_finetuned_lora_results.py" --subject-id "$SUBJECT_ID" --n-tasks "$ARRAY_SIZE" --result-dir "$RESULT_DIR"

echo "Extracting top voxels..."
"$PYTHON" "$WORKDIR/extract_top_voxels.py" --subject-id "$SUBJECT_ID" --method "$METHOD" --top-percentile "$TOP_PERCENTILE" --result-dir "$RESULT_DIR"

echo "Running SHAP analysis..."
SHAP_CMD=(
    "$PYTHON" "$WORKDIR/shap_analysis.py"
    --subject-id "$SUBJECT_ID"
    --method "$METHOD"
    --top-percentile "$TOP_PERCENTILE"
    --result-dir "$RESULT_DIR"
)
if [[ -n "$N_TOP_VOXELS" ]]; then
    SHAP_CMD+=(--n-top-voxels "$N_TOP_VOXELS")
fi
"${SHAP_CMD[@]}"

echo "All done."
