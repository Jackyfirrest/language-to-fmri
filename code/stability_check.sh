#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# stability_check.sh — SLURM batch job for stability_check.py
#
# Submit:
#   cd lab-3-group-08/code
#   sbatch stability_check.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/stability_check_<JOBID>.out
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J stability_check
#SBATCH --partition=low
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
LOG_DIR="${WORKDIR}/logs"
mkdir -p "$LOG_DIR"

JOB_ID="${SLURM_JOB_ID:-local}"
LOG_FILE="${LOG_DIR}/stability_check_${JOB_ID}.out"
exec > "$LOG_FILE" 2>&1

echo "[$(date)] Node: $(hostname)"
echo "[$(date)] Job ID: ${JOB_ID}"
echo "[$(date)] Working dir: ${WORKDIR}"

# ── Activate conda environment ────────────────────────────────────────────────
if [[ -x "${HOME}/.conda/envs/stat214/bin/python" ]]; then
    PYTHON="${HOME}/.conda/envs/stat214/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    echo "ERROR: no Python interpreter found" >&2
    exit 1
fi
echo "[$(date)] Python: ${PYTHON}"
$PYTHON --version

# ── Run ───────────────────────────────────────────────────────────────────────
cd "${WORKDIR}"
$PYTHON stability_check.py

echo "[$(date)] Done."
