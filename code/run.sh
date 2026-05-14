#!/usr/bin/env bash
# =============================================================================
# run.sh  —  Lab 3 full pipeline orchestrator (SCF / SLURM)
#
# Submits a dependency graph of SLURM jobs covering:
#   Stage 1  Simple embeddings      (BoW, Word2Vec, GloVe)        — CPU
#   Stage 2  BERT embeddings        (vanilla, pretrained, LoRA)   — GPU
#   Stage 3  Ridge regression       (all 8 methods × 2 subjects)  — CPU array
#   Stage 4  Analysis & figures     (stats, stability, check2)    — CPU
#   Stage 5  Interpretation         (SHAP / LIME)                 — CPU
#   Stage 6  LaTeX compilation      (method, stability, main)     — local
#
# Usage (from anywhere):
#   bash code/run.sh              # submit everything, skip completed stages
#   bash code/run.sh --force      # re-submit all stages regardless of existing outputs
#   bash code/run.sh --dry-run    # print what would be submitted without submitting
#
# After submission:
#   squeue --me
#   tail -f /scratch/users/shizhe_zhang/lab3/logs/<jobid>_<name>.out
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=/scratch/users/shizhe_zhang/conda/envs/lab3/bin/python
DATA=/scratch/users/s214/lab3
LOG_DIR=/scratch/users/shizhe_zhang/lab3/logs
N_TASKS=5          # parallel SLURM tasks per ridge array job

FORCE=false
DRY_RUN=false
for arg in "$@"; do
    [[ "$arg" == "--force"   ]] && FORCE=true
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
done

mkdir -p "$LOG_DIR"

# ── Colours ───────────────────────────────────────────────────────────────────
G="\033[0;32m"; Y="\033[0;33m"; C="\033[0;36m"; R="\033[0;31m"; N="\033[0m"
log()  { echo -e "${C}[pipeline]${N} $*"; }
ok()   { echo -e "${G}[skip]${N}    $*  (output exists)"; }
warn() { echo -e "${Y}[warn]${N}    $*"; }
err()  { echo -e "${R}[error]${N}   $*"; exit 1; }

# ── SLURM helpers ─────────────────────────────────────────────────────────────
# submit_low  [extra sbatch flags...] --wrap "CMD"
# submit_gpu  [extra sbatch flags...] --wrap "CMD"
# submit_after DEPS [extra sbatch flags...] --wrap "CMD"

_sbatch() {
    if $DRY_RUN; then
        echo "DRY-RUN: sbatch $*" >&2   # command to stderr — not captured by $(...)
        echo "99999"                    # fake job ID to stdout
    else
        sbatch --parsable "$@"
    fi
}

submit_low() {
    _sbatch \
        --partition=low \
        --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=08:00:00 \
        --output="$LOG_DIR/%j_%x.out" --error="$LOG_DIR/%j_%x.err" \
        "$@"
}

submit_gpu() {
    _sbatch \
        --partition=jsteinhardt --gres=gpu:A100:1 \
        --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=04:00:00 \
        --output="$LOG_DIR/%j_%x.out" --error="$LOG_DIR/%j_%x.err" \
        "$@"
}

submit_after() {
    local deps="$1"; shift
    _sbatch \
        --dependency="afterok:${deps}" --kill-on-invalid-dep=yes \
        --output="$LOG_DIR/%j_%x.out" --error="$LOG_DIR/%j_%x.err" \
        "$@"
}

submit_array_after() {
    # submit_array_after DEPS N_TASKS [extra flags...] --wrap "CMD"
    local deps="$1"; shift
    local n="$1";    shift
    _sbatch \
        --dependency="afterok:${deps}" --kill-on-invalid-dep=yes \
        --array="0-$((n-1))" \
        --output="$LOG_DIR/%A_%a_%x.out" --error="$LOG_DIR/%A_%a_%x.err" \
        "$@"
}

# ── Output-exists check ───────────────────────────────────────────────────────
# done FILE  → returns true if FILE exists and --force is not set
exists() { [[ -e "$1" ]] && ! $FORCE; }   # -e covers files and directories

# ── Dependency accumulator ────────────────────────────────────────────────────
# Join an array of job IDs with ':' for --dependency=afterok:...
join_deps() { local IFS=':'; echo "$*"; }

echo ""
log "Lab 3 pipeline  —  repo: $REPO"
log "Python : $PYTHON"
log "Log dir: $LOG_DIR"
$FORCE   && warn "--force: re-submitting all stages"
$DRY_RUN && warn "--dry-run: no jobs will actually be submitted"
echo ""

# =============================================================================
# STAGE 1 — Simple embeddings  (BoW / Word2Vec / GloVe)
# =============================================================================
log "${G}Stage 1${N}: Simple embeddings"

# BoW
EMBED_BOW="$REPO/data/embeddings/s2_train_bow_embeddings.npz"
if exists "$EMBED_BOW"; then
    ok "BoW embeddings"; J_BOW=""
else
    J_BOW=$(submit_low \
        --job-name=embed-bow --mem=16G --time=02:00:00 \
        --wrap="cd $REPO && $PYTHON code/embeddings/bow.py")
    log "BoW embeddings  → job $J_BOW"
fi

# Word2Vec
EMBED_W2V="$REPO/data/embeddings/s2_train_word2vec_embeddings.npz"
if exists "$EMBED_W2V"; then
    ok "Word2Vec embeddings"; J_W2V=""
else
    J_W2V=$(submit_low \
        --job-name=embed-w2v --mem=16G --time=02:00:00 \
        --wrap="cd $REPO && $PYTHON code/embeddings/word2vec.py")
    log "Word2Vec embeddings  → job $J_W2V"
fi

# GloVe
EMBED_GLOVE="$REPO/data/embeddings/s2_train_glove_embeddings.npy"
if exists "$EMBED_GLOVE"; then
    ok "GloVe embeddings"; J_GLOVE=""
else
    J_GLOVE=$(submit_low \
        --job-name=embed-glove --mem=16G --time=02:00:00 \
        --wrap="cd $REPO && $PYTHON code/embeddings/glove.py")
    log "GloVe embeddings  → job $J_GLOVE"
fi

# =============================================================================
# STAGE 2 — BERT embeddings
# =============================================================================
log "${G}Stage 2${N}: BERT embeddings"

# Vanilla BERT  (CPU — no sliding window, no fine-tuning)
EMBED_VANILLA="$REPO/data/embeddings/s2_train_vanilla_bert_embeddings.npz"
if exists "$EMBED_VANILLA"; then
    ok "Vanilla BERT embeddings"; J_VANILLA=""
else
    J_VANILLA=$(submit_low \
        --job-name=embed-vanilla --mem=8G --time=04:00:00 \
        --wrap="cd $REPO && \
            $PYTHON code/embeddings/pretrained_bert.py \
                --data-path $DATA \
                --split-json data/train_test_split.json \
                --output-dir data/embeddings \
                --layer last_hidden_state \
                --pooling mean \
                --window-size 0 \
                --batch-size 4 \
                --method-name vanilla_bert")
    log "Vanilla BERT embeddings  → job $J_VANILLA"
fi

# Pretrained BERT  (GPU — sliding window last4_mean)
EMBED_PBERT="$REPO/data/embeddings/s2_train_pretrained_bert_embeddings.npz"
if exists "$EMBED_PBERT"; then
    ok "Pretrained BERT embeddings"; J_PBERT=""
else
    J_PBERT=$(submit_gpu \
        --job-name=embed-pbert --time=02:00:00 \
        --wrap="cd $REPO && \
            $PYTHON code/embeddings/pretrained_bert.py \
                --data-path $DATA \
                --split-json data/train_test_split.json \
                --output-dir data/embeddings \
                --layer last4_mean \
                --pooling mean \
                --window-size 256 \
                --stride 128 \
                --batch-size 4")
    log "Pretrained BERT embeddings  → job $J_PBERT"
fi

# LoRA fine-tune  (GPU — produces checkpoints/lora_best + finetuned_lora embeddings)
EMBED_LORA="$REPO/data/embeddings/s2_train_finetuned_lora_embeddings.npz"
if exists "$EMBED_LORA"; then
    ok "LoRA fine-tune + embeddings"; J_LORA=""
else
    J_LORA=$(submit_gpu \
        --job-name=finetune-lora --mem=16G --time=04:00:00 \
        --wrap="cd $REPO && \
            $PYTHON code/embeddings/fine_tune_bert.py \
                --data-path $DATA \
                --split-json data/train_test_split.json \
                --output-dir data/embeddings \
                --epochs 8 --lr 2e-5 --batch-size 2 \
                --max-words-per-chunk 128 \
                --layer last4_mean --pooling mean \
                --window-size 256 --stride 128")
    log "LoRA fine-tune  → job $J_LORA"
fi

# mid4_mean layer extraction  (GPU — reuses lora_best checkpoint; depends on fine-tune)
EMBED_MID4="$REPO/data/embeddings_mid4_mean/s2_train_finetuned_lora_mid4_mean_embeddings.npz"
if exists "$EMBED_MID4"; then
    ok "LoRA mid4 embeddings"; J_MID4=""
else
    MID4_DEP="${J_LORA:-}"
    MID4_CMD="cd $REPO && \
        $PYTHON code/embeddings/fine_tune_bert.py \
            --data-path $DATA \
            --split-json data/train_test_split.json \
            --output-dir data/embeddings_mid4_mean \
            --layer mid4_mean --pooling mean \
            --window-size 256 --stride 128 \
            --batch-size 4 \
            --skip-training \
            --checkpoint-dir checkpoints/lora_best \
            --method-name finetuned_lora_mid4_mean"
    if [[ -n "$MID4_DEP" ]]; then
        J_MID4=$(submit_after "$MID4_DEP" \
            --partition=jsteinhardt --gres=gpu:A100:1 \
            --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=02:00:00 \
            --job-name=embed-mid4 \
            --wrap="$MID4_CMD")
    else
        J_MID4=$(submit_gpu --job-name=embed-mid4 --time=02:00:00 \
            --wrap="$MID4_CMD")
    fi
    log "LoRA mid4 embeddings  → job $J_MID4"
fi

# Improved LoRA  (GPU — independent retrain with sliding-window + whole-word masking)
EMBED_IMP="$REPO/data/embeddings_improved/s2_train_finetuned_lora_improved_embeddings.npz"
if exists "$EMBED_IMP"; then
    ok "Improved LoRA embeddings"; J_IMPROVED=""
else
    J_IMPROVED=$(submit_gpu \
        --job-name=embed-improved --mem=32G --time=04:00:00 \
        --wrap="cd $REPO && \
            $PYTHON code/embeddings/fine_tune_bert.py \
                --data-path $DATA \
                --split-json data/train_test_split.json \
                --output-dir data/embeddings_improved \
                --epochs 8 --lr 2e-5 --batch-size 2 \
                --layer last4_mean --pooling mean \
                --window-size 256 --stride 128 \
                --train-window-size 256 --train-stride 128 \
                --whole-word-masking --mlm-prob 0.15 \
                --lora-r 32 --lora-alpha 64 --lora-target-ffn \
                --checkpoint-dir checkpoints/lora_improved \
                --method-name finetuned_lora_improved")
    log "Improved LoRA embeddings  → job $J_IMPROVED"
fi

# =============================================================================
# STAGE 3 — Ridge regression  (8 methods × 2 subjects)
# =============================================================================
log "${G}Stage 3${N}: Ridge regression"

RIDGE_SH="$REPO/code/ridge_regression/run_ridge.sh"
MERGE_PY="$REPO/code/ridge_regression/merge_ridge_results.py"

# method → (embedding_dir, save_weights, embed_dep_varname)
declare -A EMB_DIRS SAVE_W EMBED_DEPS
EMB_DIRS[bow]="$REPO/data/embeddings"
EMB_DIRS[word2vec]="$REPO/data/embeddings"
EMB_DIRS[glove]="$REPO/data/embeddings"
EMB_DIRS[vanilla_bert]="$REPO/data/embeddings"
EMB_DIRS[pretrained_bert]="$REPO/data/embeddings"
EMB_DIRS[finetuned_lora]="$REPO/data/embeddings"
EMB_DIRS[finetuned_lora_improved]="$REPO/data/embeddings_improved"
EMB_DIRS[finetuned_lora_mid4_mean]="$REPO/data/embeddings_mid4_mean"

SAVE_W[bow]="no"
SAVE_W[word2vec]="yes"
SAVE_W[glove]="yes"
SAVE_W[vanilla_bert]="yes"
SAVE_W[pretrained_bert]="yes"
SAVE_W[finetuned_lora]="yes"
SAVE_W[finetuned_lora_improved]="yes"
SAVE_W[finetuned_lora_mid4_mean]="yes"

EMBED_DEPS[bow]="${J_BOW:-}"
EMBED_DEPS[word2vec]="${J_W2V:-}"
EMBED_DEPS[glove]="${J_GLOVE:-}"
EMBED_DEPS[vanilla_bert]="${J_VANILLA:-}"
EMBED_DEPS[pretrained_bert]="${J_PBERT:-}"
EMBED_DEPS[finetuned_lora]="${J_LORA:-}"
EMBED_DEPS[finetuned_lora_improved]="${J_IMPROVED:-}"
EMBED_DEPS[finetuned_lora_mid4_mean]="${J_MID4:-}"

ALL_METHODS=(bow word2vec glove vanilla_bert pretrained_bert \
             finetuned_lora finetuned_lora_improved finetuned_lora_mid4_mean)
ALL_SIDS=(s2 s3)

# Collect all merge job IDs (Stage 4 depends on them all)
MERGE_JOBS=()

for METHOD in "${ALL_METHODS[@]}"; do
    for SID in "${ALL_SIDS[@]}"; do
        case $SID in s2) SUBJ=subject2 ;; s3) SUBJ=subject3 ;; esac

        CC_FILE="$REPO/results/metrics/${SID}_${METHOD}_test_corrs.npy"
        if exists "$CC_FILE"; then
            ok "Ridge $SID/$METHOD"; continue
        fi

        SAVE_FLAG=""
        [[ "${SAVE_W[$METHOD]}" == "yes" ]] && SAVE_FLAG="--save-weights"

        EMBED_DEP="${EMBED_DEPS[$METHOD]}"
        COMMON_ARGS=(
            --partition=low
            --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G --time=08:00:00
            --job-name="ridge-${METHOD:0:7}-${SID}"
            --export="ALL,PYTHON=$PYTHON,TASK_COUNT=$N_TASKS"
        )
        RIDGE_ARGS=(
            --subject    "$SUBJ"
            --subject-id "$SID"
            --method     "$METHOD"
            --data-path  "$DATA"
            --embedding-dir "${EMB_DIRS[$METHOD]}"
            --split-path    "$REPO/data/train_test_split.json"
            --result-dir    "$REPO/results"
            $SAVE_FLAG
        )

        if [[ -n "$EMBED_DEP" ]]; then
            ARRAY_JOB=$(submit_array_after "$EMBED_DEP" "$N_TASKS" \
                "${COMMON_ARGS[@]}" \
                "$RIDGE_SH" "${RIDGE_ARGS[@]}")
        else
            ARRAY_JOB=$(_sbatch \
                --array="0-$((N_TASKS-1))" \
                --output="$LOG_DIR/%A_%a_%x.out" --error="$LOG_DIR/%A_%a_%x.err" \
                "${COMMON_ARGS[@]}" \
                "$RIDGE_SH" "${RIDGE_ARGS[@]}")
        fi
        log "Ridge $SID/$METHOD  → array $ARRAY_JOB"

        # Merge job chains after all array tasks succeed
        MERGE_JOB=$(submit_after "$ARRAY_JOB" \
            --partition=low \
            --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=00:30:00 \
            --job-name="merge-${METHOD:0:7}-${SID}" \
            --wrap="$PYTHON $MERGE_PY \
                --subject-id $SID \
                --method $METHOD \
                --result-dir $REPO/results \
                --n-tasks $N_TASKS")
        log "  Merge  $SID/$METHOD  → job $MERGE_JOB"

        MERGE_JOBS+=("$MERGE_JOB")
    done
done

# =============================================================================
# STAGE 4 — Analysis & figures
# =============================================================================
log "${G}Stage 4${N}: Analysis & figures"

if [[ ${#MERGE_JOBS[@]} -gt 0 ]]; then
    ALL_MERGE_DEP=$(join_deps "${MERGE_JOBS[@]}")
else
    ALL_MERGE_DEP=""
fi

# Helper: submit a Stage-4 job — with ridge dependency if any, otherwise directly
submit_stage4() {
    if [[ -n "$ALL_MERGE_DEP" ]]; then
        submit_after "$ALL_MERGE_DEP" "$@"
    else
        submit_low "$@"
    fi
}

# regression_stats.py  — 5 publication figures
STATS_FIG="$REPO/results/figures/regression_analysis/fig1_cc_distributions.pdf"
if exists "$STATS_FIG" && [[ ${#MERGE_JOBS[@]} -eq 0 ]]; then
    ok "Regression stats figures"
else
    J_STATS=$(submit_stage4 \
        --partition=low \
        --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=01:00:00 \
        --job-name=regr-stats \
        --wrap="cd $REPO && $PYTHON code/analysis/regression_stats.py")
    log "Regression stats  → job $J_STATS"
fi

# stability_check.py  (GloVe-based, Checks 1 + 3 + permutation test)
STAB_JSON="$REPO/results/stability/stability_summary.json"
if exists "$STAB_JSON" && [[ ${#MERGE_JOBS[@]} -eq 0 ]]; then
    ok "Stability checks (GloVe)"
else
    J_STAB=$(submit_stage4 \
        --partition=low \
        --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --time=06:00:00 \
        --job-name=stability \
        --wrap="cd $REPO/code && $PYTHON stability_check.py")
    log "Stability checks  → job $J_STAB"
fi

# stability_extended.py  (Checks 4-6; depends on stability_check finishing)
STAB_EXT_JSON="$REPO/results/stability/stability_extended_summary.json"
STAB_EXT_DEP="${J_STAB:-$ALL_MERGE_DEP}"
if exists "$STAB_EXT_JSON" && [[ ${#MERGE_JOBS[@]} -eq 0 ]]; then
    ok "Stability checks extended (4-6)"
else
    J_STAB_EXT=$(submit_stage4 \
        --partition=low \
        --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G --time=03:00:00 \
        --partition=low \
        --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G --time=03:00:00 \
        --job-name=stab-ext \
        --wrap="cd $REPO && $PYTHON code/analysis/stability_extended.py")
    log "Stability extended  → job $J_STAB_EXT"
fi

# stability.py  (cross-subject replication with best model)
CHECK2_JSON="$REPO/results/stability/check2_mid4_story_scores.json"
if exists "$CHECK2_JSON" && [[ ${#MERGE_JOBS[@]} -eq 0 ]]; then
    ok "Check 2 (LoRA mid4)"
else
    J_CHECK2=$(submit_stage4 \
        --partition=low \
        --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=01:00:00 \
        --job-name=check2-mid4 \
        --wrap="cd $REPO && $PYTHON code/analysis/stability.py")
    log "Check 2 mid4  → job $J_CHECK2"
fi

# =============================================================================
# STAGE 5 — Interpretation  (SHAP / LIME on LoRA mid4★, both subjects)
# =============================================================================
log "${G}Stage 5${N}: Interpretation (SHAP / LIME)"

INTERP_OUT="$REPO/results/interpretation/s3/finetuned_lora_mid4_mean"
if exists "$INTERP_OUT/lime" && [[ ${#MERGE_JOBS[@]} -eq 0 ]]; then
    ok "Interpretation (SHAP/LIME)"
else
    J_INTERP=$(submit_stage4 \
        --partition=low \
        --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G --time=04:00:00 \
        --job-name=interp \
        --wrap="cd $REPO && bash code/interpretation/submit_interpretation.sh")
    log "Interpretation  → job $J_INTERP"
fi

# =============================================================================
# STAGE 6 — LaTeX compilation  (local, after analysis jobs)
# =============================================================================
log "${G}Stage 6${N}: LaTeX compilation"

# Collect all analysis job IDs to build the final dependency
ANALYSIS_JOBS=()
for jid in "${J_STATS:-}" "${J_STAB:-}" "${J_STAB_EXT:-}" "${J_CHECK2:-}" "${J_INTERP:-}"; do
    [[ -n "$jid" ]] && ANALYSIS_JOBS+=("$jid")
done

if [[ ${#ANALYSIS_JOBS[@]} -gt 0 ]]; then
    ANALYSIS_DEP=$(join_deps "${ANALYSIS_JOBS[@]}")
    J_LATEX=$(submit_after "$ANALYSIS_DEP" \
        --partition=low \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:15:00 \
        --job-name=latex \
        --wrap="cd $REPO/report && \
            pdflatex -interaction=nonstopmode method.tex && \
            pdflatex -interaction=nonstopmode method.tex && \
            pdflatex -interaction=nonstopmode stability.tex && \
            pdflatex -interaction=nonstopmode stability.tex && \
            pdflatex -interaction=nonstopmode shap_lime.tex && \
            pdflatex -interaction=nonstopmode shap_lime.tex && \
            pdflatex -interaction=nonstopmode main.tex && \
            bibtex main && \
            pdflatex -interaction=nonstopmode main.tex && \
            pdflatex -interaction=nonstopmode main.tex && \
            echo 'PDFs written to report/'")
    log "LaTeX compile  → job $J_LATEX"
else
    # All analysis already done — compile locally right now
    log "Compiling LaTeX locally (all analysis already complete)..."
    cd "$REPO/report"
    for TEX in method stability shap_lime; do
        pdflatex -interaction=nonstopmode "${TEX}.tex" > /dev/null
        pdflatex -interaction=nonstopmode "${TEX}.tex" > /dev/null
        log "  ${TEX}.pdf  ✓"
    done
    pdflatex -interaction=nonstopmode main.tex > /dev/null
    bibtex main > /dev/null
    pdflatex -interaction=nonstopmode main.tex > /dev/null
    pdflatex -interaction=nonstopmode main.tex > /dev/null
    log "  main.pdf  ✓"
    cd "$REPO"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
log "All jobs submitted.  Monitor with:"
echo "    squeue --me"
echo "    tail -f $LOG_DIR/<jobid>_<name>.out"
echo ""
