#!/bin/bash
# Submit array ridge jobs for:
#   1. finetuned_lora_improved  (embeddings already in data/embeddings_improved/)
#   2. finetuned_lora_mid4_mean (embeddings already in data/embeddings_mid4_mean/)
#
# Run from repo root:
#   bash code/ridge_regression/submit_new_methods.sh

WORKDIR="/accounts/masters/shizhe_zhang/personal/lab-3-group-08"
SCRIPT="$WORKDIR/code/ridge_regression/run_new_methods_ridge.sh"

submit() {
    local method="$1" subject="$2" sid="$3" embdir="$4"
    echo "Submitting array[0-4]: method=$method subj=$sid"
    sbatch --array=0-4 \
        --export=ALL,METHOD="$method",SUBJECT="$subject",SUBJECT_ID="$sid",EMBEDDING_DIR="$embdir" \
        "$SCRIPT"
}

# improved LoRA — submit both subjects (10 tasks total)
submit finetuned_lora_improved subject2 s2 "$WORKDIR/data/embeddings_improved"
submit finetuned_lora_improved subject3 s3 "$WORKDIR/data/embeddings_improved"

# mid4_mean — submit both subjects (10 tasks total)
submit finetuned_lora_mid4_mean subject2 s2 "$WORKDIR/data/embeddings_mid4_mean"
submit finetuned_lora_mid4_mean subject3 s3 "$WORKDIR/data/embeddings_mid4_mean"

echo "All jobs submitted. After completion, merge with:"
echo "  python code/ridge_regression/merge_ridge_results.py --subject-id s2 --method finetuned_lora_improved"
echo "  python code/ridge_regression/merge_ridge_results.py --subject-id s3 --method finetuned_lora_improved"
echo "  python code/ridge_regression/merge_ridge_results.py --subject-id s2 --method finetuned_lora_mid4_mean"
echo "  python code/ridge_regression/merge_ridge_results.py --subject-id s3 --method finetuned_lora_mid4_mean"
