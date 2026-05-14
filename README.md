# Connecting Language to fMRI

This repository contains my lab project on predicting fMRI responses from language embeddings. The workflow compares bag-of-words, static embeddings, and several BERT-based representations, then fits voxelwise ridge regression models and analyzes the results with stability checks plus SHAP/LIME interpretation.

The codebase was originally developed in a course team setting and is preserved here as my personal public copy, with the report and project artifacts kept for reference.

## What is in this repo

- Embedding pipelines for `bow`, `word2vec`, `glove`, `vanilla_bert`, `pretrained_bert`, and LoRA-based BERT variants
- Ridge regression scripts for voxelwise encoding models
- Analysis scripts for regression summary figures and cross-subject stability
- Interpretation scripts for SHAP and LIME
- The final written report and report figures

## Repository layout

```text
.
|-- code/
|   |-- analysis/           # regression summaries and stability analysis
|   |-- embeddings/         # embedding extraction and fine-tuning scripts
|   |-- interpretation/     # SHAP/LIME pipeline wrappers
|   |-- lime/               # story-level LIME analysis
|   |-- provided/           # course-provided utilities
|   |-- ridge_regression/   # voxelwise ridge modeling
|   |-- ridge_utils/        # ridge/data helper code
|   |-- testing/            # small validation scripts
|   |-- preprocessing_utils.py
|   `-- run.sh              # end-to-end cluster pipeline
|-- data/                   # ignored; local/raw data and generated embeddings
|-- results/                # ignored; generated metrics, models, and figures
|-- report/
|   |-- lab3_tex_files/     # LaTeX sources
|   |-- figures/            # report figures kept in-repo
|   |-- collaboration.txt
|   `-- lab3.pdf
|-- environment.yml
|-- hf_sync.py              # helper for Hugging Face uploads/downloads
|-- LICENSE
`-- README.md
```

## Environment

Create the conda environment with:

```bash
conda env create -f environment.yml
conda activate lab3
```

The environment includes the main packages used in the project, including `numpy`, `scikit-learn`, `matplotlib`, `torch`, `transformers`, `gensim`, `lime`, and `huggingface-hub`.

## Data and outputs

Large project data is intentionally not committed.

- `data/` is for raw inputs, split files, and generated embeddings
- `results/` is for metrics, figures, interpretation outputs, and model artifacts
- some scripts also expect large scratch-space outputs on cluster storage

Common file naming patterns:

| Type | Pattern |
| --- | --- |
| Embeddings | `{sid}_{split}_{method}_embeddings.npz` or `.npy` |
| Metrics | `results/metrics/{sid}_{method}_test_corrs.npy` |
| Subject IDs | `s2`, `s3` |

Methods referenced throughout the repo:

- `bow`
- `word2vec`
- `glove`
- `vanilla_bert`
- `pretrained_bert`
- `finetuned_lora`
- `finetuned_lora_improved`
- `finetuned_lora_mid4_mean`

## Running the pipeline

The main orchestrator is `code/run.sh`:

```bash
bash code/run.sh
```

Useful options:

```bash
bash code/run.sh --dry-run
bash code/run.sh --force
```

The pipeline is organized into these stages:

1. Generate simple embeddings
2. Generate BERT-based embeddings
3. Run ridge regression
4. Produce summary and stability analyses
5. Run SHAP/LIME interpretation
6. Compile LaTeX report outputs

## Important note about reproducibility

Some pipeline pieces are still tied to the original SCF/SLURM environment and absolute scratch paths inside `code/run.sh` and related shell scripts. That means the repository is best understood as a preserved research project plus report, not yet a fully portable one-command reproduction setup on any machine.

For local reuse, the most practical path is usually:

- create the conda environment
- place the required data under `data/`
- run individual Python scripts from `code/analysis/`, `code/embeddings/`, or `code/ridge_regression/`
- adapt cluster-specific paths in shell scripts if you want the full pipeline

## Report

The final report PDF is included at [report/lab3.pdf](/C:/Users/jackyfirst/Downloads/lab-3-group-08/report/lab3.pdf), and the LaTeX sources are in [report/lab3_tex_files](/C:/Users/jackyfirst/Downloads/lab-3-group-08/report/lab3_tex_files).

## Notes

- `report/collaboration.txt` is intentionally kept as part of the original project record.
- `hf_sync.py` is a helper script for moving assets to or from Hugging Face storage.
