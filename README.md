# Connecting Language to fMRI

This repository contains a course lab project on predicting fMRI responses from language representations. The project compares sparse lexical features, static word embeddings, and several BERT-based variants, then fits voxelwise ridge regression models and analyzes both predictive performance and interpretation outputs.

The repository is best understood as a preserved research project with runnable components, a final report, and helper scripts for managing large artifacts. It is not a fully portable one-command reproduction package yet, because parts of the pipeline still assume the original SLURM/Scratch environment used during the course project.

## Project goals

- build language embeddings from narrated story stimuli
- predict voxelwise fMRI activity for two subjects
- compare simple and contextual embedding families
- evaluate cross-subject stability and method progression
- inspect model behavior with SHAP and LIME

## Methods covered

The codebase references the following embedding/model variants:

- `bow`
- `word2vec`
- `glove`
- `vanilla_bert`
- `pretrained_bert`
- `finetuned_lora`
- `finetuned_lora_improved`
- `finetuned_lora_mid4_mean`

These methods are used downstream by the ridge regression and analysis scripts.

## Repository layout

```text
.
|-- code/
|   |-- analysis/             # regression summaries, plots, and stability analysis
|   |-- eda/                  # exploratory analysis utilities
|   |-- embeddings/           # BoW, Word2Vec, GloVe, BERT, and LoRA scripts
|   |-- interpretation/       # combined SHAP + LIME pipeline
|   |-- lime/                 # story-level LIME analysis entry points
|   |-- provided/             # course-provided preprocessing / fine-tuning helpers
|   |-- ridge_regression/     # voxelwise ridge training and shard merging
|   |-- ridge_utils/          # ridge solver and stimulus/data utilities
|   |-- testing/              # lightweight validation scripts
|   |-- preprocessing_utils.py
|   |-- run.sh                # full SLURM pipeline orchestrator
|   `-- stability_check.py    # main stability script used by the pipeline
|-- data/                     # tracked metadata plus ignored large inputs/embeddings
|-- report/
|   |-- figures/              # exported report figures
|   |-- lab3_tex_files/       # LaTeX sources and figure copies
|   |-- collaboration.txt     # original project collaboration note
|   `-- lab3.pdf              # final report PDF
|-- results/                  # ignored generated metrics, models, and interpretation outputs
|-- .gitignore
|-- environment.yml
|-- hf_sync.py                # Hugging Face upload/download helper for large artifacts
`-- README.md
```

## Important files

Some especially useful entry points:

- `code/run.sh`: submits the full multi-stage pipeline on a SLURM cluster
- `code/embeddings/bow.py`: generates bag-of-words embeddings
- `code/embeddings/word2vec.py`: generates Word2Vec embeddings
- `code/embeddings/glove.py`: generates GloVe embeddings
- `code/embeddings/pretrained_bert.py`: extracts vanilla or pretrained BERT embeddings
- `code/embeddings/fine_tune_bert.py`: fine-tunes BERT with LoRA and extracts embeddings
- `code/ridge_regression/run_ridge.py`: trains ridge models for one subject/method/task shard
- `code/ridge_regression/merge_ridge_results.py`: merges per-task ridge outputs
- `code/analysis/regression_stats.py`: builds summary figures for regression results
- `code/analysis/stability.py` and `code/analysis/stability_extended.py`: cross-subject and extended stability checks
- `code/interpretation/run_interpretation.py`: end-to-end SHAP + LIME interpretation workflow
- `code/lime/lime_story_voxel_analysis.py`: focused LIME runner for selected stories/voxels

## Environment setup

Create the conda environment with:

```bash
conda env create -f environment.yml
conda activate lab3
```

The environment includes the main research dependencies used in the project, including:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `torch`
- `transformers`
- `gensim`
- `lime`
- `huggingface-hub`

If you plan to use `hf_sync.py`, also make sure `python-dotenv` is installed, since that helper reads credentials from a local `.env` file.

## Data and artifact conventions

Large data products are intentionally excluded from git by `.gitignore`.

- `data/` is used for raw inputs, train/test metadata, and generated embeddings
- `results/` is used for metrics, model shards, merged outputs, and interpretation results
- `checkpoints/` is used for fine-tuned LoRA checkpoints and is also ignored

Tracked metadata currently includes the train/test split file used by multiple scripts:

- `data/train_test_split.json`

Common naming patterns used throughout the code:

| Artifact | Pattern |
| --- | --- |
| Train/test embeddings | `{sid}_{split}_{method}_embeddings.npy` or `.npz` |
| Ridge metrics | `results/metrics/{sid}_{method}_test_corrs.npy` |
| Ridge model shards | `results/models/{sid}_{method}/...` |
| Subject ids | `s2`, `s3` |

## Typical workflow

The overall project flow is:

1. preprocess story-aligned inputs and train/test splits
2. generate embeddings for each method
3. fit voxelwise ridge regression models
4. merge shard outputs and summarize predictive performance
5. run stability checks across subjects and methods
6. run SHAP/LIME interpretation for selected models
7. compile the report figures and LaTeX outputs

## Running the full pipeline

The main orchestrator is:

```bash
bash code/run.sh
```

Useful options:

```bash
bash code/run.sh --dry-run
bash code/run.sh --force
```

`code/run.sh` is designed for the original SCF/SLURM environment and submits a staged job graph:

1. simple embeddings
2. BERT-based embeddings
3. ridge regression arrays
4. regression and stability analysis
5. interpretation
6. LaTeX compilation

Because that script contains cluster-specific paths such as scratch directories, partitions, and a fixed Python executable, it will usually require local editing before it can run on another machine or another cluster.

## Running pieces locally

For local experimentation, it is usually easier to run individual scripts rather than the full cluster pipeline.

Examples:

```bash
python code/embeddings/bow.py
python code/embeddings/word2vec.py
python code/embeddings/glove.py
```

```bash
python code/embeddings/pretrained_bert.py --help
python code/embeddings/fine_tune_bert.py --help
```

```bash
python code/ridge_regression/run_ridge.py \
  --subject subject2 \
  --subject-id s2 \
  --method glove \
  --data-path <path-to-fmri-data> \
  --embedding-dir data/embeddings \
  --split-path data/train_test_split.json \
  --result-dir results
```

```bash
python code/ridge_regression/merge_ridge_results.py \
  --subject-id s2 \
  --method glove \
  --result-dir results \
  --n-tasks 1
```

Many scripts expose additional CLI options through `--help`, especially:

- `code/embeddings/pretrained_bert.py`
- `code/embeddings/fine_tune_bert.py`
- `code/ridge_regression/run_ridge.py`
- `code/analysis/stability_extended.py`
- `code/interpretation/run_interpretation.py`
- `code/lime/lime_story_voxel_analysis.py`

## Reproducibility caveats

There are a few important limitations to keep in mind:

- some shell scripts assume SLURM and specific cluster partitions
- several paths in orchestration scripts point to the original scratch storage layout
- raw fMRI inputs are not bundled in this repository
- generated embeddings, model checkpoints, and results are intentionally ignored because they are large
- a few scripts were written for project delivery rather than polished general-purpose reuse

In practice, the easiest path for reuse is:

1. create the conda environment
2. place the required raw data in your own accessible storage location
3. adapt paths in the relevant shell or Python entry points
4. run the embedding, ridge, analysis, or interpretation stages individually

## Hugging Face sync helper

`hf_sync.py` is a utility for moving large artifacts to and from a Hugging Face dataset repository.

Supported commands include:

- `upload`
- `upload-folder`
- `download`
- `download-folder`
- `delete`
- `delete-pattern`

It expects a local `.env` file at the project root with:

```env
HF_TOKEN=...
HF_REPO_ID=stat214-group08/lab3
```

Example usage:

```bash
python hf_sync.py upload-folder data/embeddings data/embeddings
python hf_sync.py download-folder data/embeddings data
```

## Report

The final written report is included here:

- [lab3.pdf](/C:/Users/jackyfirst/Downloads/lab-3-group-08/report/lab3.pdf)
- [LaTeX sources](/C:/Users/jackyfirst/Downloads/lab-3-group-08/report/lab3_tex_files)
- [report figures](/C:/Users/jackyfirst/Downloads/lab-3-group-08/report/figures)
