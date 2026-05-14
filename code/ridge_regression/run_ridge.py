#!/usr/bin/env python3
"""
Unified ridge regression pipeline for fMRI voxel prediction.

Two solvers, selected automatically by method:
  - Dense (word2vec, glove, bert, ...): SVDRidgeSolver
      Implements the SVD trick from STAT 230A notes / Huth lab:
        β̂(λ) = V · diag(dᵢ/(dᵢ² + λ)) · Uᵀ · Y
      SVD of X is computed ONCE. Trying K alphas then costs K cheap
      diagonal updates instead of K matrix inversions.
      Gives per-voxel alpha selection at no extra cost.

  - Sparse (bow): sparse_ridge solver
      BoW has a huge sparse feature matrix — dense SVD is not viable.
      Uses LU decomposition via scipy, designed for sparse X.
      Uses a single global alpha selected on a random voxel sample.

Usage (interactive):
    python run_ridge.py --subject subject2 --subject-id s2 --method glove
    python run_ridge.py --subject subject3 --subject-id s3 --method bert

Usage (SLURM array — splits voxel chunks across tasks):
    sbatch --array=0-9 run_ridge.sh --subject subject2 --subject-id s2 --method word2vec
    # after all tasks finish:
    python merge_ridge_results.py --subject-id s2 --method word2vec --n-tasks 10
"""

import argparse
import gc
import json
import os
import pickle
import sys

import numpy as np
from scipy import sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ridge_utils.sparse_ridge import choose_alpha_sparse, ridge_corrs_sparse

# Methods that use the dense SVD solver — add new embedding names here as needed
DENSE_METHODS = {'word2vec', 'glove', 'bert', 'bert_finetuned', 'bert_lora',
                 'pretrained_bert', 'finetuned_lora', 'vanilla_bert',
                 'finetuned_lora_improved', 'finetuned_lora_mid4_mean',
                 'finetuned_lora_all_mean'}


# ── helpers ───────────────────────────────────────────────────────────────────

def _zscore_cols(arr):
    """Z-score each column (voxel/feature) independently."""
    arr = np.asarray(arr, dtype=np.float32)
    mu    = arr.mean(axis=0, keepdims=True)
    sigma = arr.std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    return (arr - mu) / sigma


def _col_corr(y_true, y_pred):
    """Pearson correlation between matching columns of two matrices."""
    return (_zscore_cols(y_true) * _zscore_cols(y_pred)).mean(axis=0).astype(np.float32)


def load_embedding(path):
    """Load embedding from .npy or .npz (numpy or scipy sparse)."""
    if path.endswith('.npy'):
        return np.asarray(np.load(path, mmap_mode='r'), dtype=np.float32)
    try:                                         # scipy sparse .npz
        return sp.load_npz(path).tocsr().astype(np.float32)
    except Exception:
        pass
    data = np.load(path, allow_pickle=False)     # numpy .npz
    key  = 'X' if 'X' in data.files else data.files[0]
    arr  = np.asarray(data[key], dtype=np.float32)
    if (1.0 - np.count_nonzero(arr) / arr.size) >= 0.80:
        return sp.csr_matrix(arr)
    return arr


def zscore_X(X_train, X_test):
    """Z-score dense X using train statistics. Skips sparse matrices."""
    if sp.issparse(X_train):
        return X_train, X_test
    mu    = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    return (X_train - mu) / sigma, (X_test - mu) / sigma


def find_embedding(embedding_dir, sid, split, method):
    for ext in ('npy', 'npz'):
        p = os.path.join(embedding_dir, f'{sid}_{split}_{method}_embeddings.{ext}')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f'No embedding found for {sid}/{split}/{method} in {embedding_dir}'
    )


def load_fmri_chunk(story_maps, stories, lengths, total_rows, voxel_idx):
    """Load a voxel subset from fMRI mmaps with per-story z-scoring.

    Y is already pre-trimmed on disk — no row slicing applied here.
    Per-story z-scoring removes baseline differences between scan runs.
    """
    out = np.empty((total_rows, len(voxel_idx)), dtype=np.float32)
    offset = 0
    for story, n in zip(stories, lengths):
        block = np.asarray(story_maps[story][:, voxel_idx], dtype=np.float32)
        np.nan_to_num(block, nan=0.0, copy=False)
        mu    = block.mean(axis=0, keepdims=True)
        sigma = block.std(axis=0, keepdims=True)
        sigma[sigma < 1e-6] = 1.0
        out[offset:offset + n] = (block - mu) / sigma
        offset += n
    return out


# ── dense SVD solver ──────────────────────────────────────────────────────────

class SVDRidgeSolver:
    """Precomputes SVD of X once, then solves for any Y and any alpha cheaply.

    Implements the workflow from STAT 230A notes (page 6):
      1. Standardize X  (done externally via zscore_X)
      2. Compute SVD of X once  <- this class does this step
      3. For each λ: diagonal update dᵢ/(dᵢ² + λ) — nearly free
      4. Pick best λ per voxel by cross-validation

    Two SVDs are precomputed once, outside the voxel chunk loop:
      cv_*   : SVD of X_train_cv (80% of train) — for per-voxel alpha selection
      full_* : SVD of X_train_full               — for final model weights
    """

    def __init__(self, X_train, X_test, cv_frac=0.2, singcutoff=1e-10):
        n      = X_train.shape[0]
        n_val  = max(1, int(n * cv_frac))
        self._n_cv  = n - n_val
        self._n_val = n_val
        self._X_test = np.asarray(X_test, dtype=np.float32)

        X_cv  = np.asarray(X_train[:self._n_cv], dtype=np.float32)
        X_val = np.asarray(X_train[self._n_cv:], dtype=np.float32)

        print(f'  SVD (cv split  {X_cv.shape})  ... ', end='', flush=True)
        U, S, Vh = np.linalg.svd(X_cv, full_matrices=False)
        good = S > singcutoff
        self._U_cv    = U[:, good].astype(np.float32)
        self._S_cv    = S[good].astype(np.float32)
        self._Vh_cv   = Vh[good].astype(np.float32)
        self._X_val   = X_val
        print('done')

        print(f'  SVD (full train {X_train.shape}) ... ', end='', flush=True)
        X_full = np.asarray(X_train, dtype=np.float32)
        U, S, Vh = np.linalg.svd(X_full, full_matrices=False)
        good = S > singcutoff
        self._U_full  = U[:, good].astype(np.float32)
        self._S_full  = S[good].astype(np.float32)
        self._Vh_full = Vh[good].astype(np.float32)
        print('done')

    def select_alpha(self, Y_train_full, alphas):
        """Per-voxel alpha selection using the cv split.

        SVD already computed — each alpha is just a diagonal update (cheap).
        Returns best alpha per voxel, shape (M,).
        """
        Y_cv  = Y_train_full[:self._n_cv]
        Y_val = Y_train_full[self._n_cv:]

        UR    = (self._U_cv.T @ Y_cv).astype(np.float32)   # (r, M)
        PVh_val = (self._X_val @ self._Vh_cv.T).astype(np.float32)  # (T_val, r)
        zY_val  = _zscore_cols(Y_val)

        best_alpha = np.full(Y_cv.shape[1], alphas[0], dtype=np.float32)
        best_corr  = np.full(Y_cv.shape[1], -np.inf,  dtype=np.float32)

        for alpha in alphas:
            D    = self._S_cv / (self._S_cv**2 + np.float32(alpha)**2)  # diagonal update
            pred = PVh_val @ (D[:, None] * UR)                          # (T_val, M)
            corr = (_zscore_cols(pred) * zY_val).mean(axis=0)

            better = corr > best_corr
            best_corr[better]  = corr[better]
            best_alpha[better] = np.float32(alpha)

        return best_alpha

    def fit_predict(self, Y_train_full, Y_test, valphas):
        """Fit final model with per-voxel alphas on full training data.

        SVD of X_train_full already computed — reused across all unique alphas.
        Returns (test_corrs, weights).
        """
        UR = (self._U_full.T @ Y_train_full).astype(np.float32)  # (r, M)

        n_features = self._Vh_full.shape[1]
        W = np.empty((n_features, Y_train_full.shape[1]), dtype=np.float32)

        for alpha in np.unique(valphas):
            mask = valphas == alpha
            D    = self._S_full / (self._S_full**2 + np.float32(alpha)**2)
            W[:, mask] = self._Vh_full.T @ (D[:, None] * UR[:, mask])

        pred  = self._X_test @ W
        corrs = _col_corr(Y_test, pred)
        return corrs, W


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Ridge regression for fMRI voxel prediction.')
    p.add_argument('--subject',    default='subject2')
    p.add_argument('--subject-id', default='s2')
    p.add_argument('--method',     required=True,
                   help='Embedding method: bow | word2vec | glove | bert | bert_finetuned | ...')
    p.add_argument('--data-path',
                   default='/ocean/projects/mth250011p/shared/215a/final_project/data')
    p.add_argument('--embedding-dir', default='../../data/embeddings')
    p.add_argument('--split-path',    default='../../data/train_test_split.json')
    p.add_argument('--result-dir',    default='../../results')
    p.add_argument('--n-alphas',          type=int,   default=10)
    p.add_argument('--alpha-min-log10',   type=float, default=0.0)
    p.add_argument('--alpha-max-log10',   type=float, default=3.0)
    p.add_argument('--chunk-size',        type=int,   default=500,
                   help='Voxels per chunk')
    p.add_argument('--cv-frac',           type=float, default=0.2,
                   help='Fraction of training rows held out for alpha CV (dense only)')
    p.add_argument('--n-alpha-voxels',    type=int,   default=200,
                   help='Random voxels used for alpha selection (sparse/bow only)')
    p.add_argument('--save-weights', action='store_true', default=False,
                   help='Include regression weights in .pkl (large files)')
    # SLURM array support
    p.add_argument('--task-id',    type=int,
                   default=int(os.getenv('SLURM_ARRAY_TASK_ID', os.getenv('TASK_ID', '0'))))
    p.add_argument('--task-count', type=int,
                   default=int(os.getenv('SLURM_ARRAY_TASK_COUNT', os.getenv('TASK_COUNT', '1'))))
    return p.parse_args()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    sid    = args.subject_id
    method = args.method

    script_dir = os.path.dirname(os.path.abspath(__file__))
    def resolve(path):
        return path if os.path.isabs(path) else os.path.normpath(os.path.join(script_dir, path))

    embedding_dir = resolve(args.embedding_dir)
    split_path    = resolve(args.split_path)
    result_dir    = resolve(args.result_dir)

    metric_dir = os.path.join(result_dir, 'metrics')
    model_dir  = os.path.join(result_dir, 'models', f'{sid}_{method}')
    os.makedirs(metric_dir, exist_ok=True)
    os.makedirs(model_dir,  exist_ok=True)

    use_dense = method in DENSE_METHODS
    solver_name = 'SVD (dense)' if use_dense else 'sparse LU (bow)'
    print(f'Subject : {args.subject} ({sid})')
    print(f'Method  : {method}  →  solver: {solver_name}')
    print(f'Task    : {args.task_id + 1}/{args.task_count}')

    # ── load split ───────────────────────────────────────────────────────────
    with open(split_path) as f:
        split = json.load(f)
    train_stories = split['train']
    test_stories  = split['test']

    # ── load and z-score embeddings ──────────────────────────────────────────
    print('Loading embeddings...')
    X_train = load_embedding(find_embedding(embedding_dir, sid, 'train', method))
    X_test  = load_embedding(find_embedding(embedding_dir, sid, 'test',  method))
    X_train, X_test = zscore_X(X_train, X_test)
    print(f'  X_train: {X_train.shape}  X_test: {X_test.shape}')

    # ── open fMRI mmaps (no trimming — Y is pre-trimmed on disk) ────────────
    print('Opening fMRI mmaps...')
    fmri_dir   = os.path.join(args.data_path, args.subject)
    train_maps = {s: np.load(os.path.join(fmri_dir, f'{s}.npy'), mmap_mode='r')
                  for s in train_stories}
    test_maps  = {s: np.load(os.path.join(fmri_dir, f'{s}.npy'), mmap_mode='r')
                  for s in test_stories}

    train_lengths = [train_maps[s].shape[0] for s in train_stories]
    test_lengths  = [test_maps[s].shape[0]  for s in test_stories]
    train_total   = sum(train_lengths)
    test_total    = sum(test_lengths)
    n_voxels      = train_maps[train_stories[0]].shape[1]

    # ── dimension check ──────────────────────────────────────────────────────
    if X_train.shape[0] != train_total:
        raise ValueError(
            f'X_train rows ({X_train.shape[0]}) != stacked fMRI train rows ({train_total}). '
            f'Re-run preprocessing with the current preprocessing_utils.py.'
        )
    if X_test.shape[0] != test_total:
        raise ValueError(
            f'X_test rows ({X_test.shape[0]}) != stacked fMRI test rows ({test_total}).'
        )
    print(f'  fMRI voxels: {n_voxels}  train rows: {train_total}  test rows: {test_total}')

    alphas = np.logspace(args.alpha_min_log10, args.alpha_max_log10,
                         args.n_alphas, dtype=np.float32)

    # ── solver setup (done once, outside chunk loop) ─────────────────────────
    if use_dense:
        print('Precomputing SVD (expensive step, done once)...')
        solver = SVDRidgeSolver(X_train, X_test, cv_frac=args.cv_frac)
        global_alpha = None  # per-voxel alphas selected inside the loop
    else:
        print(f'Selecting global alpha on {args.n_alpha_voxels} random voxels...')
        rng = np.random.default_rng(42)
        alpha_voxels  = np.sort(rng.choice(n_voxels, min(args.n_alpha_voxels, n_voxels), replace=False))
        Y_alpha_train = load_fmri_chunk(train_maps, train_stories, train_lengths, train_total, alpha_voxels)
        global_alpha  = choose_alpha_sparse(X_train, Y_alpha_train, alphas=alphas,
                                            holdout_frac=0.1, sample_voxels=min(32, len(alpha_voxels)))
        print(f'  Selected alpha = {global_alpha:.4f}')
        del Y_alpha_train
        gc.collect()

    # ── distribute chunks across SLURM tasks ─────────────────────────────────
    chunk_starts = list(range(0, n_voxels, args.chunk_size))
    my_starts    = [c for i, c in enumerate(chunk_starts) if i % args.task_count == args.task_id]
    print(f'  Processing {len(my_starts)}/{len(chunk_starts)} chunks '
          f'({args.chunk_size} voxels each)')

    # ── main voxel loop ───────────────────────────────────────────────────────
    all_voxel_idx = []
    all_corrs     = []
    all_weights   = [] if args.save_weights else None

    for chunk_start in my_starts:
        chunk_end = min(chunk_start + args.chunk_size, n_voxels)
        voxel_idx = np.arange(chunk_start, chunk_end)

        Y_train_chunk = load_fmri_chunk(
            train_maps, train_stories, train_lengths, train_total, voxel_idx)
        Y_test_chunk = load_fmri_chunk(
            test_maps, test_stories, test_lengths, test_total, voxel_idx)

        if use_dense:
            # SVD already computed — just diagonal updates per alpha (cheap)
            valphas = solver.select_alpha(Y_train_chunk, alphas)  # per-voxel
            corrs, W = solver.fit_predict(Y_train_chunk, Y_test_chunk, valphas)
        else:
            corrs, W = ridge_corrs_sparse(X_train, Y_train_chunk, X_test, Y_test_chunk,
                                          global_alpha)

        all_voxel_idx.append(voxel_idx)
        all_corrs.append(corrs.astype(np.float32))
        if all_weights is not None:
            all_weights.append(W.astype(np.float32))

        del Y_train_chunk, Y_test_chunk, corrs, W
        gc.collect()

        print(f'  [{chunk_start}:{chunk_end}]  running mean cc = '
              f'{np.mean(np.concatenate(all_corrs)):.4f}')

    # ── aggregate ─────────────────────────────────────────────────────────────
    voxel_indices = np.concatenate(all_voxel_idx)
    corrs         = np.concatenate(all_corrs)

    print(f'\n=== Results: {sid} {method} ===')
    print(f'  Mean CC   : {np.mean(corrs):.4f}')
    print(f'  Median CC : {np.median(corrs):.4f}')
    print(f'  Top 1% CC : {np.percentile(corrs, 99):.4f}')
    print(f'  Top 5% CC : {np.percentile(corrs, 95):.4f}')
    print(f'  Voxels    : {len(voxel_indices)}/{n_voxels}')

    # ── save ──────────────────────────────────────────────────────────────────
    task_tag  = f'_task{args.task_id:04d}' if args.task_count > 1 else ''
    corr_path = os.path.join(metric_dir, f'{sid}_{method}_test_corrs{task_tag}.npy')
    np.save(corr_path, corrs)

    model_obj = {
        'subject'         : args.subject,
        'subject_id'      : sid,
        'method'          : method,
        'solver'          : solver_name,
        'alpha'           : float(global_alpha) if global_alpha is not None else 'per_voxel',
        'alphas_grid'     : alphas,
        'voxel_indices'   : voxel_indices.astype(np.int32),
        'n_voxels_total'  : int(n_voxels),
        'n_features'      : int(X_train.shape[1]),
        'test_corrs'      : corrs,
        'mean_test_cc'    : float(np.mean(corrs)),
        'median_test_cc'  : float(np.median(corrs)),
        'top1pct_test_cc' : float(np.percentile(corrs, 99)),
        'top5pct_test_cc' : float(np.percentile(corrs, 95)),
    }
    if all_weights is not None:
        model_obj['weights'] = np.concatenate(all_weights, axis=1).astype(np.float32)

    model_path = os.path.join(model_dir, f'{sid}_{method}_model{task_tag}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_obj, f)

    print(f'\nSaved corrs -> {corr_path}')
    print(f'Saved model -> {model_path}')


if __name__ == '__main__':
    main()
