#!/usr/bin/env python3
"""
Stability checks for the fMRI ridge regression encoding model.

Three tests that verify the results are real, reproducible, and robust:

  CHECK 1 — Temporal permutation test (s2, GloVe)
    Reuse fitted weights; circularly-shift X_test to break temporal alignment.
    True CC >> null CC proves the model captures real language-brain coupling,
    not spectral artefacts from correlated time-series.

  CHECK 2 — Cross-subject replication (s2 vs s3, GloVe)
    The two subjects have DIFFERENT voxel counts (94 251 vs 95 556), so direct
    voxel-index comparison would be meaningless.  Instead we compare:
      (a) story-level encoding scores  — per test story mean CC, same model
      (b) functional-space similarity  — PCA of top-voxel weight vectors;
                                         colour by CC to reveal spatial structure

  CHECK 3 — Cross-method embedding agreement (s2: GloVe vs BERT variants)
    Run ridge for three methods on the same subject.  High Pearson r between
    CC profiles (or high top-k Jaccard after aligning by CC percentile) shows
    the encoding is driven by the stimulus, not the embedding model.

Usage:
    sbatch stability_check.sh          # full run — compute + save artifacts + plot
    python stability_check.py --plots-only   # re-plot from saved artifacts only

Artifacts saved to results/stability/:
    {sid}_glove_test_corrs.npy          per-voxel CC for s2/s3 GloVe
    s2_{method}_test_corrs.npy          per-voxel CC for BERT methods on s2
    s2_glove_top5pct_weights.npy        top-5% voxel weights for functional PCA
    s3_glove_top5pct_weights.npy
    check1_null_mat.npy                 (N_PERMS, n_voxels) permutation CCs
    check1_shifts.npy                   shift values in TRs
    stability_summary.json              all scalar results
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.normpath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'ridge_regression'))

# ── config ───────────────────────────────────────────────────────────────────
DATA_PATH   = '/scratch/users/s214/lab3'
EMBED_DIR   = os.path.join(REPO_ROOT, 'data/embeddings')
SPLIT_PATH  = os.path.join(REPO_ROOT, 'data/train_test_split.json')
OUTPUT_DIR  = os.path.join(REPO_ROOT, 'results/stability')

ALPHAS      = np.logspace(0, 3, 10, dtype=np.float32)
CHUNK_SIZE  = 500
TOP_PCT     = 5
N_PERMS     = 7

SUBJECTS = [('s2', 'subject2'), ('s3', 'subject3')]
METHODS_CHECK3 = ['glove', 'pretrained_bert', 'finetuned_lora']


# ── helpers ───────────────────────────────────────────────────────────────────

def jaccard(mask_a, mask_b):
    i = int(np.sum(mask_a & mask_b))
    u = int(np.sum(mask_a | mask_b))
    return i / u if u > 0 else 0.0


def top_mask(corrs, pct):
    return corrs >= np.percentile(corrs, 100 - pct)


def _embed_path(sid, split, method):
    for ext in ('npy', 'npz'):
        p = os.path.join(EMBED_DIR, f'{sid}_{split}_{method}_embeddings.{ext}')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'Embedding not found: {sid}/{split}/{method} in {EMBED_DIR}')


def _artifact(name):
    return os.path.join(OUTPUT_DIR, name)


def _savefig(fig, name):
    """Save as both PDF (for LaTeX) and PNG (for quick view)."""
    stem = name.rsplit('.', 1)[0]
    for ext in ('pdf', 'png'):
        p = _artifact(f'{stem}.{ext}')
        fig.savefig(p, dpi=150, bbox_inches='tight')
        if ext == 'pdf':
            print(f'  Saved → {p}')
    plt.close(fig)


def _functional_pca(weights_top):
    """Return 2D PCA projection and variance explained for top-voxel weights."""
    from numpy.linalg import svd
    W   = weights_top.T
    W_c = W - W.mean(axis=0, keepdims=True)
    U, S, _ = svd(W_c, full_matrices=False)
    pcs     = U[:, :2] * S[:2]
    var_exp = S[:2]**2 / (S**2).sum()
    return pcs, var_exp


# ── computation ───────────────────────────────────────────────────────────────

def run_ridge(sid, subject, method, train_stories, test_stories, save_weights=True):
    from run_ridge import SVDRidgeSolver, load_embedding, zscore_X, load_fmri_chunk, _col_corr

    print(f"\n{'='*70}")
    print(f"  Ridge  |  subject={sid}  method={method}")
    print(f"{'='*70}")

    X_train = load_embedding(_embed_path(sid, 'train', method))
    X_test  = load_embedding(_embed_path(sid, 'test',  method))
    X_train, X_test = zscore_X(X_train, X_test)
    X_test = np.asarray(X_test, dtype=np.float32)

    fmri_dir      = os.path.join(DATA_PATH, subject)
    train_maps    = {s: np.load(os.path.join(fmri_dir, f'{s}.npy'), mmap_mode='r')
                     for s in train_stories}
    test_maps     = {s: np.load(os.path.join(fmri_dir, f'{s}.npy'), mmap_mode='r')
                     for s in test_stories}
    train_lengths = [train_maps[s].shape[0] for s in train_stories]
    test_lengths  = [test_maps[s].shape[0]  for s in test_stories]
    train_total   = sum(train_lengths)
    test_total    = sum(test_lengths)
    n_voxels      = train_maps[train_stories[0]].shape[1]

    if X_train.shape[0] != train_total:
        raise ValueError(f'X_train rows {X_train.shape[0]} != fMRI train rows {train_total}')
    if X_test.shape[0] != test_total:
        raise ValueError(f'X_test rows {X_test.shape[0]} != fMRI test rows {test_total}')

    print(f'  X_train {X_train.shape}  X_test {X_test.shape}')
    print(f'  fMRI: {n_voxels} voxels | {train_total} train TRs | {test_total} test TRs')

    solver = SVDRidgeSolver(X_train, X_test)

    n_features  = X_test.shape[1]
    all_corrs   = np.empty(n_voxels, dtype=np.float32)
    all_weights = np.empty((n_features, n_voxels), dtype=np.float32) if save_weights else None

    story_slices = {}
    offset = 0
    for s, n in zip(test_stories, test_lengths):
        story_slices[s] = (offset, offset + n)
        offset += n

    n_chunks = (n_voxels + CHUNK_SIZE - 1) // CHUNK_SIZE
    t0 = time.time()
    for ci, start in enumerate(range(0, n_voxels, CHUNK_SIZE)):
        end       = min(start + CHUNK_SIZE, n_voxels)
        chunk_idx = np.arange(start, end)

        Y_train = load_fmri_chunk(train_maps, train_stories, train_lengths,
                                   train_total, chunk_idx)
        Y_test  = load_fmri_chunk(test_maps, test_stories, test_lengths,
                                   test_total, chunk_idx)

        best_a        = solver.select_alpha(Y_train, ALPHAS)
        corrs, W      = solver.fit_predict(Y_train, Y_test, best_a)
        all_corrs[start:end] = corrs
        if save_weights:
            all_weights[:, start:end] = W

        if ci % 40 == 0 or ci == n_chunks - 1:
            elapsed = time.time() - t0
            eta     = elapsed / (ci + 1) * (n_chunks - ci - 1)
            print(f'  chunk {ci+1:4d}/{n_chunks}  voxel {start:6d}/{n_voxels}'
                  f'  {elapsed/60:.1f}min  ETA {eta/60:.1f}min', end='\r')

    print(f'\n  Done.  mean CC={all_corrs.mean():.4f}'
          f'  median={np.median(all_corrs):.4f}'
          f'  >0: {(all_corrs > 0).mean()*100:.1f}%')

    story_corrs = {}
    if save_weights and all_weights is not None:
        from run_ridge import _col_corr
        print('  Computing per-story CC scores...')
        for story, (r0, r1) in story_slices.items():
            X_st = X_test[r0:r1]
            story_voxel_ccs = []
            for start in range(0, n_voxels, CHUNK_SIZE * 4):
                end       = min(start + CHUNK_SIZE * 4, n_voxels)
                chunk_idx = np.arange(start, end)
                from run_ridge import load_fmri_chunk as _lfc
                Y_st = _lfc(test_maps, [story], [r1 - r0], r1 - r0, chunk_idx)
                pred = X_st @ all_weights[:, chunk_idx]
                story_voxel_ccs.append(_col_corr(Y_st, pred))
            story_corrs[story] = float(np.concatenate(story_voxel_ccs).mean())
        print('  Per-story CC:', {k: f'{v:.3f}' for k, v in story_corrs.items()})

    return all_corrs, all_weights, X_test, test_maps, test_stories, test_lengths, test_total, story_corrs


def permutation_null(X_test, weights, test_maps, test_stories, test_lengths, test_total,
                     n_perms):
    from run_ridge import load_fmri_chunk, _col_corr
    n_test   = X_test.shape[0]
    n_voxels = weights.shape[1]
    min_s    = n_test // 8
    max_s    = n_test - n_test // 8
    shifts   = np.linspace(min_s, max_s, n_perms, dtype=int)

    null_mat = np.empty((n_perms, n_voxels), dtype=np.float32)
    for pi, shift in enumerate(shifts):
        X_perm  = np.roll(X_test, int(shift), axis=0)
        perm_cc = np.empty(n_voxels, dtype=np.float32)
        for start in range(0, n_voxels, CHUNK_SIZE):
            end       = min(start + CHUNK_SIZE, n_voxels)
            chunk_idx = np.arange(start, end)
            Y_chunk   = load_fmri_chunk(test_maps, test_stories, test_lengths,
                                         test_total, chunk_idx)
            pred      = X_perm @ weights[:, chunk_idx]
            perm_cc[start:end] = _col_corr(Y_chunk, pred)
        null_mat[pi] = perm_cc
        print(f'  perm {pi+1}/{n_perms}  shift={shift} TRs ({shift*2}s)'
              f'  median={np.median(perm_cc):.4f}')
    return null_mat, shifts


# ── plot functions (called from both full run and --plots-only) ───────────────

def plot_check1(true_cc, null_mat, shifts):
    """Three panels for Check 1 — temporal permutation test."""
    true_med    = float(np.median(true_cc))
    null_meds   = np.median(null_mat, axis=1)
    null_mean   = float(null_meds.mean())
    null_std    = float(null_meds.std())
    null_max_vx = null_mat.max(axis=0)
    frac_beats  = float((true_cc > null_max_vx).mean())

    # Panel A — histogram
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(true_cc, bins=150, density=True, alpha=0.75, color='steelblue',
            label=f'True  (med={true_med:.3f})', zorder=3)
    for ni, nc in enumerate(null_mat):
        ax.hist(nc, bins=150, density=True, alpha=0.2, color='tomato',
                label=f'Null (mean med$\\approx${null_mean:.3f})' if ni == 0 else '')
    ax.set_xlabel('Pearson $r$'); ax.set_ylabel('Density')
    ax.set_title('True vs.\ Permuted CC (s2, GloVe)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    _savefig(fig, 'check1_A_histogram.pdf')

    # Panel B — median CC vs shift
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(shifts * 2, null_meds, 'o-', color='tomato', lw=1.5, ms=6, label='Null')
    ax.axhline(true_med, color='steelblue', lw=2, ls='--',
               label=f'True ($r={true_med:.3f}$)')
    ax.fill_between(shifts * 2, null_mean - null_std, null_mean + null_std,
                    alpha=0.2, color='tomato')
    ax.set_xlabel('Circular shift (seconds)'); ax.set_ylabel('Median CC')
    ax.set_title('Median CC vs.\ Temporal Misalignment')
    ax.legend(fontsize=8)
    plt.tight_layout()
    _savefig(fig, 'check1_B_misalignment.pdf')

    # Panel C — true vs best-null scatter
    fig, ax = plt.subplots(figsize=(5, 4))
    rng  = np.random.default_rng(0)
    samp = rng.choice(len(true_cc), min(10000, len(true_cc)), replace=False)
    ax.scatter(null_max_vx[samp], true_cc[samp], s=2, alpha=0.3, color='gray')
    lim = [min(null_max_vx.min(), true_cc.min()) - 0.01,
           max(null_max_vx.max(), true_cc.max()) + 0.01]
    ax.plot(lim, lim, 'k--', lw=1, label='$y=x$')
    ax.set_xlabel('Best-null CC (max over shifts)'); ax.set_ylabel('True CC')
    ax.set_title(f'True vs.\ Best-null per Voxel\n({frac_beats*100:.1f}\\% above diagonal)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    _savefig(fig, 'check1_C_voxel_scatter.pdf')


def plot_check2(corrs_s2, corrs_s3, story_scores_s2, story_scores_s3,
                top_weights_s2, top_weights_s3):
    """Four panels for Check 2 — cross-subject replication."""
    shared_stories = sorted(set(story_scores_s2) & set(story_scores_s3))
    v2 = np.array([story_scores_s2[s] for s in shared_stories])
    v3 = np.array([story_scores_s3[s] for s in shared_stories])
    r_story = float(np.corrcoef(v2, v3)[0, 1])

    # Panel A — story-level scatter
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(v2, v3, s=40, alpha=0.85, color='darkorchid', zorder=3)
    for i, s in enumerate(shared_stories):
        ax.annotate(s[:12], (v2[i], v3[i]), fontsize=5, alpha=0.7)
    lim = [min(v2.min(), v3.min()) - 0.005, max(v2.max(), v3.max()) + 0.005]
    ax.plot(lim, lim, 'k--', lw=1)
    ax.set_xlabel('s2 mean CC (GloVe)'); ax.set_ylabel('s3 mean CC (GloVe)')
    ax.set_title(f'Story-level Encoding Agreement ($r={r_story:.3f}$)')
    plt.tight_layout()
    _savefig(fig, 'check2_A_story_scatter.pdf')

    # Panels B-s2, B-s3 — functional PCA per subject
    for sid, corrs, w_top in [('s2', corrs_s2, top_weights_s2),
                               ('s3', corrs_s3, top_weights_s3)]:
        m      = top_mask(corrs, TOP_PCT)
        cc_top = corrs[m]
        pcs, var_exp = _functional_pca(w_top)

        fig, ax = plt.subplots(figsize=(5, 4.5))
        sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=cc_top,
                        cmap='viridis', s=8, alpha=0.7, vmin=0)
        ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}\\% var)')
        ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}\\% var)')
        ax.set_title(f'{sid}: Top-{TOP_PCT}\\% Voxels — Functional PCA')
        plt.colorbar(sc, ax=ax, label='Test CC')
        plt.tight_layout()
        _savefig(fig, f'check2_B_{sid}_pca.pdf')

    # Panel C — CC distribution comparison
    fig, ax = plt.subplots(figsize=(5, 4))
    bins = np.linspace(min(corrs_s2.min(), corrs_s3.min()),
                       max(corrs_s2.max(), corrs_s3.max()), 120)
    ax.hist(corrs_s2, bins=bins, density=True, alpha=0.6, color='steelblue',
            label=f's2 (n={len(corrs_s2)}, med={np.median(corrs_s2):.3f})')
    ax.hist(corrs_s3, bins=bins, density=True, alpha=0.6, color='tomato',
            label=f's3 (n={len(corrs_s3)}, med={np.median(corrs_s3):.3f})')
    ax.set_xlabel('Test CC (GloVe)'); ax.set_ylabel('Density')
    ax.set_title('CC Distributions (s2 vs.\ s3, GloVe)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    _savefig(fig, 'check2_C_cc_distributions.pdf')


def plot_check3(method_corrs, methods):
    """Four panels for Check 3 — cross-method embedding agreement."""
    n_methods = len(methods)
    min_vox   = min(len(method_corrs[m]) for m in methods)
    mc        = {m: method_corrs[m][:min_vox] for m in methods}

    # Build r_matrix and jacc_table
    r_matrix   = np.eye(n_methods)
    jacc_table = np.zeros((n_methods, n_methods))
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if j <= i:
                continue
            r = float(np.corrcoef(mc[m1], mc[m2])[0, 1])
            r_matrix[i, j] = r_matrix[j, i] = r
            j5 = jaccard(top_mask(mc[m1], 5), top_mask(mc[m2], 5))
            jacc_table[i, j] = jacc_table[j, i] = j5

    # Panel A — pairwise scatter (GloVe vs finetuned_lora)
    fig, ax = plt.subplots(figsize=(5, 4))
    m1, m2 = methods[0], methods[-1]
    rng  = np.random.default_rng(3)
    samp = rng.choice(min_vox, min(8000, min_vox), replace=False)
    r_12 = float(np.corrcoef(mc[m1], mc[m2])[0, 1])
    ax.scatter(mc[m1][samp], mc[m2][samp], s=2, alpha=0.3, color='gray')
    shared_idx = np.where(top_mask(mc[m1], 5) & top_mask(mc[m2], 5))[0]
    if len(shared_idx) > 0:
        sh_samp = rng.choice(shared_idx, min(500, len(shared_idx)), replace=False)
        ax.scatter(mc[m1][sh_samp], mc[m2][sh_samp], s=8, alpha=0.7, color='darkorange',
                   label=f'Shared top-{TOP_PCT}\\% (n={len(shared_idx)})')
    ax.set_xlabel(f'{m1} CC'); ax.set_ylabel(f'{m2} CC')
    ax.set_title(f'{m1} vs.\\ {m2}  ($r={r_12:.3f}$)')
    ax.legend(fontsize=8)
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    plt.tight_layout()
    _savefig(fig, 'check3_A_voxel_scatter.pdf')

    # Panel B — Pearson r heatmap
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(r_matrix, vmin=0, vmax=1, cmap='Blues')
    ax.set_xticks(range(n_methods)); ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(n_methods)); ax.set_yticklabels(methods, fontsize=8)
    for i in range(n_methods):
        for j in range(n_methods):
            ax.text(j, i, f'{r_matrix[i,j]:.2f}', ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('CC Profile Pearson $r$ (per-voxel)')
    plt.tight_layout()
    _savefig(fig, 'check3_B_pearson_heatmap.pdf')

    # Panel C — Jaccard heatmap
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(jacc_table, vmin=0, cmap='Oranges')
    ax.set_xticks(range(n_methods)); ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(n_methods)); ax.set_yticklabels(methods, fontsize=8)
    for i in range(n_methods):
        for j in range(n_methods):
            ax.text(j, i, f'{jacc_table[i,j]:.3f}', ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f'Top-{TOP_PCT}\\% Jaccard Overlap (chance$\\approx$0.0025)')
    plt.tight_layout()
    _savefig(fig, 'check3_C_jaccard_heatmap.pdf')

    # Panel D — CC distributions all methods
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ['steelblue', 'darkorange', 'seagreen']
    bins = np.linspace(min(mc[m].min() for m in methods),
                       max(mc[m].max() for m in methods), 120)
    for ci, m in enumerate(methods):
        cc = mc[m]
        ax.hist(cc, bins=bins, density=True, alpha=0.55, color=colors[ci % len(colors)],
                label=f'{m} (med={np.median(cc):.3f})')
    ax.set_xlabel('Test CC'); ax.set_ylabel('Density')
    ax.set_title('CC Distributions by Embedding Method (s2)')
    ax.legend(fontsize=7)
    plt.tight_layout()
    _savefig(fig, 'check3_D_cc_distributions.pdf')


# ── plots-only mode ───────────────────────────────────────────────────────────

def plots_only():
    """Reload saved artifacts and regenerate all figures — no recomputation."""
    print('Loading saved artifacts...')

    # Check 1
    true_cc  = np.load(_artifact('s2_glove_test_corrs.npy'))
    null_mat = np.load(_artifact('check1_null_mat.npy'))
    shifts   = np.load(_artifact('check1_shifts.npy'))
    print(f'  check1: true_cc {true_cc.shape}  null_mat {null_mat.shape}  shifts {shifts.shape}')

    # Check 2
    corrs_s2     = np.load(_artifact('s2_glove_test_corrs.npy'))
    corrs_s3     = np.load(_artifact('s3_glove_test_corrs.npy'))
    top_w_s2     = np.load(_artifact('s2_glove_top5pct_weights.npy'))
    top_w_s3     = np.load(_artifact('s3_glove_top5pct_weights.npy'))
    with open(_artifact('stability_summary.json')) as f:
        summary = json.load(f)
    story_scores_s2 = summary['check2']['story_scores_s2']
    story_scores_s3 = summary['check2']['story_scores_s3']
    print(f'  check2: s2 {corrs_s2.shape}  s3 {corrs_s3.shape}'
          f'  top_w_s2 {top_w_s2.shape}  top_w_s3 {top_w_s3.shape}')

    # Check 3
    method_corrs = {
        'glove':          np.load(_artifact('s2_glove_test_corrs.npy')),
        'pretrained_bert': np.load(_artifact('s2_pretrained_bert_test_corrs.npy')),
        'finetuned_lora':  np.load(_artifact('s2_finetuned_lora_test_corrs.npy')),
    }
    print(f'  check3: {[(m, v.shape) for m, v in method_corrs.items()]}')

    print('\nRegenerating figures...')
    plot_check1(true_cc, null_mat, shifts)
    plot_check2(corrs_s2, corrs_s3, story_scores_s2, story_scores_s3, top_w_s2, top_w_s3)
    plot_check3(method_corrs, METHODS_CHECK3)
    print(f'\nAll figures saved to → {OUTPUT_DIR}')


# ── full computation run ──────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(SPLIT_PATH) as f:
        split = json.load(f)
    train_stories = split['train']
    test_stories  = split['test']

    summary = {}

    # ── GloVe ridge for both subjects ────────────────────────────────────────
    glove = {}
    for sid, subject in SUBJECTS:
        corrs, weights, X_test, tmaps, ts, tl, tt, sc = run_ridge(
            sid, subject, 'glove', train_stories, test_stories, save_weights=True
        )
        glove[sid] = dict(corrs=corrs, weights=weights, X_test=X_test,
                          tmaps=tmaps, ts=ts, tl=tl, tt=tt, story_corrs=sc)

        # Save per-voxel CC
        np.save(_artifact(f'{sid}_glove_test_corrs.npy'), corrs)

        # Save top-5% voxel weights (needed for functional PCA in --plots-only)
        m     = top_mask(corrs, TOP_PCT)
        w_top = weights[:, m]
        np.save(_artifact(f'{sid}_glove_top5pct_weights.npy'), w_top)
        print(f'  Saved top-{TOP_PCT}% weights for {sid}: {w_top.shape}')

    # ════════════════════════════════════════════════════════════════════════
    # CHECK 1 — Temporal permutation test  (s2, GloVe)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*70}")
    print("  CHECK 1: Temporal permutation test  (s2 / GloVe)")
    print(f"{'#'*70}")

    r2 = glove['s2']
    null_mat, shifts = permutation_null(
        r2['X_test'], r2['weights'],
        r2['tmaps'], r2['ts'], r2['tl'], r2['tt'], N_PERMS,
    )

    # Save permutation artifacts
    np.save(_artifact('check1_null_mat.npy'), null_mat)
    np.save(_artifact('check1_shifts.npy'), shifts)

    true_cc   = r2['corrs']
    true_med  = float(np.median(true_cc))
    null_meds = np.median(null_mat, axis=1)
    null_mean = float(null_meds.mean())
    null_std  = float(null_meds.std())
    fold      = true_med / max(abs(null_mean), 1e-9)
    frac_beats = float((true_cc > null_mat.max(axis=0)).mean())

    print(f"\n  True median CC : {true_med:.4f}")
    print(f"  Null mean±std  : {null_mean:.4f} ± {null_std:.4f}")
    print(f"  Fold above null: {fold:.1f}×")
    print(f"  Voxels beating ALL nulls: {frac_beats*100:.1f}%")

    summary['check1'] = dict(
        true_median_cc=true_med, null_mean_cc=null_mean, null_std_cc=null_std,
        fold_above_null=fold, frac_beats_all_nulls=frac_beats,
    )

    # ════════════════════════════════════════════════════════════════════════
    # CHECK 2 — Cross-subject replication
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*70}")
    print("  CHECK 2: Cross-subject replication  (GloVe, s2 vs s3)")
    print(f"{'#'*70}")

    sc2 = glove['s2']['story_corrs']
    sc3 = glove['s3']['story_corrs']
    shared_stories = sorted(set(sc2) & set(sc3))
    v2 = np.array([sc2[s] for s in shared_stories])
    v3 = np.array([sc3[s] for s in shared_stories])
    r_story = float(np.corrcoef(v2, v3)[0, 1])
    print(f"  Story-level Pearson r (s2 vs s3): {r_story:.4f}")
    for s, a, b in zip(shared_stories, v2, v3):
        print(f"    {s:40s}  s2={a:.3f}  s3={b:.3f}")

    pcts = [1, 2, 5, 10, 20]
    print(f"\n  CC percentile comparison ...")
    for pct in pcts:
        t2 = np.percentile(glove['s2']['corrs'], 100 - pct)
        t3 = np.percentile(glove['s3']['corrs'], 100 - pct)
        print(f'  Top {pct:2d}% threshold:  s2={t2:.4f}  s3={t3:.4f}')

    summary['check2'] = dict(
        story_pearson_r=r_story,
        story_scores_s2={s: float(v) for s, v in zip(shared_stories, v2)},
        story_scores_s3={s: float(v) for s, v in zip(shared_stories, v3)},
        s2_mean_cc=float(glove['s2']['corrs'].mean()),
        s3_mean_cc=float(glove['s3']['corrs'].mean()),
        s2_n_voxels=int(len(glove['s2']['corrs'])),
        s3_n_voxels=int(len(glove['s3']['corrs'])),
    )

    # ════════════════════════════════════════════════════════════════════════
    # CHECK 3 — Cross-method embedding agreement  (s2)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*70}")
    print("  CHECK 3: Cross-method embedding agreement  (s2)")
    print(f"{'#'*70}")

    method_corrs = {'glove': glove['s2']['corrs']}
    for method in METHODS_CHECK3:
        if method == 'glove':
            continue
        corrs_m, _, _, _, _, _, _, _ = run_ridge(
            's2', 'subject2', method, train_stories, test_stories, save_weights=False
        )
        np.save(_artifact(f's2_{method}_test_corrs.npy'), corrs_m)
        method_corrs[method] = corrs_m

    methods   = list(method_corrs.keys())
    min_vox   = min(len(v) for v in method_corrs.values())
    mc        = {m: method_corrs[m][:min_vox] for m in methods}
    n_methods = len(methods)

    print(f"\n  Per-method summary ({min_vox} voxels):")
    for m in methods:
        cc = mc[m]
        print(f'    {m:20s}  mean={cc.mean():.4f}  median={np.median(cc):.4f}'
              f'  >0: {(cc>0).mean()*100:.1f}%')

    r_matrix   = np.eye(n_methods)
    jacc_table = np.zeros((n_methods, n_methods))
    print(f"\n  CC profile Pearson r matrix:")
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if j <= i: continue
            r = float(np.corrcoef(mc[m1], mc[m2])[0, 1])
            r_matrix[i, j] = r_matrix[j, i] = r
            print(f'    {m1} vs {m2}: r={r:.4f}')
    print(f"\n  Jaccard overlap at top-5%:")
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if j <= i: continue
            j5 = jaccard(top_mask(mc[m1], 5), top_mask(mc[m2], 5))
            jacc_table[i, j] = jacc_table[j, i] = j5
            print(f'    {m1} vs {m2}: Jaccard={j5:.4f}  ({j5/0.0025:.0f}× chance)')

    summary['check3'] = dict(
        n_voxels=min_vox,
        methods=methods,
        mean_cc={m: float(mc[m].mean()) for m in methods},
        median_cc={m: float(np.median(mc[m])) for m in methods},
        profile_pearson_r={
            f'{m1}_vs_{m2}': float(r_matrix[i, j])
            for i, m1 in enumerate(methods)
            for j, m2 in enumerate(methods) if j > i
        },
        top5pct_jaccard={
            f'{m1}_vs_{m2}': float(jacc_table[i, j])
            for i, m1 in enumerate(methods)
            for j, m2 in enumerate(methods) if j > i
        },
    )

    # ── Save summary JSON ─────────────────────────────────────────────────────
    summary['meta'] = dict(
        date=__import__('datetime').datetime.now().isoformat(),
        n_permutations=N_PERMS,
        top_percentile=TOP_PCT,
        alphas=ALPHAS.tolist(),
        subjects={sid: subject for sid, subject in SUBJECTS},
        methods_check3=METHODS_CHECK3,
    )
    out = _artifact('stability_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n  Saved summary → {out}')

    # ── Generate all figures ──────────────────────────────────────────────────
    print('\nGenerating figures...')

    top_w_s2 = np.load(_artifact('s2_glove_top5pct_weights.npy'))
    top_w_s3 = np.load(_artifact('s3_glove_top5pct_weights.npy'))

    plot_check1(true_cc, null_mat, shifts)
    plot_check2(
        glove['s2']['corrs'], glove['s3']['corrs'],
        sc2, sc3,
        top_w_s2, top_w_s3,
    )
    plot_check3(method_corrs, methods)

    print(f"\n{'='*70}")
    print("  STABILITY SUMMARY")
    print(f"{'='*70}")
    c1 = summary['check1']
    print(f"  Check 1: True median CC={c1['true_median_cc']:.4f}"
          f"  null={c1['null_mean_cc']:.4f}"
          f"  → {c1['fold_above_null']:.1f}× above null"
          f"  ({c1['frac_beats_all_nulls']*100:.1f}% voxels beat all nulls)")
    c2 = summary['check2']
    print(f"  Check 2: story-level r={c2['story_pearson_r']:.4f}"
          f"  s2 mean CC={c2['s2_mean_cc']:.4f}  s3 mean CC={c2['s3_mean_cc']:.4f}")
    c3 = summary['check3']
    print(f"  Check 3: top-5% Jaccard among methods:")
    for pair, j in c3['top5pct_jaccard'].items():
        print(f"    {pair}: {j:.4f}  ({j/0.0025:.0f}× chance)")
    print(f"\n  All outputs saved to → {OUTPUT_DIR}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plots-only', action='store_true',
                        help='Skip computation; reload saved artifacts and regenerate figures only')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.plots_only:
        plots_only()
    else:
        main()
