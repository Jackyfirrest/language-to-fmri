"""
CC distribution analysis for Lab 3.1 Part 2.

Loads merged ridge regression correlation results and produces:
  1. Histogram of CC across all voxels for the best embedding (word2vec, subject3)
  2. Side-by-side comparison of all methods and both subjects
  3. Summary table printed to stdout

Run from repo root:
    python code/analysis/plot_cc_distribution.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../../results/metrics')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '../../results/figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

METHODS  = ['bow', 'word2vec', 'glove']
SUBJECTS = ['s2', 's3']
COLORS   = {'bow': '#4C72B0', 'word2vec': '#DD8452', 'glove': '#55A868'}
LABELS   = {'bow': 'BoW', 'word2vec': 'Word2Vec', 'glove': 'GloVe'}


def load_corrs(sid, method):
    path = os.path.join(RESULTS_DIR, f'{sid}_{method}_test_corrs.npy')
    return np.load(path)


def print_summary(corrs, sid, method):
    print(f'  {sid} {method:10s}: mean={np.mean(corrs):.4f}  '
          f'median={np.median(corrs):.4f}  '
          f'top1%={np.percentile(corrs, 99):.4f}  '
          f'top5%={np.percentile(corrs, 95):.4f}')


# ── load all results ──────────────────────────────────────────────────────────
print('Loading results...')
all_corrs = {}
for sid in SUBJECTS:
    for method in METHODS:
        key = (sid, method)
        all_corrs[key] = load_corrs(sid, method)
        print_summary(all_corrs[key], sid, method)


# ── Figure 1: best embedding detailed distribution ────────────────────────────
# Best = word2vec, subject3 (highest top-1% CC)
best_corrs = all_corrs[('s3', 'word2vec')]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# left: full distribution
ax = axes[0]
ax.hist(best_corrs, bins=100, color=COLORS['word2vec'], alpha=0.8, edgecolor='none')
ax.axvline(np.mean(best_corrs),   color='black',  lw=1.5, ls='--', label=f'Mean = {np.mean(best_corrs):.3f}')
ax.axvline(np.median(best_corrs), color='gray',   lw=1.5, ls=':',  label=f'Median = {np.median(best_corrs):.3f}')
ax.axvline(np.percentile(best_corrs, 95), color='red', lw=1.5, ls='-.',
           label=f'Top 5% threshold = {np.percentile(best_corrs, 95):.3f}')
ax.set_xlabel('Pearson Correlation Coefficient (CC)')
ax.set_ylabel('Number of Voxels')
ax.set_title('CC Distribution — Word2Vec, Subject 3\n(all 95,556 voxels)')
ax.legend(fontsize=9)

# right: zoomed into positive tail (top 10%)
ax = axes[1]
threshold = np.percentile(best_corrs, 90)
top_corrs  = best_corrs[best_corrs > threshold]
ax.hist(top_corrs, bins=60, color=COLORS['word2vec'], alpha=0.8, edgecolor='none')
ax.axvline(np.percentile(best_corrs, 95), color='red', lw=1.5, ls='-.',
           label=f'Top 5% = {np.percentile(best_corrs, 95):.3f}')
ax.axvline(np.percentile(best_corrs, 99), color='darkred', lw=1.5, ls='--',
           label=f'Top 1% = {np.percentile(best_corrs, 99):.3f}')
ax.set_xlabel('Pearson Correlation Coefficient (CC)')
ax.set_ylabel('Number of Voxels')
ax.set_title('Top 10% Voxels (language-responsive)')
ax.legend(fontsize=9)

fig.suptitle('Word2Vec Ridge Regression — Voxel CC Distribution (Subject 3)', fontsize=12)
fig.tight_layout()
out = os.path.join(FIGURES_DIR, 'cc_distribution_word2vec_s3.pdf')
fig.savefig(out, bbox_inches='tight')
fig.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
print(f'\nSaved -> {out}')


# ── Figure 2: all methods × subjects comparison ───────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=False)

for row, sid in enumerate(SUBJECTS):
    for col, method in enumerate(METHODS):
        ax    = axes[row, col]
        corrs = all_corrs[(sid, method)]
        ax.hist(corrs, bins=80, color=COLORS[method], alpha=0.8, edgecolor='none')
        ax.axvline(np.percentile(corrs, 99), color='darkred', lw=1.2, ls='--',
                   label=f'Top 1% = {np.percentile(corrs, 99):.3f}')
        ax.set_title(f'{LABELS[method]} — Subject {sid.upper()}\n'
                     f'mean={np.mean(corrs):.3f}, top1%={np.percentile(corrs, 99):.3f}')
        ax.set_xlabel('CC')
        ax.set_ylabel('Voxels' if col == 0 else '')
        ax.legend(fontsize=8)

fig.suptitle('CC Distribution Across All Embeddings and Subjects', fontsize=13)
fig.tight_layout()
out2 = os.path.join(FIGURES_DIR, 'cc_distribution_all_methods.pdf')
fig.savefig(out2, bbox_inches='tight')
fig.savefig(out2.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
print(f'Saved -> {out2}')


# ── Figure 3: mean CC bar chart comparison ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

metrics = ['Mean CC', 'Top 1% CC']
getters = [np.mean, lambda c: np.percentile(c, 99)]

for ax, metric, getter in zip(axes, metrics, getters):
    x      = np.arange(len(METHODS))
    width  = 0.35
    vals_s2 = [getter(all_corrs[('s2', m)]) for m in METHODS]
    vals_s3 = [getter(all_corrs[('s3', m)]) for m in METHODS]
    ax.bar(x - width/2, vals_s2, width, label='Subject 2',
           color=[COLORS[m] for m in METHODS], alpha=0.6)
    ax.bar(x + width/2, vals_s3, width, label='Subject 3',
           color=[COLORS[m] for m in METHODS], alpha=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in METHODS])
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.legend(['Subject 2 (lighter)', 'Subject 3 (darker)'], fontsize=8)

fig.suptitle('Embedding Comparison — Ridge Regression Performance', fontsize=12)
fig.tight_layout()
out3 = os.path.join(FIGURES_DIR, 'cc_comparison_bar.pdf')
fig.savefig(out3, bbox_inches='tight')
fig.savefig(out3.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
print(f'Saved -> {out3}')

print('\nDone. Figures saved to results/figures/')
