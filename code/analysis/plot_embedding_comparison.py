"""
Cross-embedding comparison analysis for Lab 3.2.

Answers: do different embedding methods perform equally well across voxels?

Produces:
  1. Bar chart — mean & top-1% CC for all 5 methods × 2 subjects
  2. Overlaid CC histograms — all methods on one plot per subject
  3. Scatter plots — CC per voxel: Word2Vec vs BERT variants
  4. Gain map — CC_finetuned_lora - CC_word2vec per voxel (histogram)
  5. Voxel overlap — what fraction of top-5% voxels are shared across methods?

Run from repo root:
    python code/analysis/plot_embedding_comparison.py
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import pearsonr

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../../results/metrics')
MODELS_DIR  = os.path.join(os.path.dirname(__file__), '../../results/models')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '../../results/figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

METHODS = ['bow', 'word2vec', 'glove', 'pretrained_bert', 'finetuned_lora']
LABELS  = {
    'bow':            'BoW',
    'word2vec':       'Word2Vec',
    'glove':          'GloVe',
    'pretrained_bert':'BERT (pretrained)',
    'finetuned_lora': 'BERT (LoRA)',
}
COLORS  = {
    'bow':            '#4C72B0',
    'word2vec':       '#DD8452',
    'glove':          '#55A868',
    'pretrained_bert':'#C44E52',
    'finetuned_lora': '#8172B2',
}
SUBJECTS = ['s2', 's3']
SUB_LABELS = {'s2': 'Subject 2', 's3': 'Subject 3'}


def load(sid, method):
    path = os.path.join(RESULTS_DIR, f'{sid}_{method}_test_corrs.npy')
    return np.load(path)


def load_sorted(sid, method):
    """Load CC array sorted to sequential voxel order using voxel_indices from pkl."""
    cc = load(sid, method)
    pkl_path = os.path.join(MODELS_DIR, f'{sid}_{method}', f'{sid}_{method}_model.pkl')
    if not os.path.exists(pkl_path):
        return cc  # no pkl available, return as-is
    with open(pkl_path, 'rb') as f:
        model = pickle.load(f)
    voxel_indices = model.get('voxel_indices', None)
    if voxel_indices is None or len(voxel_indices) != len(cc):
        return cc
    sort_idx = np.argsort(voxel_indices)
    return cc[sort_idx]


# ── load ──────────────────────────────────────────────────────────────────────
print('Loading results...')
corrs = {}
corrs_sorted = {}
for sid in SUBJECTS:
    for m in METHODS:
        corrs[(sid, m)] = load(sid, m)
        corrs_sorted[(sid, m)] = load_sorted(sid, m)
        c = corrs[(sid, m)]
        print(f'  {sid} {m:20s}: mean={np.mean(c):.4f}  '
              f'top1%={np.percentile(c, 99):.4f}')


# ── Figure 1: bar chart (mean CC and top-1% CC) ───────────────────────────────
# Use two colors (one per subject) + hatching per method so the legend is accurate
HATCHES  = ['', '///', 'xxx', '...', '+++']
SUB_COLORS = {'s2': '#5B9BD5', 's3': '#E07B54'}   # blue=s2, orange=s3

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x     = np.arange(len(METHODS))
width = 0.35

for ax, (metric_label, getter) in zip(
        axes,
        [('Mean CC', np.mean),
         ('Top 1% CC', lambda c: np.percentile(c, 99))]):
    vals_s2 = [getter(corrs[('s2', m)]) for m in METHODS]
    vals_s3 = [getter(corrs[('s3', m)]) for m in METHODS]

    for i, (m, h) in enumerate(zip(METHODS, HATCHES)):
        ax.bar(x[i] - width/2, vals_s2[i], width,
               color=SUB_COLORS['s2'], hatch=h, edgecolor='white', linewidth=0.5)
        ax.bar(x[i] + width/2, vals_s3[i], width,
               color=SUB_COLORS['s3'], hatch=h, edgecolor='white', linewidth=0.5)

    # legend: subject color patches + method hatch patches
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=SUB_COLORS['s2'], label='Subject 2'),
        Patch(facecolor=SUB_COLORS['s3'], label='Subject 3'),
    ] + [
        Patch(facecolor='gray', hatch=h, edgecolor='white', label=LABELS[m])
        for m, h in zip(METHODS, HATCHES)
    ]
    ax.legend(handles=legend_handles, fontsize=7.5, ncol=2)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in METHODS], rotation=15, ha='right', fontsize=9)
    ax.set_ylabel(metric_label)
    ax.set_title(metric_label)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

fig.suptitle('Ridge Regression Performance — All Embedding Methods', fontsize=13)
fig.tight_layout()
path = os.path.join(FIGURES_DIR, 'comparison_bar_all_methods.pdf')
fig.savefig(path, bbox_inches='tight')
fig.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
print(f'\nSaved {path}')


# ── Figure 2: overlaid histograms per subject ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, sid in zip(axes, SUBJECTS):
    for m in METHODS:
        c = corrs[(sid, m)]
        ax.hist(c, bins=100, alpha=0.45, color=COLORS[m],
                label=f'{LABELS[m]} (top1%={np.percentile(c,99):.3f})',
                density=True, edgecolor='none')
    ax.set_xlabel('Pearson CC')
    ax.set_ylabel('Density')
    ax.set_title(f'CC Distribution — {SUB_LABELS[sid]}')
    ax.legend(fontsize=7.5)
    ax.set_xlim(-0.05, 0.20)

fig.suptitle('CC Distributions Across All Embedding Methods', fontsize=13)
fig.tight_layout()
path = os.path.join(FIGURES_DIR, 'comparison_histograms.pdf')
fig.savefig(path, bbox_inches='tight')
fig.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
print(f'Saved {path}')


# ── Figure 3: scatter plots — Word2Vec vs BERT variants (subject 3) ───────────
# Use voxel-sorted arrays so both methods index the same voxel at each position
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
ref_method = 'word2vec'
compare_methods = ['pretrained_bert', 'finetuned_lora']
sid = 's3'

for ax, m in zip(axes, compare_methods):
    x_vals = corrs_sorted[(sid, ref_method)]
    y_vals = corrs_sorted[(sid, m)]
    # align lengths in case voxel counts differ slightly
    n = min(len(x_vals), len(y_vals))
    x_vals, y_vals = x_vals[:n], y_vals[:n]
    r, _ = pearsonr(x_vals, y_vals)

    # subsample for readability
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=min(10000, n), replace=False)
    ax.scatter(x_vals[idx], y_vals[idx], s=1, alpha=0.3,
               color=COLORS[m], rasterized=True)
    ax.plot([-0.05, 0.3], [-0.05, 0.3], 'k--', lw=0.8, alpha=0.5)
    ax.set_xlabel(f'{LABELS[ref_method]} CC')
    ax.set_ylabel(f'{LABELS[m]} CC')
    ax.set_title(f'{LABELS[ref_method]} vs {LABELS[m]}\n'
                 f'{SUB_LABELS[sid]}  (r={r:.3f})')
    ax.set_xlim(-0.05, 0.20)
    ax.set_ylim(-0.05, 0.20)

fig.suptitle('Per-Voxel CC: Word2Vec vs BERT — do they predict the same voxels?',
             fontsize=12)
fig.tight_layout()
path = os.path.join(FIGURES_DIR, 'comparison_scatter_word2vec_vs_bert.pdf')
fig.savefig(path, bbox_inches='tight', dpi=150)
fig.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
print(f'Saved {path}')


# ── Figure 4: gain map — finetuned_lora vs word2vec ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, sid in zip(axes, SUBJECTS):
    lora = corrs_sorted[(sid, 'finetuned_lora')]
    w2v  = corrs_sorted[(sid, 'word2vec')]
    n    = min(len(lora), len(w2v))
    gain = lora[:n] - w2v[:n]
    pct_improved = 100 * np.mean(gain > 0)
    ax.hist(gain, bins=100, color=COLORS['finetuned_lora'], alpha=0.8, edgecolor='none')
    ax.axvline(0, color='black', lw=1.2, ls='--')
    ax.axvline(np.mean(gain), color='red', lw=1.2, ls='-',
               label=f'Mean gain = {np.mean(gain):.4f}')
    ax.set_xlabel('CC gain (LoRA − Word2Vec)')
    ax.set_ylabel('Voxels')
    ax.set_title(f'{SUB_LABELS[sid]}\n{pct_improved:.1f}% of voxels improved')
    ax.legend(fontsize=9)

fig.suptitle('Per-Voxel CC Gain: Fine-tuned LoRA vs Word2Vec', fontsize=12)
fig.tight_layout()
path = os.path.join(FIGURES_DIR, 'comparison_gain_lora_vs_word2vec.pdf')
fig.savefig(path, bbox_inches='tight')
fig.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
print(f'Saved {path}')


# ── Figure 5: voxel overlap — top-5% voxels shared across methods ─────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, sid in zip(axes, SUBJECTS):
    n_vox = len(corrs[(sid, 'word2vec')])
    threshold_pct = 95  # top 5%

    # get top-5% voxel sets per method (use sorted arrays so indices = voxel IDs)
    top_sets = {}
    for m in METHODS:
        c = corrs_sorted[(sid, m)]
        thresh = np.percentile(c, threshold_pct)
        top_sets[m] = set(np.where(c > thresh)[0])

    # pairwise overlap matrix
    n = len(METHODS)
    overlap = np.zeros((n, n))
    for i, m1 in enumerate(METHODS):
        for j, m2 in enumerate(METHODS):
            shared = len(top_sets[m1] & top_sets[m2])
            union  = len(top_sets[m1] | top_sets[m2])
            overlap[i, j] = shared / union if union > 0 else 0  # Jaccard

    im = ax.imshow(overlap, vmin=0, vmax=1, cmap='YlOrRd')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([LABELS[m] for m in METHODS], rotation=30, ha='right', fontsize=8)
    ax.set_yticklabels([LABELS[m] for m in METHODS], fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{overlap[i,j]:.2f}', ha='center', va='center',
                    fontsize=8, color='black' if overlap[i,j] < 0.7 else 'white')
    ax.set_title(f'{SUB_LABELS[sid]}\nJaccard overlap of top-5% voxels')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle('Do Different Methods Predict the Same Voxels?\n(Jaccard similarity of top-5% voxel sets)',
             fontsize=12)
fig.tight_layout()
path = os.path.join(FIGURES_DIR, 'comparison_voxel_overlap.pdf')
fig.savefig(path, bbox_inches='tight')
fig.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
print(f'Saved {path}')


print('\nDone. All figures saved to results/figures/')
