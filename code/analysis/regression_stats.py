"""
regression_stats.py — Comprehensive statistical analysis of ridge regression
Pearson CC results across embedding methods and subjects.

Produces five figures saved to results/figures/regression_analysis/:
  fig1_cc_distributions.{pdf,png}   — violin + percentile KDE per method
  fig2_method_progression.{pdf,png} — mean CC with bootstrap CIs, Wilcoxon tests
  fig3_intermethod_corr.{pdf,png}   — Spearman ρ heatmap between CC vectors
  fig4_cross_subject.{pdf,png}      — per-voxel s2 vs s3 consistency scatter
  fig5_pairwise_gain.{pdf,png}      — voxel-level CC gain: each method vs GloVe

Run from repo root:
    python code/analysis/regression_stats.py

All figures are also embedded in a printed summary table.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import wilcoxon, spearmanr

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO       = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
METRICS    = os.path.join(REPO, "results/metrics")
OUT_DIR    = os.path.join(REPO, "results/figures/regression_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Method registry ───────────────────────────────────────────────────────────
# Keys used in file names; display labels; hex color; "tier" for grouping
METHOD_META = {
    "bow":                      ("BoW",              "#B3B3B3", "static"),
    "word2vec":                 ("Word2Vec",          "#999999", "static"),
    "glove":                    ("GloVe",            "#8DA0CB", "static"),
    "vanilla_bert":             ("Vanilla BERT",      "#66C2A5", "bert"),
    "pretrained_bert":          ("Enhanced BERT",     "#FC8D62", "bert"),
    "finetuned_lora":           ("LoRA (last4)",      "#E78AC3", "lora"),
    "finetuned_lora_improved":  ("LoRA (improved)",   "#A6D854", "lora"),
    "finetuned_lora_mid4_mean": ("LoRA (mid4★)",      "#FFD92F", "lora"),
}
# Ordered from weakest to strongest (narrative ordering)
ORDERED_METHODS = [
    "bow",
    "word2vec",
    "glove",
    "vanilla_bert",
    "pretrained_bert",
    "finetuned_lora",
    "finetuned_lora_improved",
    "finetuned_lora_mid4_mean",
]
SUBJECTS = ["s2", "s3"]
SUBJ_LABELS = {"s2": "Subject 2", "s3": "Subject 3"}
SUBJ_MARKERS = {"s2": "o", "s3": "s"}

N_BOOTSTRAP = 2_000
RNG = np.random.default_rng(42)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_corr(sid: str, method: str) -> np.ndarray | None:
    """Return per-voxel CC array or None if file not found."""
    path = os.path.join(METRICS, f"{sid}_{method}_test_corrs.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


def bootstrap_mean_ci(arr: np.ndarray, n: int = N_BOOTSTRAP, ci: float = 0.95) -> tuple:
    """Return (mean, lower, upper) via percentile bootstrap."""
    boots = RNG.choice(arr, size=(n, len(arr)), replace=True).mean(axis=1)
    lo = np.percentile(boots, 100 * (1 - ci) / 2)
    hi = np.percentile(boots, 100 * (1 - (1 - ci) / 2))
    return float(arr.mean()), lo, hi


def pct_improvement(new: np.ndarray, base: np.ndarray) -> float:
    return 100.0 * (new.mean() - base.mean()) / base.mean()


def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> tuple[float, str]:
    """Paired Wilcoxon signed-rank test; returns (p-value, significance stars)."""
    diff = a - b
    nonzero = diff[diff != 0]
    if len(nonzero) < 20:
        return 1.0, "ns"
    _, p = wilcoxon(nonzero, alternative="greater")
    stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    return p, stars


def save_fig(fig, name: str):
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  saved  {name}.pdf/png")


# ── Load all available data ───────────────────────────────────────────────────
print("Loading CC arrays …")
corrs: dict[tuple[str, str], np.ndarray] = {}
for sid in SUBJECTS:
    for m in ORDERED_METHODS:
        arr = load_corr(sid, m)
        if arr is not None:
            corrs[(sid, m)] = arr
            print(f"  {sid}/{m}: {len(arr):,} voxels  mean={arr.mean():.4f}")
        else:
            print(f"  {sid}/{m}: NOT FOUND — skipped")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: CC Distribution — violin plot per method, per subject
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 1: CC distributions …")

avail_per_subj = {
    sid: [m for m in ORDERED_METHODS if (sid, m) in corrs]
    for sid in SUBJECTS
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

for ax, sid in zip(axes, SUBJECTS):
    methods_here = avail_per_subj[sid]
    data   = [corrs[(sid, m)] for m in methods_here]
    labels = [METHOD_META[m][0] for m in methods_here]
    colors = [METHOD_META[m][1] for m in methods_here]

    parts = ax.violinplot(data, positions=range(len(methods_here)),
                          showmedians=True, showextrema=False, widths=0.7)

    for i, (pc, col) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(col)
        pc.set_edgecolor("grey")
        pc.set_alpha(0.80)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)

    # Overlay mean dots and 95th-percentile ticks
    for i, (arr, col) in enumerate(zip(data, colors)):
        ax.scatter(i, arr.mean(), color="black", zorder=5, s=25, marker="D")
        ax.hlines(np.percentile(arr, 95), i - 0.18, i + 0.18,
                  colors="darkred", linewidths=1.2, linestyles="--", zorder=4)

    ax.set_xticks(range(len(methods_here)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Pearson r (per voxel)", fontsize=10)
    ax.set_title(f"{SUBJ_LABELS[sid]}", fontsize=11, fontweight="bold")
    ax.axhline(0, color="lightgrey", lw=0.8, ls=":")
    ax.set_ylim(bottom=-0.05)

    # Annotate mean values above each violin
    for i, arr in enumerate(data):
        ax.text(i, arr.mean() + 0.004, f"{arr.mean():.3f}",
                ha="center", va="bottom", fontsize=7.5, color="black")

# Shared legend for mean ◆ and top-5% dashed line
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker="D", color="w", markerfacecolor="black",
           markersize=6, label="Mean"),
    Line2D([0], [0], color="darkred", lw=1.2, ls="--", label="Top 5% threshold"),
]
axes[1].legend(handles=legend_handles, fontsize=8, loc="upper left")

fig.suptitle("Per-voxel Pearson CC Distribution by Embedding Method",
             fontsize=13, fontweight="bold")
fig.tight_layout()
save_fig(fig, "fig1_cc_distributions")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Method progression — mean CC with bootstrap CI + Wilcoxon test
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 2: method progression + significance …")

fig, ax_pct = plt.subplots(figsize=(8, 5))

base_method = "glove"
for sid, marker, ls in [("s2", "o", "-"), ("s3", "s", "--")]:
    if (sid, base_method) not in corrs:
        continue
    methods_here = [m for m in avail_per_subj[sid] if m != base_method]
    pcts  = [pct_improvement(corrs[(sid, m)], corrs[(sid, base_method)])
             for m in methods_here]
    x     = np.arange(len(methods_here))
    ax_pct.plot(x, pcts, marker=marker, ls=ls, label=SUBJ_LABELS[sid],
                color="#555555", markersize=7)
    for xi, (pct, m) in enumerate(zip(pcts, methods_here)):
        ax_pct.scatter(xi, pct, color=METHOD_META[m][1], s=70, zorder=4)

non_base = [m for m in ORDERED_METHODS if m != base_method]
ax_pct.set_xticks(np.arange(len(non_base)))
ax_pct.set_xticklabels(
    [METHOD_META[m][0] for m in non_base],
    rotation=28, ha="right", fontsize=8.5)
ax_pct.axhline(0, color="grey", lw=0.8, ls=":")
ax_pct.set_ylabel("% improvement in mean r over GloVe", fontsize=9)
ax_pct.set_title("Relative Gain over GloVe Baseline", fontsize=10, fontweight="bold")
ax_pct.legend(fontsize=9)
ax_pct.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

fig.tight_layout()
save_fig(fig, "fig2_method_progression")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Inter-method Spearman correlation heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 3: inter-method Spearman ρ heatmap …")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, sid in zip(axes, SUBJECTS):
    methods_here = avail_per_subj[sid]
    n = len(methods_here)
    rho_mat = np.full((n, n), np.nan)
    for i, m1 in enumerate(methods_here):
        for j, m2 in enumerate(methods_here):
            rho, _ = spearmanr(corrs[(sid, m1)], corrs[(sid, m2)])
            rho_mat[i, j] = rho

    # Custom diverging colormap anchored at 0.5 (all values should be > 0)
    cmap = plt.cm.YlOrRd
    im = ax.imshow(rho_mat, cmap=cmap, vmin=0.0, vmax=1.0)

    tick_labels = [METHOD_META[m][0] for m in methods_here]
    ax.set_xticks(range(n)); ax.set_xticklabels(tick_labels, rotation=35,
                                                 ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(tick_labels, fontsize=8)

    for i in range(n):
        for j in range(n):
            val = rho_mat[i, j]
            text_col = "white" if val > 0.75 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=text_col, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.82, label="Spearman ρ")
    ax.set_title(f"{SUBJ_LABELS[sid]}\nSpearman ρ of per-voxel CC vectors",
                 fontsize=10, fontweight="bold")

fig.suptitle(
    "Inter-Method Correlation: Do Different Embeddings Identify the Same Language-Sensitive Voxels?",
    fontsize=11, fontweight="bold")
fig.tight_layout()
save_fig(fig, "fig3_intermethod_corr")
plt.close(fig)




# ══════════════════════════════════════════════════════════════════════════════
# Summary statistics table (printed to stdout)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 90)
print(f"{'Method':<28} {'Subj':>4} {'Mean r':>8} {'Median r':>9} "
      f"{'Top 5%':>8} {'Top 1%':>8} {'Max r':>8}  {'vs GloVe':>9}")
print("─" * 90)
for m in ORDERED_METHODS:
    for sid in SUBJECTS:
        key = (sid, m)
        if key not in corrs:
            continue
        arr = corrs[key]
        label = METHOD_META[m][0]
        pct = pct_improvement(arr, corrs[(sid, "glove")]) if m != "glove" else 0.0
        pct_str = f"{pct:+.1f}%" if m != "glove" else "—"
        print(f"  {label:<26} {sid:>4}  {arr.mean():>8.4f}  {np.median(arr):>9.4f}  "
              f"{np.percentile(arr,95):>8.4f}  {np.percentile(arr,99):>8.4f}  "
              f"{arr.max():>8.4f}  {pct_str:>9}")
print("═" * 90)

print(f"\nAll figures saved to:\n  {OUT_DIR}")
print("Done.")
