#!/usr/bin/env python3
"""
check2_crosssubject.py — Comprehensive Check 2 visualizations.

Cross-Subject Replication: does the GloVe encoding model capture a
generalizable language-to-brain mapping that replicates across individuals?

Five panels (saved individually + combined overview):
  A  Story-level mean CC scatter  (s2 vs s3, regression + labels)
  B  Per-story sorted bar chart   (side-by-side s2/s3, 20 stories)
  C  Voxelwise CC distributions   (both subjects, GloVe)
  D  Functional PCA of top-5% voxel weights  (s2 and s3 side-by-side)
  E  Method-level cross-subject mean CC scatter  (all 8 methods)

Run from repo root:
    python code/analysis/check2_crosssubject.py

Outputs saved to results/stability/  as  check2_v2_{A-E}.{pdf,png}
and  check2_v2_overview.{pdf,png}.
"""

import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ── paths ────────────────────────────────────────────────────────────────────
REPO     = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
STAB_DIR = os.path.join(REPO, "results/stability")
METRICS  = os.path.join(REPO, "results/metrics")

# ── method registry ───────────────────────────────────────────────────────────
METHOD_META = {
    "bow":                      ("BoW",             "#AAAAAA"),
    "word2vec":                 ("Word2Vec",         "#888888"),
    "glove":                    ("GloVe",            "#8DA0CB"),
    "vanilla_bert":             ("Vanilla BERT",     "#66C2A5"),
    "pretrained_bert":          ("Enhanced BERT",    "#FC8D62"),
    "finetuned_lora":           ("LoRA (last4)",     "#E78AC3"),
    "finetuned_lora_improved":  ("LoRA (improved)",  "#A6D854"),
    "finetuned_lora_mid4_mean": ("LoRA (mid4★)","#FFD92F"),
}
ORDERED_METHODS = list(METHOD_META.keys())

# Colour palette for subjects (used consistently across panels)
COL_S2, COL_S3 = "#4878CF", "#D65F5F"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_corr(sid: str, method: str):
    p = os.path.join(METRICS, f"{sid}_{method}_test_corrs.npy")
    return np.load(p) if os.path.exists(p) else None


def functional_pca(weights_top: np.ndarray):
    """2-D PCA of (n_feat, n_voxels) weight matrix; returns (n_voxels, 2) coords + var_exp."""
    W   = weights_top.T                            # (n_voxels, n_feat)
    W_c = W - W.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(W_c, full_matrices=False)
    pcs     = U[:, :2] * S[:2]
    var_exp = S[:2] ** 2 / (S ** 2).sum()
    return pcs, var_exp


def clean_story(name: str, maxlen: int = 26) -> str:
    """Convert camelCase / lowercase story names to readable short labels."""
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    s = s.replace("part1", "Pt1").replace("part2", "Pt2").replace("part3", "Pt3")
    s = s[:1].upper() + s[1:]
    return s[:maxlen]


def savefig(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(STAB_DIR, f"{name}.{ext}"),
                    dpi=150, bbox_inches="tight")
    print(f"  saved  {name}.pdf/png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Panel A — Story-level CC scatter
# ══════════════════════════════════════════════════════════════════════════════

def panel_A(stories, v2, v3, r_story, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 6))

    avg_cc = (v2 + v3) / 2
    sc = ax.scatter(v2, v3, c=avg_cc, cmap="viridis",
                    s=65, zorder=4, edgecolors="#444", lw=0.5)

    # Regression line
    slope, intercept, *_ = stats.linregress(v2, v3)
    x_lo, x_hi = v2.min() - 0.003, v2.max() + 0.003
    xs = np.linspace(x_lo, x_hi, 200)
    ax.plot(xs, slope * xs + intercept, "k--", lw=1.5, alpha=0.75,
            label=f"Regression  r = {r_story:.3f}")

    # y = x reference
    lo = min(v2.min(), v3.min()) - 0.003
    hi = max(v2.max(), v3.max()) + 0.003
    ax.plot([lo, hi], [lo, hi], color="#999", lw=0.8, ls=":", label="y = x")

    # Story labels
    short = {s: clean_story(s) for s in stories}
    for i, s in enumerate(stories):
        ax.annotate(short[s], (v2[i], v3[i]),
                    fontsize=5.5, xytext=(3, 3),
                    textcoords="offset points", alpha=0.85)

    if standalone:
        plt.colorbar(sc, ax=ax, label="Mean CC (s2+s3)/2", shrink=0.85)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Subject 2 — Story mean CC (GloVe)", fontsize=10)
    ax.set_ylabel("Subject 3 — Story mean CC (GloVe)", fontsize=10)
    ax.set_title(f"(A)  Story-level Encoding Agreement\nPearson r = {r_story:.3f},  n = 20 test stories",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    if standalone:
        fig.tight_layout()
        savefig(fig, "check2_v2_A_story_scatter")


# ══════════════════════════════════════════════════════════════════════════════
# Panel B — Per-story bar chart (sorted by average CC)
# ══════════════════════════════════════════════════════════════════════════════

def panel_B(stories, v2, v3, mean_s2_all, mean_s3_all, r_story, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(13, 4.5))

    avg_cc   = (v2 + v3) / 2
    sort_idx = np.argsort(avg_cc)[::-1]
    labels   = [clean_story(stories[i], maxlen=28) for i in sort_idx]
    v2s, v3s = v2[sort_idx], v3[sort_idx]

    x = np.arange(len(stories))
    w = 0.38
    ax.bar(x - w / 2, v2s, w, color=COL_S2, alpha=0.85, edgecolor="#666",
           label="Subject 2")
    ax.bar(x + w / 2, v3s, w, color=COL_S3, alpha=0.85, edgecolor="#666",
           label="Subject 3")

    # Global mean lines
    ax.axhline(mean_s2_all, color=COL_S2, lw=1.2, ls="--", alpha=0.6)
    ax.axhline(mean_s3_all, color=COL_S3, lw=1.2, ls="--", alpha=0.6)
    ax.text(len(stories) - 0.4, mean_s2_all + 0.0003, f"s2 mean", fontsize=7,
            color=COL_S2, ha="right", va="bottom")
    ax.text(len(stories) - 0.4, mean_s3_all + 0.0003, f"s3 mean", fontsize=7,
            color=COL_S3, ha="right", va="bottom")

    # Mark stories where s2 > s3 (only 3 cases)
    for xi, (a, b) in enumerate(zip(v2s, v3s)):
        if a > b:
            ax.text(xi, max(a, b) + 0.0005, "★", ha="center",
                    fontsize=7, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Mean per-voxel CC (GloVe)", fontsize=10)
    ax.set_title(
        f"(B)  Per-Story Encoding Accuracy — Both Subjects  (sorted by mean CC)\n"
        f"★ = stories where s2 > s3  |  Pearson r(s2, s3) = {r_story:.3f}",
        fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    if standalone:
        fig.tight_layout()
        savefig(fig, "check2_v2_B_story_bars")


# ══════════════════════════════════════════════════════════════════════════════
# Panel C — Voxelwise CC distributions
# ══════════════════════════════════════════════════════════════════════════════

def panel_C(corrs_s2, corrs_s3, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    lo = min(corrs_s2.min(), corrs_s3.min())
    hi = max(corrs_s2.max(), corrs_s3.max())
    bins = np.linspace(lo, hi, 130)

    for corrs, col, sid in [(corrs_s2, COL_S2, "s2"), (corrs_s3, COL_S3, "s3")]:
        med  = np.median(corrs)
        mean = corrs.mean()
        p99  = np.percentile(corrs, 99)
        lbl  = (f"Subject {sid[1]}  "
                f"n={len(corrs):,}  "
                f"mean={mean:.4f}  "
                f"med={med:.4f}")
        ax.hist(corrs, bins=bins, density=True, alpha=0.55, color=col, label=lbl)
        ax.axvline(mean, color=col, lw=1.8, ls="--")
        ax.axvline(p99,  color=col, lw=1.0, ls=":",
                   label=f"  top-1% threshold ({sid}): {p99:.3f}")

    ax.axvline(0, color="black", lw=0.8, ls=":", alpha=0.6)

    # Positive-voxel annotation
    pct2 = 100 * (corrs_s2 > 0).mean()
    pct3 = 100 * (corrs_s3 > 0).mean()
    ax.text(0.97, 0.97,
            f"Positive voxels\ns2: {pct2:.1f}%\ns3: {pct3:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.set_xlabel("Per-voxel Pearson r  (GloVe, test set)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("(C)  Voxelwise CC Distributions — Subject 2 vs. Subject 3",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, ncol=2)

    if standalone:
        fig.tight_layout()
        savefig(fig, "check2_v2_C_distributions")


# ══════════════════════════════════════════════════════════════════════════════
# Panel D — Functional PCA (s2 and s3 side-by-side)
# ══════════════════════════════════════════════════════════════════════════════

def panel_D(corrs_s2, corrs_s3, top_w_s2, top_w_s3, axes=None):
    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    vmin_global = 0
    vmax_global = max(np.percentile(corrs_s2[corrs_s2 >= np.percentile(corrs_s2, 95)], 99),
                      np.percentile(corrs_s3[corrs_s3 >= np.percentile(corrs_s3, 95)], 99))

    for ax, (sid, corrs, w_top) in zip(axes, [
        ("s2", corrs_s2, top_w_s2),
        ("s3", corrs_s3, top_w_s3),
    ]):
        mask   = corrs >= np.percentile(corrs, 95)
        cc_top = corrs[mask]
        pcs, var_exp = functional_pca(w_top)

        sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=cc_top, cmap="viridis",
                        s=7, alpha=0.65, vmin=vmin_global, vmax=vmax_global)
        ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% var)", fontsize=9)
        ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% var)", fontsize=9)
        ax.set_title(
            f"Subject {sid[1]}  —  top-5% voxels  (n = {mask.sum():,})\n"
            f"Functional PCA of GloVe weight vectors",
            fontsize=9)
        plt.colorbar(sc, ax=ax, label="Test CC", shrink=0.85)

    if standalone:
        fig.suptitle(
            "(D)  Cross-Subject Functional PCA\n"
            "High-CC voxels (yellow) cluster in a similar region of embedding-preference space "
            "for both subjects,\nindicating shared functional organisation despite different voxel counts.",
            fontsize=9, fontweight="bold")
        fig.tight_layout()
        savefig(fig, "check2_v2_D_pca")


# ══════════════════════════════════════════════════════════════════════════════
# Panel E — Method-level cross-subject mean CC scatter
# ══════════════════════════════════════════════════════════════════════════════

def panel_E(ax=None):
    """
    Each point = one embedding method.
    x = s2 mean CC,  y = s3 mean CC.
    The s3/s2 ratio decreasing monotonically with model quality shows that
    better embeddings converge more across subjects.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))

    ms2, ms3, labels, colors, ratios = [], [], [], [], []
    for method in ORDERED_METHODS:
        c2 = load_corr("s2", method)
        c3 = load_corr("s3", method)
        if c2 is None or c3 is None:
            continue
        m2, m3 = float(c2.mean()), float(c3.mean())
        ms2.append(m2); ms3.append(m3)
        labels.append(METHOD_META[method][0])
        colors.append(METHOD_META[method][1])
        ratios.append(m3 / m2 if m2 > 0 else 0)

    ms2, ms3 = np.array(ms2), np.array(ms3)

    # y = x reference
    lo = min(ms2.min(), ms3.min()) - 0.003
    hi = max(ms2.max(), ms3.max()) + 0.003
    ax.plot([lo, hi], [lo, hi], color="#999", lw=0.9, ls=":", label="y = x  (equal CC)")

    # Points
    for x, y, lbl, col, ratio in zip(ms2, ms3, labels, colors, ratios):
        ax.scatter(x, y, s=130, color=col, edgecolors="#333", lw=0.7, zorder=4)
        ax.annotate(
            f"{lbl}\n(×{ratio:.2f})",
            (x, y), fontsize=7.5,
            xytext=(6, -4), textcoords="offset points",
        )

    # Arrow from BoW to mid4 to show convergence direction
    ax.annotate("", xy=(ms2[-1], ms3[-1]), xytext=(ms2[0], ms3[0]),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.0))
    ax.text((ms2[0] + ms2[-1]) / 2 - 0.003,
            (ms3[0] + ms3[-1]) / 2 + 0.003,
            "improving model", fontsize=7, color="#555",
            rotation=44, ha="left")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Subject 2 — Mean per-voxel CC", fontsize=10)
    ax.set_ylabel("Subject 3 — Mean per-voxel CC", fontsize=10)
    ax.set_title(
        "(E)  Cross-Subject CC by Embedding Method\n"
        "s3/s2 ratio (×) shrinks as models improve: subjects converge",
        fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    if standalone:
        fig.tight_layout()
        savefig(fig, "check2_v2_E_method_scatter")

    return ms2, ms3, labels, colors, ratios


# ══════════════════════════════════════════════════════════════════════════════
# Combined overview figure
# ══════════════════════════════════════════════════════════════════════════════

def overview(stories, v2, v3, r_story, corrs_s2, corrs_s3,
             top_w_s2, top_w_s3, mean_s2_all, mean_s3_all):
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 4, figure=fig,
                            hspace=0.52, wspace=0.38)

    ax_A  = fig.add_subplot(gs[0, 0:2])
    ax_E  = fig.add_subplot(gs[0, 2:4])
    ax_B  = fig.add_subplot(gs[1, 0:4])
    ax_C  = fig.add_subplot(gs[2, 0:2])
    ax_D1 = fig.add_subplot(gs[2, 2])
    ax_D2 = fig.add_subplot(gs[2, 3])

    panel_A(stories, v2, v3, r_story, ax=ax_A)
    panel_B(stories, v2, v3, mean_s2_all, mean_s3_all, r_story, ax=ax_B)
    panel_C(corrs_s2, corrs_s3, ax=ax_C)
    panel_D(corrs_s2, corrs_s3, top_w_s2, top_w_s3, axes=[ax_D1, ax_D2])
    panel_E(ax=ax_E)

    fig.suptitle(
        "Check 2 — Cross-Subject Replication (GloVe)\n"
        "Does the encoding model capture a generalizable language-to-brain mapping?",
        fontsize=13, fontweight="bold", y=1.01)

    savefig(fig, "check2_v2_overview")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(STAB_DIR, exist_ok=True)

    # Load story-level scores
    with open(os.path.join(STAB_DIR, "stability_summary.json")) as f:
        summary = json.load(f)
    c2_meta  = summary["check2"]
    ss2, ss3 = c2_meta["story_scores_s2"], c2_meta["story_scores_s3"]
    stories  = sorted(ss2.keys())
    v2 = np.array([ss2[s] for s in stories])
    v3 = np.array([ss3[s] for s in stories])
    r_story = float(np.corrcoef(v2, v3)[0, 1])

    # Load per-voxel CC arrays and weight matrices
    corrs_s2 = np.load(os.path.join(STAB_DIR, "s2_glove_test_corrs.npy"))
    corrs_s3 = np.load(os.path.join(STAB_DIR, "s3_glove_test_corrs.npy"))
    top_w_s2 = np.load(os.path.join(STAB_DIR, "s2_glove_top5pct_weights.npy"))
    top_w_s3 = np.load(os.path.join(STAB_DIR, "s3_glove_top5pct_weights.npy"))

    mean_s2_all = float(corrs_s2.mean())
    mean_s3_all = float(corrs_s3.mean())

    print(f"Loaded: s2 {corrs_s2.shape}  s3 {corrs_s3.shape}  "
          f"story r = {r_story:.4f}")

    print("\nGenerating Panel A — story scatter...")
    panel_A(stories, v2, v3, r_story)

    print("Generating Panel B — story bar chart...")
    panel_B(stories, v2, v3, mean_s2_all, mean_s3_all, r_story)

    print("Generating Panel C — CC distributions...")
    panel_C(corrs_s2, corrs_s3)

    print("Generating Panel D — functional PCA...")
    panel_D(corrs_s2, corrs_s3, top_w_s2, top_w_s3)

    print("Generating Panel E — method-level scatter...")
    panel_E()

    print("Generating combined overview figure...")
    overview(stories, v2, v3, r_story, corrs_s2, corrs_s3,
             top_w_s2, top_w_s3, mean_s2_all, mean_s3_all)

    print(f"\nAll figures saved to {STAB_DIR}/")


if __name__ == "__main__":
    main()
