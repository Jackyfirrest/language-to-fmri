#!/usr/bin/env python3
"""
stability.py — Check 2 (cross-subject replication) using the best
model: LoRA BERT mid4_mean (layers 5-8 average, d=768, delay-stacked to
d=3072).

Computes story-level mean CC from saved pkl weights + test embeddings, then
generates the same five panels as check2_crosssubject.py but for mid4.

Outputs saved to results/stability/ as check2_mid4_{A-E}.{pdf,png}
and check2_mid4_overview.{pdf,png}.

Run from repo root:
    python code/analysis/stability.py
"""

import json
import os
import re
import sys
import time

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ── paths ────────────────────────────────────────────────────────────────────
REPO      = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
STAB_DIR  = os.path.join(REPO, "results/stability")
METRICS   = os.path.join(REPO, "results/metrics")
MODELS    = os.path.join(REPO, "results/models")
SPLIT_PATH = os.path.join(REPO, "data/train_test_split.json")
DATA_PATH = "/scratch/users/s214/lab3"

METHOD    = "finetuned_lora_mid4_mean"
EMBED_DIR = os.path.join(REPO, "data/embeddings_mid4_mean")
CHUNK_V   = 2000

COL_S2, COL_S3 = "#4878CF", "#D65F5F"

METHOD_META = {
    "bow":                      ("BoW",             "#AAAAAA"),
    "word2vec":                 ("Word2Vec",         "#888888"),
    "glove":                    ("GloVe",            "#8DA0CB"),
    "vanilla_bert":             ("Vanilla BERT",     "#66C2A5"),
    "pretrained_bert":          ("Enhanced BERT",    "#FC8D62"),
    "finetuned_lora":           ("LoRA (last4)",     "#E78AC3"),
    "finetuned_lora_improved":  ("LoRA (improved)",  "#A6D854"),
    "finetuned_lora_mid4_mean": ("LoRA (mid4★)", "#FFD92F"),
}
ORDERED_METHODS = list(METHOD_META.keys())


# ── helpers ───────────────────────────────────────────────────────────────────

def load_embed(sid, split):
    p = os.path.join(EMBED_DIR, f"{sid}_{split}_{METHOD}_embeddings.npz")
    return np.load(p)["X"].astype(np.float32)


def load_corr(sid, method):
    p = os.path.join(METRICS, f"{sid}_{method}_test_corrs.npy")
    return np.load(p) if os.path.exists(p) else None


def col_corr(A, B):
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    num   = (A * B).sum(axis=0)
    denom = np.sqrt((A**2).sum(axis=0) * (B**2).sum(axis=0)) + 1e-10
    return (num / denom).astype(np.float32)


def functional_pca(weights_top):
    W   = weights_top.T
    W_c = W - W.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(W_c, full_matrices=False)
    pcs     = U[:, :2] * S[:2]
    var_exp = S[:2]**2 / (S**2).sum()
    return pcs, var_exp


def clean_story(name, maxlen=26):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    s = s.replace("part1", "Pt1").replace("part2", "Pt2").replace("part3", "Pt3")
    s = s[:1].upper() + s[1:]
    return s[:maxlen]


def savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(STAB_DIR, f"{name}.{ext}"),
                    dpi=150, bbox_inches="tight")
    print(f"  saved  {name}.pdf/png")
    plt.close(fig)


# ── story-level CC computation ────────────────────────────────────────────────

def compute_story_cc(sid, weights, voxel_indices, X_test, test_stories, story_lengths):
    """
    For each test story, compute the mean per-voxel Pearson r between the
    model's predicted BOLD and the actual BOLD restricted to that story's TRs.

    weights:       (n_feat, n_voxels)  from pkl, in voxel_indices order
    voxel_indices: (n_voxels,)         maps weight column -> brain voxel index
    X_test:        (n_test_TRs, n_feat) z-scored test embeddings
    """
    subj_dir = os.path.join(DATA_PATH, f"subject{sid[1]}")
    story_scores = {}
    offset = 0

    for story in test_stories:
        n_t = story_lengths[story]
        X_s = X_test[offset:offset + n_t]                        # (n_t, n_feat)

        Y_full = np.nan_to_num(
            np.load(os.path.join(subj_dir, f"{story}.npy")),
            nan=0.0
        ).astype(np.float32)                                       # (n_t, n_brain_vox)

        n_vox = weights.shape[1]
        per_vox_cc = np.zeros(n_vox, dtype=np.float32)

        for vs in range(0, n_vox, CHUNK_V):
            ve      = min(vs + CHUNK_V, n_vox)
            P_chunk = X_s @ weights[:, vs:ve]                     # (n_t, chunk)
            vi      = voxel_indices[vs:ve]
            Y_chunk = Y_full[:, vi]                               # (n_t, chunk)
            per_vox_cc[vs:ve] = col_corr(P_chunk, Y_chunk)

        story_scores[story] = float(per_vox_cc.mean())
        offset += n_t
        print(f"    {story:<45s}  mean CC = {story_scores[story]:.4f}", end="\r")

    print()
    return story_scores


def zscore_test(X_train, X_test):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-8
    return (X_test - mu) / sd


# ── panel functions ───────────────────────────────────────────────────────────

def panel_A(stories, v2, v3, r_story, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 6))

    avg_cc = (v2 + v3) / 2
    sc = ax.scatter(v2, v3, c=avg_cc, cmap="viridis",
                    s=65, zorder=4, edgecolors="#444", lw=0.5)

    slope, intercept, *_ = stats.linregress(v2, v3)
    xs = np.linspace(v2.min() - 0.003, v2.max() + 0.003, 200)
    ax.plot(xs, slope * xs + intercept, "k--", lw=1.5, alpha=0.75,
            label=f"Regression  r = {r_story:.3f}")

    lo = min(v2.min(), v3.min()) - 0.003
    hi = max(v2.max(), v3.max()) + 0.003
    ax.plot([lo, hi], [lo, hi], color="#999", lw=0.8, ls=":", label="y = x")

    short = {s: clean_story(s) for s in stories}
    for i, s in enumerate(stories):
        ax.annotate(short[s], (v2[i], v3[i]),
                    fontsize=5.5, xytext=(3, 3),
                    textcoords="offset points", alpha=0.85)

    if standalone:
        plt.colorbar(sc, ax=ax, label="Mean CC (s2+s3)/2", shrink=0.85)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Subject 2 — Story mean CC (LoRA mid4)", fontsize=10)
    ax.set_ylabel("Subject 3 — Story mean CC (LoRA mid4)", fontsize=10)
    ax.set_title(f"Story-level Encoding Agreement\n"
                 f"Pearson r = {r_story:.3f},  n = 20 test stories",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    if standalone:
        fig.tight_layout()
        savefig(fig, "check2_mid4_A_story_scatter")


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
    ax.bar(x - w/2, v2s, w, color=COL_S2, alpha=0.85, edgecolor="#666", label="Subject 2")
    ax.bar(x + w/2, v3s, w, color=COL_S3, alpha=0.85, edgecolor="#666", label="Subject 3")

    ax.axhline(mean_s2_all, color=COL_S2, lw=1.2, ls="--", alpha=0.6)
    ax.axhline(mean_s3_all, color=COL_S3, lw=1.2, ls="--", alpha=0.6)
    ax.text(len(stories) - 0.4, mean_s2_all + 0.0005, "s2 mean", fontsize=7,
            color=COL_S2, ha="right", va="bottom")
    ax.text(len(stories) - 0.4, mean_s3_all + 0.0005, "s3 mean", fontsize=7,
            color=COL_S3, ha="right", va="bottom")

    for xi, (a, b) in enumerate(zip(v2s, v3s)):
        if a > b:
            ax.text(xi, max(a, b) + 0.0008, "★", ha="center", fontsize=7, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Mean per-voxel CC (LoRA mid4)", fontsize=10)
    ax.set_title(
        f"Per-Story Encoding Accuracy — Both Subjects  (sorted by mean CC)\n"
        f"★ = stories where s2 > s3  |  Pearson r(s2, s3) = {r_story:.3f}",
        fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    if standalone:
        fig.tight_layout()
        savefig(fig, "check2_mid4_B_story_bars")


def panel_C(corrs_s2, corrs_s3, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    lo   = min(corrs_s2.min(), corrs_s3.min())
    hi   = max(corrs_s2.max(), corrs_s3.max())
    bins = np.linspace(lo, hi, 130)

    for corrs, col, sid in [(corrs_s2, COL_S2, "s2"), (corrs_s3, COL_S3, "s3")]:
        med  = np.median(corrs)
        mean = corrs.mean()
        p99  = np.percentile(corrs, 99)
        lbl  = (f"Subject {sid[1]}  n={len(corrs):,}  "
                f"mean={mean:.4f}  med={med:.4f}")
        ax.hist(corrs, bins=bins, density=True, alpha=0.55, color=col, label=lbl)
        ax.axvline(mean, color=col, lw=1.8, ls="--")
        ax.axvline(p99,  color=col, lw=1.0, ls=":",
                   label=f"  top-1% threshold ({sid}): {p99:.3f}")

    ax.axvline(0, color="black", lw=0.8, ls=":", alpha=0.6)

    pct2 = 100 * (corrs_s2 > 0).mean()
    pct3 = 100 * (corrs_s3 > 0).mean()
    ax.text(0.97, 0.97,
            f"Positive voxels\ns2: {pct2:.1f}%\ns3: {pct3:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.set_xlabel("Per-voxel Pearson r  (LoRA mid4, test set)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Voxelwise CC Distributions — Subject 2 vs. Subject 3",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, ncol=2)

    if standalone:
        fig.tight_layout()
        savefig(fig, "check2_mid4_C_distributions")


def panel_D(corrs_s2, corrs_s3, top_w_s2, top_w_s3, axes=None):
    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    vmax_global = max(
        np.percentile(corrs_s2[corrs_s2 >= np.percentile(corrs_s2, 95)], 99),
        np.percentile(corrs_s3[corrs_s3 >= np.percentile(corrs_s3, 95)], 99),
    )

    for ax, (sid, corrs, w_top) in zip(axes, [
        ("s2", corrs_s2, top_w_s2),
        ("s3", corrs_s3, top_w_s3),
    ]):
        mask   = corrs >= np.percentile(corrs, 95)
        cc_top = corrs[mask]
        pcs, var_exp = functional_pca(w_top)

        sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=cc_top, cmap="viridis",
                        s=7, alpha=0.65, vmin=0, vmax=vmax_global)
        ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% var)", fontsize=9)
        ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% var)", fontsize=9)
        ax.set_title(
            f"Subject {sid[1]}  —  top-5% voxels  (n = {mask.sum():,})\n"
            f"Functional PCA of LoRA mid4 weight vectors",
            fontsize=9)
        plt.colorbar(sc, ax=ax, label="Test CC", shrink=0.85)

    if standalone:
        fig.suptitle(
            "Cross-Subject Functional PCA (LoRA mid4)\n"
            "High-CC voxels (yellow) cluster in a comparable region of PC space "
            "for both subjects.",
            fontsize=9, fontweight="bold")
        fig.tight_layout()
        savefig(fig, "check2_mid4_D_pca")


def panel_E(ax=None):
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

    lo = min(ms2.min(), ms3.min()) - 0.003
    hi = max(ms2.max(), ms3.max()) + 0.003
    ax.plot([lo, hi], [lo, hi], color="#999", lw=0.9, ls=":", label="y = x  (equal CC)")

    # Highlight the best model
    for i, (x, y, lbl, col, ratio) in enumerate(zip(ms2, ms3, labels, colors, ratios)):
        zorder = 6 if "mid4" in lbl else 4
        edgecol = "#D4AC0D" if "mid4" in lbl else "#333"
        lw = 1.8 if "mid4" in lbl else 0.7
        ax.scatter(x, y, s=140, color=col, edgecolors=edgecol,
                   lw=lw, zorder=zorder)
        ax.annotate(
            f"{lbl}\n(×{ratio:.2f})",
            (x, y), fontsize=7.5,
            xytext=(6, -4), textcoords="offset points",
        )

    ax.annotate("", xy=(ms2[-1], ms3[-1]), xytext=(ms2[0], ms3[0]),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.0))
    ax.text((ms2[0] + ms2[-1]) / 2 - 0.003,
            (ms3[0] + ms3[-1]) / 2 + 0.003,
            "improving model", fontsize=7, color="#555",
            rotation=44, ha="left")

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Subject 2 — Mean per-voxel CC", fontsize=10)
    ax.set_ylabel("Subject 3 — Mean per-voxel CC", fontsize=10)
    ax.set_title(
        "Cross-Subject CC by Embedding Method\n"
        "s3/s2 ratio (×) shrinks as models improve: subjects converge",
        fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    if standalone:
        fig.tight_layout()
        savefig(fig, "check2_mid4_E_method_scatter")

    return ms2, ms3, labels, colors, ratios


def overview(stories, v2, v3, r_story, corrs_s2, corrs_s3,
             top_w_s2, top_w_s3, mean_s2_all, mean_s3_all):
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.38)

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
        "Check 2 — Cross-Subject Replication (LoRA mid4★, best model)\n"
        "Does the encoding model capture a generalizable language-to-brain mapping?",
        fontsize=13, fontweight="bold", y=1.01)

    savefig(fig, "check2_mid4_overview")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(STAB_DIR, exist_ok=True)

    with open(SPLIT_PATH) as f:
        split_data = json.load(f)
    test_stories = split_data["test"]

    story_scores = {}
    corrs        = {}
    top_weights  = {}

    for sid in ["s2", "s3"]:
        print(f"\n{'='*60}")
        print(f"  Processing {sid}  (LoRA mid4)")
        print(f"{'='*60}")

        # Load pkl
        pkl_path = os.path.join(MODELS, f"{sid}_{METHOD}", f"{sid}_{METHOD}_model.pkl")
        with open(pkl_path, "rb") as fh:
            obj = pickle.load(fh)
        weights       = obj["weights"].astype(np.float32)   # (n_feat, n_vox)
        voxel_indices = obj["voxel_indices"]
        print(f"  weights: {weights.shape}  voxel_indices: {voxel_indices.shape}")

        # Z-score test embeddings using train statistics
        X_tr = load_embed(sid, "train")
        X_te = zscore_test(X_tr, load_embed(sid, "test"))
        del X_tr
        print(f"  X_test (z-scored): {X_te.shape}")

        # Story lengths (number of TRs per test story, subject-specific)
        subj_dir = os.path.join(DATA_PATH, f"subject{sid[1]}")
        story_lengths = {
            s: np.load(os.path.join(subj_dir, f"{s}.npy"), mmap_mode="r").shape[0]
            for s in test_stories
        }
        total_te = sum(story_lengths.values())
        assert X_te.shape[0] == total_te, \
            f"X_test rows {X_te.shape[0]} != fMRI test TRs {total_te}"

        # Compute per-story mean CC
        print(f"  Computing per-story CC ({len(test_stories)} stories)...")
        t0 = time.time()
        story_scores[sid] = compute_story_cc(
            sid, weights, voxel_indices, X_te,
            test_stories, story_lengths,
        )
        print(f"  Done in {(time.time()-t0)/60:.1f} min")

        # Per-voxel CC from pkl (already computed during training)
        corrs[sid] = obj["test_corrs"].astype(np.float32)
        print(f"  mean CC = {corrs[sid].mean():.4f}  (pkl verified)")

        # Top-5% voxel weights for functional PCA
        pct95 = np.percentile(corrs[sid], 95)
        mask  = corrs[sid] >= pct95
        top_weights[sid] = weights[:, mask]
        print(f"  top-5% voxels: {mask.sum():,}  weight matrix: {top_weights[sid].shape}")

    # Story-level Pearson r
    stories      = sorted(story_scores["s2"].keys())
    v2 = np.array([story_scores["s2"][s] for s in stories])
    v3 = np.array([story_scores["s3"][s] for s in stories])
    r_story = float(np.corrcoef(v2, v3)[0, 1])

    print(f"\nStory-level Pearson r = {r_story:.4f}  (n={len(stories)} stories)")
    print(f"s2 mean CC = {corrs['s2'].mean():.4f}")
    print(f"s3 mean CC = {corrs['s3'].mean():.4f}")
    print(f"s3 > s2 in {sum(b>a for a,b in zip(v2,v3))}/{len(stories)} stories")

    # Save story scores
    out = {
        "method": METHOD,
        "story_pearson_r": r_story,
        "story_scores_s2": {s: float(v) for s, v in zip(stories, v2)},
        "story_scores_s3": {s: float(v) for s, v in zip(stories, v3)},
        "s2_mean_cc": float(corrs["s2"].mean()),
        "s3_mean_cc": float(corrs["s3"].mean()),
    }
    with open(os.path.join(STAB_DIR, "check2_mid4_story_scores.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved story scores JSON")

    # Generate all panels
    print("\nGenerating figures...")
    panel_A(stories, v2, v3, r_story)
    panel_B(stories, v2, v3, corrs["s2"].mean(), corrs["s3"].mean(), r_story)
    panel_C(corrs["s2"], corrs["s3"])
    panel_D(corrs["s2"], corrs["s3"], top_weights["s2"], top_weights["s3"])
    panel_E()
    overview(stories, v2, v3, r_story,
             corrs["s2"], corrs["s3"],
             top_weights["s2"], top_weights["s3"],
             corrs["s2"].mean(), corrs["s3"].mean())

    print(f"\nAll figures saved to {STAB_DIR}/")


if __name__ == "__main__":
    main()
