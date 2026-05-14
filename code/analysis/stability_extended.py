"""
stability_extended.py — Three new stability checks complementing stability_check.py.

CHECK 4 — Split-Half Voxel Map Reliability
  Using saved ridge weights (no retraining), split the 20 test stories into two
  random halves of 10 stories each (100 splits).  For each split, compute the
  per-voxel mean CC for each half independently, then measure Spearman ρ between
  the two half-map CC vectors.  Tests: is the language-responsive voxel map
  stable regardless of which test stories happen to be used?
  A Spearman-Brown-corrected estimate of the full 20-story reliability is also
  reported.  Methods: GloVe, Enhanced BERT, LoRA-last4.

CHECK 5 — Cross-Subject Top-k Jaccard Stability Curve
  For all 8 embedding methods, compute the Jaccard overlap between s2 and s3's
  top-k% voxels as k varies from 0.05% to 30% (log-spaced).  Normalised by the
  theoretical chance rate k²/(2k−k²).  Tests: do better models identify the
  same language regions more consistently across subjects?

CHECK 6 — Multi-Method Voxel Consensus Score
  For each voxel in s2 and s3, count how many of the 8 embedding methods rank it
  above their own subject-specific median CC.  Under independence (each method
  equally likely to call any voxel above-median), the count follows Binomial(8, 0.5).
  The actual distribution should be skewed toward 0 and 8 if a shared set of
  language-responsive voxels is consistently identified.

Run from repo root:
    python code/analysis/stability_extended.py

Outputs saved to results/stability/:
    check4_splithalf.{pdf,png}
    check5_jaccard_curve.{pdf,png}
    check6_consensus.{pdf,png}
    stability_extended_summary.json
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, binom as binom_dist

# ── paths ────────────────────────────────────────────────────────────────────
REPO       = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
METRICS    = os.path.join(REPO, "results/metrics")
MODELS     = os.path.join(REPO, "results/models")
EMBED_DIR  = os.path.join(REPO, "data/embeddings")
SPLIT_PATH = os.path.join(REPO, "data/train_test_split.json")
OUT_DIR    = os.path.join(REPO, "results/stability")
DATA_PATH  = "/scratch/users/s214/lab3"
os.makedirs(OUT_DIR, exist_ok=True)

# ── method registry (same as regression_stats.py) ────────────────────────────
METHOD_META = {
    "bow":                      ("BoW",             "#B3B3B3"),
    "word2vec":                 ("Word2Vec",         "#999999"),
    "glove":                    ("GloVe",           "#8DA0CB"),
    "vanilla_bert":             ("Vanilla BERT",     "#66C2A5"),
    "pretrained_bert":          ("Enhanced BERT",    "#FC8D62"),
    "finetuned_lora":           ("LoRA (last4)",     "#E78AC3"),
    "finetuned_lora_improved":  ("LoRA (improved)",  "#A6D854"),
    "finetuned_lora_mid4_mean": ("LoRA (mid4)",      "#FFD92F"),
}
ORDERED_METHODS = list(METHOD_META.keys())
SUBJECTS     = ["s2", "s3"]
SUBJ_LABELS  = {"s2": "Subject 2", "s3": "Subject 3"}
SUBJ_FMRI    = {"s2": "subject2",  "s3": "subject3"}
N_VOXELS     = {"s2": 94_251, "s3": 95_556}
METHODS_C4   = ["glove", "pretrained_bert", "finetuned_lora"]  # have saved embeddings
N_SPLITS     = 100
CHUNK_V      = 2_000    # voxel chunk size for memory-safe matrix multiply
RNG          = np.random.default_rng(42)

# ── helpers ───────────────────────────────────────────────────────────────────

def load_corr(sid: str, method: str) -> np.ndarray | None:
    p = os.path.join(METRICS, f"{sid}_{method}_test_corrs.npy")
    return np.load(p) if os.path.exists(p) else None


def load_embedding(sid: str, split: str, method: str) -> np.ndarray:
    for ext in ("npy", "npz"):
        p = os.path.join(EMBED_DIR, f"{sid}_{split}_{method}_embeddings.{ext}")
        if os.path.exists(p):
            if ext == "npz":
                return np.load(p)["X"].astype(np.float32)
            return np.load(p).astype(np.float32)
    raise FileNotFoundError(f"Embedding not found: {sid}/{split}/{method}")


def zscore_test(X_train: np.ndarray, X_test: np.ndarray):
    """Z-score X_test using train mean/std (same as run_ridge.py)."""
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-8
    return (X_test - mu) / sd


def col_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson r for each column pair. A, B: (T, V)."""
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    num   = (A * B).sum(axis=0)
    denom = np.sqrt((A * A).sum(axis=0) * (B * B).sum(axis=0)) + 1e-10
    return num / denom


def savefig(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"),
                    bbox_inches="tight", dpi=150)
    print(f"  saved  {name}.pdf/png")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 4 — Split-half voxel map reliability
# ══════════════════════════════════════════════════════════════════════════════

def check4_story_cc_matrix(sid: str, method: str, test_stories: list,
                            story_lengths: dict, X_te: np.ndarray,
                            weights: np.ndarray, voxel_indices: np.ndarray
                            ) -> np.ndarray:
    """
    Return (n_stories, n_voxels) matrix of per-story per-voxel mean CC.

    Weights are stored in 'chunk order' (column i of weights corresponds to
    brain voxel voxel_indices[i]).  The fMRI files are in spatial order
    (column j = voxel j).  We handle this by reading Y[:, voxel_indices[chunk]]
    for each voxel chunk.
    """
    fmri_subject = SUBJ_FMRI[sid]
    n_voxels     = weights.shape[1]
    n_stories    = len(test_stories)
    story_mat    = np.zeros((n_stories, n_voxels), dtype=np.float32)

    offset = 0
    for si, story in enumerate(test_stories):
        n_t = story_lengths[story]
        X_s = X_te[offset : offset + n_t]          # (n_t, n_feat)
        Y_s = np.nan_to_num(
            np.load(os.path.join(DATA_PATH, fmri_subject, f"{story}.npy")),
            nan=0.0
        ).astype(np.float32)                        # (n_t, n_brain_voxels)

        for vstart in range(0, n_voxels, CHUNK_V):
            vend    = min(vstart + CHUNK_V, n_voxels)
            W_chunk = weights[:, vstart:vend]       # (n_feat, chunk_v)
            P_chunk = X_s @ W_chunk                 # (n_t, chunk_v)
            # voxel_indices maps chunk position → spatial brain position
            vi      = voxel_indices[vstart:vend]
            Y_chunk = Y_s[:, vi]                    # (n_t, chunk_v)
            story_mat[si, vstart:vend] = col_corr(P_chunk, Y_chunk)

        offset += n_t
        print(f"    {si+1:2d}/{n_stories}  {story:<42s}  "
              f"mean_cc={story_mat[si].mean():.4f}", end="\r")

    print()
    return story_mat


def run_check4():
    print("\n" + "=" * 70)
    print("  CHECK 4: Split-half voxel map reliability")
    print("=" * 70)

    with open(SPLIT_PATH) as f:
        split_data = json.load(f)
    test_stories = split_data["test"]
    n_test = len(test_stories)

    all_results = {}

    for method in METHODS_C4:
        all_results[method] = {}
        print(f"\n  Method: {METHOD_META[method][0]}")

        for sid in SUBJECTS:
            print(f"    Subject: {sid}")

            pkl_path = os.path.join(MODELS, f"{sid}_{method}",
                                    f"{sid}_{method}_model.pkl")
            if not os.path.exists(pkl_path):
                print(f"    SKIP: pkl not found at {pkl_path}")
                continue

            with open(pkl_path, "rb") as fh:
                obj = pickle.load(fh)
            weights       = obj["weights"].astype(np.float32)  # (n_feat, n_v)
            voxel_indices = obj["voxel_indices"]

            # z-score test embeddings using train statistics (match ridge pipeline)
            X_tr = load_embedding(sid, "train", method)
            X_te = zscore_test(X_tr, load_embedding(sid, "test", method))
            del X_tr

            # Get per-subject story lengths (subjects differ)
            story_lengths = {
                s: np.load(
                    os.path.join(DATA_PATH, SUBJ_FMRI[sid], f"{s}.npy"),
                    mmap_mode="r"
                ).shape[0]
                for s in test_stories
            }

            print(f"    Computing per-story CC matrix "
                  f"({n_test} stories × {weights.shape[1]:,} voxels)…")
            story_mat = check4_story_cc_matrix(
                sid, method, test_stories, story_lengths,
                X_te, weights, voxel_indices
            )

            # Sanity: recomputed mean CC should match saved pkl value
            recomputed_mean = float(story_mat.mean())
            saved_mean      = float(obj["mean_test_cc"])
            print(f"    Sanity: recomputed mean CC = {recomputed_mean:.4f} "
                  f"(pkl says {saved_mean:.4f})")

            # Bootstrap split-half
            rho_list = []
            for _ in range(N_SPLITS):
                perm = RNG.permutation(n_test)
                h1, h2 = perm[: n_test // 2], perm[n_test // 2 :]
                cc1 = story_mat[h1].mean(axis=0)
                cc2 = story_mat[h2].mean(axis=0)
                rho, _ = spearmanr(cc1, cc2)
                rho_list.append(float(rho))

            rho_arr = np.array(rho_list)
            rho_mean = float(rho_arr.mean())
            rho_sb   = float(2 * rho_mean / (1 + rho_mean))   # Spearman-Brown
            print(f"    Split-half Spearman ρ: mean={rho_mean:.4f}  "
                  f"SD={rho_arr.std():.4f}  SB-corrected={rho_sb:.4f}")

            all_results[method][sid] = {
                "rho_mean": rho_mean,
                "rho_std":  float(rho_arr.std()),
                "rho_sb":   rho_sb,
                "rho_list": rho_list,
            }

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    x = np.arange(len(METHODS_C4))

    for ax, sid in zip(axes, SUBJECTS):
        rho_means = []
        rho_stds  = []
        labels    = []
        colors    = []
        for method in METHODS_C4:
            res = all_results.get(method, {}).get(sid)
            if res is None:
                rho_means.append(np.nan)
                rho_stds.append(np.nan)
            else:
                rho_means.append(res["rho_mean"])
                rho_stds.append(res["rho_std"])
            labels.append(METHOD_META[method][0])
            colors.append(METHOD_META[method][1])

        bars = ax.bar(x, rho_means, color=colors, edgecolor="grey",
                      alpha=0.85, width=0.55)
        ax.errorbar(x, rho_means, yerr=rho_stds, fmt="none",
                    color="black", capsize=4, linewidth=1.2)

        for xi, (rm, method) in enumerate(zip(rho_means, METHODS_C4)):
            res = all_results.get(method, {}).get(sid)
            if res is None:
                continue
            # Annotate SB-corrected value above the bar
            ax.text(xi, rm + rho_stds[xi] + 0.01,
                    f"SB={res['rho_sb']:.3f}",
                    ha="center", va="bottom", fontsize=8, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Mean split-half Spearman ρ", fontsize=9)
        ax.set_title(f"{SUBJ_LABELS[sid]}", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.8, color="grey", lw=0.7, ls=":", label="ρ = 0.80 ref")
        ax.legend(fontsize=7)

    fig.suptitle(
        "Check 4 — Split-Half Voxel Map Reliability\n"
        "(100 random 10+10 test-story splits; bar = mean ρ ± SD; SB = Spearman-Brown corrected)",
        fontsize=10, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "check4_splithalf")
    plt.close(fig)

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 5 — Top-k voxel selectivity curve: each method vs. best model
# ══════════════════════════════════════════════════════════════════════════════

def run_check5():
    """
    Within each subject, compare the top-k% voxels identified by each embedding
    method against those identified by the best model (LoRA mid4_mean) as k
    varies from 0.05% to 30% (log-spaced, ~80 values).

    Insight: at what selectivity do simpler models diverge from the best model?
    If GloVe and mid4_mean converge at large k but diverge at small k, it means
    the most language-responsive voxels require richer representations to be
    identified, while the broader language network is consistent across methods.

    Normalised Jaccard = Jaccard(method, mid4) / Jaccard(mid4, mid4) = raw Jaccard / k_chance,
    where k_chance = k / (2 - k)  is the Jaccard for two independent random top-k sets.
    """
    print("\n" + "=" * 70)
    print("  CHECK 5: Top-k voxel selectivity stability vs. best model")
    print("=" * 70)

    REF_METHOD = "finetuned_lora_mid4_mean"
    k_vals     = np.logspace(np.log10(0.05), np.log10(30), 80)  # 0.05% … 30%

    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, sid in zip(axes, SUBJECTS):
        ref_cc = load_corr(sid, REF_METHOD)
        if ref_cc is None:
            print(f"  SKIP {sid}: reference model not found")
            continue

        n_v = len(ref_cc)
        results[sid] = {}

        for method in ORDERED_METHODS:
            if method == REF_METHOD:
                continue
            cc = load_corr(sid, method)
            if cc is None:
                continue

            jacc_norm = []
            for k in k_vals:
                fk = k / 100.0
                th_ref = np.percentile(ref_cc, 100 - k)
                th_m   = np.percentile(cc,     100 - k)
                mask_ref = ref_cc >= th_ref
                mask_m   = cc     >= th_m

                intersection = int(np.sum(mask_ref & mask_m))
                union        = int(np.sum(mask_ref | mask_m))
                j = intersection / max(union, 1)
                # Normalise by chance Jaccard for two independent top-k% sets
                chance = fk / (2.0 - fk) if fk < 1 else 1.0
                jacc_norm.append(j / chance if chance > 0 else 0.0)

            label, color = METHOD_META[method]
            ax.plot(k_vals, jacc_norm, color=color, lw=1.8, label=label)
            results[sid][method] = {
                "k_vals":     k_vals.tolist(),
                "jacc_norm":  jacc_norm,
            }
            # print at k=1%
            idx1 = int(np.searchsorted(k_vals, 1.0))
            print(f"  {sid} {label:<24s}  "
                  f"norm-Jaccard@1%={jacc_norm[idx1]:.2f}×  "
                  f"@5%={jacc_norm[int(np.searchsorted(k_vals, 5.0))]:.2f}×")

        ax.axhline(1.0, color="grey", lw=0.8, ls=":", label="Chance (= 1×)")
        ax.set_xscale("log")
        ax.set_xlabel("Selectivity threshold k (%)", fontsize=10)
        ax.set_ylabel("Jaccard / chance rate", fontsize=10)
        ax.set_title(f"{SUBJ_LABELS[sid]} — vs. LoRA (mid4)", fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "Check 5 — Top-k Voxel Overlap with Best Model (LoRA mid4) as a Function of Selectivity\n"
        "(Chance-normalised Jaccard; higher = method selects the same voxels as the best model)",
        fontsize=10, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "check5_jaccard_curve")
    plt.close(fig)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 6 — Multi-method voxel consensus score
# ══════════════════════════════════════════════════════════════════════════════

def run_check6():
    print("\n" + "=" * 70)
    print("  CHECK 6: Multi-method voxel consensus score")
    print("=" * 70)

    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    for ax, sid in zip(axes, SUBJECTS):
        avail_methods = [m for m in ORDERED_METHODS if load_corr(sid, m) is not None]
        n_m = len(avail_methods)
        n_v = N_VOXELS[sid]

        # Build binary indicator matrix: (n_methods, n_voxels)
        # indicator[i, v] = 1 if voxel v is above median CC for method i
        indicator = np.zeros((n_m, n_v), dtype=np.int8)
        for i, method in enumerate(avail_methods):
            cc = load_corr(sid, method)
            med = np.median(cc)
            indicator[i, :] = (cc >= med).astype(np.int8)

        consensus = indicator.sum(axis=0)   # (n_v,) values in [0, n_m]
        counts = np.bincount(consensus, minlength=n_m + 1)

        # Binomial(n_m, 0.5) null
        null_counts = binom_dist.pmf(np.arange(n_m + 1), n_m, 0.5) * n_v

        scores = np.arange(n_m + 1)
        width  = 0.38

        bars1 = ax.bar(scores - width / 2, counts,   width=width, color="#5B8DB8",
                       alpha=0.85, label="Observed", edgecolor="grey")
        bars2 = ax.bar(scores + width / 2, null_counts, width=width, color="#E0763E",
                       alpha=0.70, label=f"Binomial({n_m}, 0.5) null", edgecolor="grey")

        ax.set_xticks(scores)
        ax.set_xticklabels([str(s) for s in scores])
        ax.set_xlabel(f"Number of methods (out of {n_m})\nranking voxel above median", fontsize=9)
        ax.set_ylabel("Number of voxels", fontsize=9)
        ax.set_title(f"{SUBJ_LABELS[sid]}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)

        # Annotate extreme bins
        pct_all  = 100.0 * counts[-1] / n_v
        pct_none = 100.0 * counts[0]  / n_v
        ax.text(n_m, counts[-1] + n_v * 0.003,
                f"{pct_all:.1f}%\n(all {n_m})", ha="center", va="bottom", fontsize=8)
        ax.text(0, counts[0] + n_v * 0.003,
                f"{pct_none:.1f}%\n(none)", ha="center", va="bottom", fontsize=8)

        # Excess fraction in extreme bins relative to null
        excess_top  = float((counts[-1] - null_counts[-1]) / n_v * 100)
        excess_bot  = float((counts[0]  - null_counts[0])  / n_v * 100)
        print(f"  {sid}: {n_m} methods, {n_v:,} voxels")
        print(f"    top-score ({n_m}/{n_m}): {counts[-1]:,}  ({pct_all:.1f}%)  "
              f"excess vs null: {excess_top:+.1f}%")
        print(f"    bot-score (0/{n_m}): {counts[0]:,}  ({pct_none:.1f}%)  "
              f"excess vs null: {excess_bot:+.1f}%")

        results[sid] = {
            "n_methods":    n_m,
            "methods_used": avail_methods,
            "consensus_counts": counts.tolist(),
            "null_expected":    null_counts.tolist(),
            "pct_all_methods":  pct_all,
            "pct_no_methods":   pct_none,
            "excess_top_pct":   excess_top,
            "excess_bot_pct":   excess_bot,
        }

    fig.suptitle(
        "Check 6 — Multi-Method Voxel Consensus Score\n"
        "(How many of 8 embedding methods rank each voxel above their median CC?)",
        fontsize=10, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "check6_consensus")
    plt.close(fig)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-check4", action="store_true",
                    help="Skip Check 4 (requires scratch + saved pkl weights)")
    args = ap.parse_args()

    summary = {}

    if not args.skip_check4:
        summary["check4"] = run_check4()
    else:
        print("Skipping Check 4 (--skip-check4 passed)")

    summary["check5"] = run_check5()
    summary["check6"] = run_check6()

    out_path = os.path.join(OUT_DIR, "stability_extended_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {out_path}")
    print(f"All figures saved to → {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
