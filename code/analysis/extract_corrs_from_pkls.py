#!/usr/bin/env python3
"""
extract_corrs_from_pkls.py — rebuild per-voxel CC arrays in sequential voxel order.

The SLURM merge pipeline (merge_ridge_results.py) concatenates task corr files
in task-index order, not voxel-index order.  Different methods used different
numbers of tasks (2, 5, 10), so the merged .npy files have incompatible orderings
across methods.  Comparing them position-by-position (e.g. Spearman ρ) therefore
yields incorrect results.

This script rebuilds every {sid}_{method}_test_corrs.npy from the corresponding
model pkl using the stored voxel_indices mapping, guaranteeing that position v
in the output array corresponds to cortical voxel v for every method.

Usage:
    python code/analysis/extract_corrs_from_pkls.py          # all methods
    python code/analysis/extract_corrs_from_pkls.py --dry-run  # show plan only

Output:
    results/metrics/{sid}_{method}_test_corrs.npy  (overwritten in place)
"""

import argparse
import os
import pickle

import numpy as np

# ── config ────────────────────────────────────────────────────────────────────

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODELS_DIR  = os.path.join(REPO, "results", "models")
METRICS_DIR = os.path.join(REPO, "results", "metrics")

SUBJECTS = {
    "s2": 94_251,
    "s3": 95_556,
}

METHODS = [
    "bow",
    "word2vec",
    "glove",
    "vanilla_bert",
    "pretrained_bert",
    "finetuned_lora",
    "finetuned_lora_improved",
    "finetuned_lora_mid4_mean",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_pkl(sid: str, method: str) -> dict:
    path = os.path.join(MODELS_DIR, f"{sid}_{method}", f"{sid}_{method}_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def sequential_corrs(obj: dict, n_voxels: int) -> np.ndarray:
    """Scatter test_corrs from task-chunk order into sequential voxel order."""
    vi = obj["voxel_indices"]   # (N,) — voxel indices in chunk-task order
    tc = obj["test_corrs"]      # (N,) — CC values in the same order

    if len(vi) != n_voxels:
        raise ValueError(
            f"voxel_indices length {len(vi)} != expected {n_voxels}; "
            "pkl may be from a partial or corrupted merge."
        )
    if len(np.unique(vi)) != n_voxels:
        raise ValueError("voxel_indices contains duplicates or gaps.")

    out = np.empty(n_voxels, dtype=np.float32)
    out[vi] = tc
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=METHODS,
                    help="Methods to process (default: all)")
    ap.add_argument("--subjects", nargs="+", default=list(SUBJECTS),
                    help="Subject IDs to process (default: s2 s3)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be written without writing anything")
    args = ap.parse_args()

    os.makedirs(METRICS_DIR, exist_ok=True)

    n_ok, n_skip, n_err = 0, 0, 0

    for method in args.methods:
        for sid in args.subjects:
            n_voxels = SUBJECTS[sid]
            out_path = os.path.join(METRICS_DIR, f"{sid}_{method}_test_corrs.npy")

            try:
                obj = load_pkl(sid, method)
            except FileNotFoundError as e:
                print(f"  SKIP  {sid}/{method}: pkl not found ({e})")
                n_skip += 1
                continue

            try:
                corrs_seq = sequential_corrs(obj, n_voxels)
            except ValueError as e:
                print(f"  ERROR {sid}/{method}: {e}")
                n_err += 1
                continue

            # sanity: mean must be preserved
            mean_orig = float(np.mean(obj["test_corrs"]))
            mean_seq  = float(corrs_seq.mean())
            assert abs(mean_seq - mean_orig) < 1e-5, \
                f"mean changed after reorder: {mean_orig:.6f} -> {mean_seq:.6f}"

            if args.dry_run:
                print(f"  DRY   {sid}/{method}: would write {out_path}  "
                      f"mean={mean_seq:.4f}")
            else:
                np.save(out_path, corrs_seq)
                print(f"  OK    {sid}/{method}: {out_path}  mean={mean_seq:.4f}")
            n_ok += 1

    print(f"\nDone: {n_ok} written, {n_skip} skipped, {n_err} errors")


if __name__ == "__main__":
    main()
