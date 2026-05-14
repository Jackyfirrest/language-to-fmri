#!/usr/bin/env python3
"""
Fast SHAP-style influence analysis for ridge regression.

For linear models y = X @ w, SHAP values under an independent-feature background
can be computed in closed form without KernelExplainer:
    phi_j(x) = (x_j - E[x_j]) * w_j

This script uses that closed-form relation to produce feature influence scores
quickly and saves numeric outputs for downstream visualizations.
"""

import argparse
import csv
import json
import os
import pickle
import sys

import numpy as np
from scipy import sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from run_ridge import load_embedding, zscore_X


def parse_args():
    p = argparse.ArgumentParser(
        description="Fast SHAP-style analysis for ridge regression top voxels"
    )
    p.add_argument("--subject-id", required=True, help="Subject ID (s2, s3)")
    p.add_argument("--subject", default=None, help="Full subject name")
    p.add_argument("--method", required=True, help="Embedding method")
    p.add_argument("--result-dir", default="../../../results")
    p.add_argument("--embedding-dir", default="../../../data/embeddings")
    p.add_argument("--split-path", default="../../../data/train_test_split.json")
    p.add_argument("--top-percentile", type=float, default=5)
    p.add_argument("--n-top-voxels", type=int, default=12,
                   help="How many top voxels to aggregate (smaller = faster)")
    p.add_argument("--n-background", type=int, default=64,
                   help="Background sample count to estimate E[X]")
    p.add_argument("--max-test-samples-per-story", type=int, default=400,
                   help="Cap test samples used per selected story")
    p.add_argument("--top-k-features", type=int, default=100,
                   help="How many top influential dimensions to save")
    p.add_argument("--output-dir", default="../../../results/shap_analysis")
    p.add_argument("--filter-stories", nargs="+", default=None,
                   help="Story names used in filtered top-voxel extraction")
    return p.parse_args()


def _resolve(script_dir, path):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(script_dir, path))


def _find_embedding(embedding_dir, sid, split, method):
    for ext in ("npy", "npz"):
        p = os.path.join(embedding_dir, f"{sid}_{split}_{method}_embeddings.{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No embedding found for {sid}/{split}/{method}")


def _story_sample_indices(split_path, filter_stories, max_per_story):
    with open(split_path) as f:
        split = json.load(f)
    test_stories = split["test"]

    if not filter_stories:
        return None, test_stories

    selected = set(filter_stories)
    missing = [s for s in filter_stories if s not in test_stories]
    if missing:
        raise ValueError(f"Stories not in test split: {missing}")

    # Story lengths come from embeddings in test order; caller will pass them in.
    return selected, test_stories


def _build_eval_indices_from_lengths(test_lengths, test_stories, selected_stories, max_per_story):
    if selected_stories is None:
        return np.arange(sum(test_lengths), dtype=np.int64)

    rng = np.random.default_rng(0)
    indices = []
    offset = 0
    for story, n_rows in zip(test_stories, test_lengths):
        row_idx = np.arange(offset, offset + n_rows, dtype=np.int64)
        if story in selected_stories:
            if max_per_story is not None and max_per_story > 0 and n_rows > max_per_story:
                pick = np.sort(rng.choice(row_idx, size=max_per_story, replace=False))
            else:
                pick = row_idx
            indices.append(pick)
        offset += n_rows

    if not indices:
        raise ValueError("No test rows selected; check --filter-stories values")

    return np.concatenate(indices)


def main():
    args = parse_args()

    if args.subject is None:
        subject_map = {"s2": "subject2", "s3": "subject3"}
        args.subject = subject_map.get(args.subject_id)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = _resolve(script_dir, args.result_dir)
    embedding_dir = _resolve(script_dir, args.embedding_dir)
    split_path = _resolve(script_dir, args.split_path)
    output_dir = _resolve(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    filter_tag = ""
    if args.filter_stories:
        filter_tag = "_filtered_" + "_".join(args.filter_stories[:2])

    top_voxels_file = os.path.join(
        result_dir,
        "top_voxels",
        f"{args.subject_id}_{args.method}_top{int(args.top_percentile)}{filter_tag}.pkl",
    )
    if not os.path.exists(top_voxels_file):
        raise FileNotFoundError(f"Top voxels file not found: {top_voxels_file}")

    model_file = os.path.join(
        result_dir,
        "models",
        f"{args.subject_id}_{args.method}",
        f"{args.subject_id}_{args.method}_model.pkl",
    )
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")

    print("=" * 80)
    print(f"Fast influence analysis for {args.subject_id} / {args.method}")
    print("=" * 80)
    print(f"Top voxels input: {top_voxels_file}")

    with open(top_voxels_file, "rb") as f:
        top_voxels = pickle.load(f)
    top_voxel_indices = np.asarray(top_voxels["voxel_indices"])
    top_corrs = np.asarray(top_voxels["test_correlations"], dtype=np.float32)

    n_vox = min(args.n_top_voxels, len(top_voxel_indices))
    sort_idx = np.argsort(-top_corrs)[:n_vox]
    top_voxel_indices = top_voxel_indices[sort_idx]
    top_corrs = top_corrs[sort_idx]
    print(f"Analyzing {len(top_voxel_indices)} top voxels")

    with open(model_file, "rb") as f:
        model_obj = pickle.load(f)
    weights = model_obj.get("weights")
    if weights is None:
        raise RuntimeError("Model does not contain weights. Re-run ridge with --save-weights.")
    model_voxel_indices = np.asarray(model_obj["voxel_indices"])

    print("\nLoading embeddings...")
    X_train = load_embedding(_find_embedding(embedding_dir, args.subject_id, "train", args.method))
    X_test = load_embedding(_find_embedding(embedding_dir, args.subject_id, "test", args.method))
    X_train, X_test = zscore_X(X_train, X_test)

    if sp.issparse(X_train):
        n_background = min(args.n_background, X_train.shape[0])
        rng = np.random.default_rng(0)
        bg_idx = rng.choice(X_train.shape[0], n_background, replace=False)
        X_bg = X_train[bg_idx].toarray().astype(np.float32)
    else:
        X_train = np.asarray(X_train, dtype=np.float32)
        n_background = min(args.n_background, X_train.shape[0])
        rng = np.random.default_rng(0)
        bg_idx = rng.choice(X_train.shape[0], n_background, replace=False)
        X_bg = X_train[bg_idx].astype(np.float32)

    if sp.issparse(X_test):
        X_test_arr = X_test.toarray().astype(np.float32)
    else:
        X_test_arr = np.asarray(X_test, dtype=np.float32)

    selected_stories, test_stories = _story_sample_indices(
        split_path,
        args.filter_stories,
        args.max_test_samples_per_story,
    )

    # Derive story lengths from test fMRI files to preserve split row order.
    fmri_dir = os.path.join("/ocean/projects/mth250011p/shared/215a/final_project/data", args.subject)
    test_lengths = [np.load(os.path.join(fmri_dir, f"{s}.npy"), mmap_mode="r").shape[0] for s in test_stories]
    eval_idx = _build_eval_indices_from_lengths(
        test_lengths,
        test_stories,
        selected_stories,
        args.max_test_samples_per_story,
    )

    X_eval = X_test_arr[eval_idx]
    x_bg_mean = X_bg.mean(axis=0)

    print(f"Background shape: {X_bg.shape}")
    print(f"Eval samples used: {X_eval.shape[0]} / {X_test_arr.shape[0]}")

    feature_accum = np.zeros(X_eval.shape[1], dtype=np.float64)
    voxel_rows = []
    voxel_analysis = {}

    for i, (vox_id, cc) in enumerate(zip(top_voxel_indices, top_corrs), 1):
        pos = np.where(model_voxel_indices == vox_id)[0]
        if len(pos) == 0:
            print(f"[WARN] voxel {vox_id} not in model index; skipping")
            continue

        w = np.asarray(weights[:, pos[0]], dtype=np.float32)

        # Closed-form linear SHAP values for each eval sample and feature.
        shap_vals = (X_eval - x_bg_mean) * w[None, :]
        abs_mean = np.abs(shap_vals).mean(axis=0)
        signed_mean = shap_vals.mean(axis=0)

        feature_accum += abs_mean
        top_dims = np.argsort(-abs_mean)[: min(20, len(abs_mean))]

        print(f"[{i}] voxel={int(vox_id)} cc={float(cc):.4f} top_dim={int(top_dims[0])} score={float(abs_mean[top_dims[0]]):.6f}")

        voxel_analysis[f"voxel_{int(vox_id)}"] = {
            "voxel_id": int(vox_id),
            "test_correlation": float(cc),
            "top_20_dims": top_dims.tolist(),
            "top_20_abs_mean_shap": [float(abs_mean[d]) for d in top_dims],
            "top_20_signed_mean_shap": [float(signed_mean[d]) for d in top_dims],
        }

        for rank, d in enumerate(top_dims, 1):
            voxel_rows.append(
                {
                    "voxel_id": int(vox_id),
                    "test_correlation": float(cc),
                    "feature_dim": int(d),
                    "rank_within_voxel": rank,
                    "abs_mean_shap": float(abs_mean[d]),
                    "signed_mean_shap": float(signed_mean[d]),
                    "weight": float(w[d]),
                }
            )

    if np.sum(feature_accum) <= 0:
        raise RuntimeError("No influence scores computed; check selected voxels/samples")

    influence = feature_accum / (np.sum(feature_accum) + 1e-12)
    top_k = min(args.top_k_features, influence.shape[0])
    top_dims = np.argsort(-influence)[:top_k]

    dim_rows = []
    cumulative = 0.0
    for rank, d in enumerate(top_dims, 1):
        score = float(influence[d])
        cumulative += score
        dim_rows.append(
            {
                "rank": rank,
                "feature_dim": int(d),
                "influence_score": score,
                "cumulative_influence": cumulative,
            }
        )

    base_name = f"{args.subject_id}_{args.method}_shap{filter_tag}"

    pkl_path = os.path.join(output_dir, f"{base_name}.pkl")
    json_path = os.path.join(output_dir, f"{base_name}_summary.json")
    dim_csv_path = os.path.join(output_dir, f"{base_name}_dimension_influence.csv")
    voxel_csv_path = os.path.join(output_dir, f"{base_name}_voxel_influence.csv")

    out_obj = {
        "subject_id": args.subject_id,
        "method": args.method,
        "top_percentile": args.top_percentile,
        "filter_stories": args.filter_stories,
        "n_voxels_analyzed": int(len(top_voxel_indices)),
        "n_eval_samples": int(X_eval.shape[0]),
        "n_background": int(X_bg.shape[0]),
        "top_feature_dims": [int(d) for d in top_dims],
        "top_feature_scores": [float(influence[d]) for d in top_dims],
        "voxel_analysis": voxel_analysis,
    }

    with open(pkl_path, "wb") as f:
        pickle.dump(out_obj, f)

    summary = {
        "subject_id": args.subject_id,
        "method": args.method,
        "filter_stories": args.filter_stories,
        "n_voxels_analyzed": int(len(top_voxel_indices)),
        "n_eval_samples": int(X_eval.shape[0]),
        "n_background": int(X_bg.shape[0]),
        "top_100_dims": [int(d) for d in top_dims],
        "top_100_influence_scores": [float(influence[d]) for d in top_dims],
        "files": {
            "pickle": pkl_path,
            "dimension_csv": dim_csv_path,
            "voxel_csv": voxel_csv_path,
        },
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    with open(dim_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "feature_dim", "influence_score", "cumulative_influence"],
        )
        writer.writeheader()
        writer.writerows(dim_rows)

    with open(voxel_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "voxel_id",
                "test_correlation",
                "feature_dim",
                "rank_within_voxel",
                "abs_mean_shap",
                "signed_mean_shap",
                "weight",
            ],
        )
        writer.writeheader()
        writer.writerows(voxel_rows)

    print("\nSaved outputs:")
    print(f"  {pkl_path}")
    print(f"  {json_path}")
    print(f"  {dim_csv_path}")
    print(f"  {voxel_csv_path}")


if __name__ == "__main__":
    main()
