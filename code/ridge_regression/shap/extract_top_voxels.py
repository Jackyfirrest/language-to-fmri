#!/usr/bin/env python3
"""
Extract top 5% performing voxels from ridge regression results.

This script loads the ridge regression results and extracts the top 5% of voxels
by test correlation. It creates outputs useful for SHAP analysis.

Usage:
    python extract_top_voxels.py --subject-id s2 --method finetuned_lora
    python extract_top_voxels.py --subject-id s3 --method finetuned_lora --top-percentile 5
"""

import argparse
import json
import os
import pickle
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_ridge import load_embedding, zscore_X, load_fmri_chunk


def parse_args():
    p = argparse.ArgumentParser(
        description='Extract top performing voxels from ridge regression results'
    )
    p.add_argument('--subject-id', required=True, help='Subject ID (s2, s3)')
    p.add_argument('--method', required=True, help='Embedding method used')
    p.add_argument('--result-dir', default='../../../results',
                   help='Result directory from ridge regression')
    p.add_argument('--top-percentile', type=float, default=5,
                   help='Top percentile to extract (default: 5)')
    p.add_argument('--output-dir', default='../../../results/top_voxels',
                   help='Output directory for top voxels analysis')
    p.add_argument('--subject', default=None, help='Full subject name')
    p.add_argument('--data-path',
                   default='/ocean/projects/mth250011p/shared/215a/final_project/data')
    p.add_argument('--embedding-dir', default='../../../data/embeddings')
    p.add_argument('--split-path', default='../../../data/train_test_split.json')
    p.add_argument('--filter-stories', nargs='+', default=None,
                   help='List of stories to filter by (compute correlations only on these stories)')
    return p.parse_args()


def _zscore_cols(arr):
    """Z-score each column independently."""
    arr = np.asarray(arr, dtype=np.float32)
    mu    = arr.mean(axis=0, keepdims=True)
    sigma = arr.std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    return (arr - mu) / sigma


def _col_corr(y_true, y_pred):
    """Pearson correlation between matching columns of two matrices."""
    return (_zscore_cols(y_true) * _zscore_cols(y_pred)).mean(axis=0).astype(np.float32)


def find_embedding(embedding_dir, sid, split, method):
    """Locate embedding file by subject/split/method."""
    for ext in ('npy', 'npz'):
        p = os.path.join(embedding_dir, f'{sid}_{split}_{method}_embeddings.{ext}')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f'No embedding found for {sid}/{split}/{method} in {embedding_dir}'
    )


def compute_correlations_on_stories(X_test, Y_test_all, Y_test_story_mask, 
                                     voxel_indices, weights=None):
    """Compute per-voxel correlations on a subset of test timepoints (by story mask).
    
    Args:
        X_test: test embeddings (n_test, n_features)
        Y_test_all: all test fMRI data (n_test, n_voxels_all)
        Y_test_story_mask: boolean array of which test timepoints to use
        voxel_indices: indices of voxels in Y_test_all that were fit
        weights: optional pre-computed regression weights (n_features, n_voxels)
    
    Returns:
        test_corrs: correlations on the filtered timepoints
    """
    Y_test_subset = Y_test_all[Y_test_story_mask]
    X_test_subset = X_test[Y_test_story_mask]
    
    if weights is not None:
        # Use pre-computed weights
        pred = X_test_subset @ weights
        test_corrs = _col_corr(Y_test_subset, pred)
    else:
        # Would need to refit; for now just return all-zero correlations
        # (this path is if you don't have weights)
        test_corrs = np.zeros(len(voxel_indices), dtype=np.float32)
    
    return test_corrs


def main():
    args = parse_args()
    
    # Auto-detect subject name
    if args.subject is None:
        subject_map = {'s2': 'subject2', 's3': 'subject3'}
        args.subject = subject_map.get(args.subject_id)
        if not args.subject:
            raise ValueError(f"Unknown subject-id: {args.subject_id}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    def resolve(path):
        return path if os.path.isabs(path) else \
            os.path.normpath(os.path.join(script_dir, path))
    
    result_dir = resolve(args.result_dir)
    output_dir = resolve(args.output_dir)
    embedding_dir = resolve(args.embedding_dir)
    split_path = resolve(args.split_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    metric_dir = os.path.join(result_dir, 'metrics')
    model_dir = os.path.join(result_dir, 'models', f'{args.subject_id}_{args.method}')
    
    # ── Load merged results ──────────────────────────────────────────────────
    corr_file = os.path.join(metric_dir, f'{args.subject_id}_{args.method}_test_corrs.npy')
    model_file = os.path.join(model_dir, f'{args.subject_id}_{args.method}_model.pkl')
    
    if not os.path.exists(corr_file):
        raise FileNotFoundError(f"Correlation file not found: {corr_file}\n"
                               f"Run ridge regression first!")
    
    print(f"Loading results from {corr_file}")
    test_corrs_all = np.load(corr_file)
    print(f"Loaded {len(test_corrs_all)} voxel correlations (all test data)")
    
    if os.path.exists(model_file):
        print(f"Loading model from {model_file}")
        with open(model_file, 'rb') as f:
            model_obj = pickle.load(f)
        voxel_indices = model_obj['voxel_indices']
        weights = model_obj.get('weights', None)
    else:
        print(f"Model file not found: {model_file}")
        voxel_indices = np.arange(len(test_corrs_all))
        weights = None
    
    # ── Filter by stories if requested ────────────────────────────────────────
    test_corrs = test_corrs_all
    filter_tag = ""
    
    if args.filter_stories is not None and len(args.filter_stories) > 0:
        print(f"\n=== Filtering by stories: {args.filter_stories} ===")
        
        # Load test split and embeddings
        with open(split_path) as f:
            split = json.load(f)
        test_stories = split['test']
        
        # Load test embeddings
        X_test = load_embedding(
            find_embedding(embedding_dir, args.subject_id, 'test', args.method)
        )
        X_test_zscored, _ = zscore_X(X_test, X_test)  # z-score using itself
        
        # Load all test fMRI data and create mask for requested stories
        fmri_dir = os.path.join(args.data_path, args.subject)
        test_maps = {s: np.load(os.path.join(fmri_dir, f'{s}.npy'), mmap_mode='r')
                     for s in test_stories}
        test_lengths = [test_maps[s].shape[0] for s in test_stories]
        test_total = sum(test_lengths)
        
        # Build mask for which timepoints belong to requested stories
        story_mask = np.zeros(test_total, dtype=bool)
        offset = 0
        for story, n in zip(test_stories, test_lengths):
            if story in args.filter_stories:
                story_mask[offset:offset + n] = True
            offset += n
        
        n_filtered = story_mask.sum()
        print(f"Using {n_filtered}/{test_total} test timepoints "
              f"({100*n_filtered/test_total:.1f}%) from requested stories")
        
        # Load fMRI data only for the voxels in the model and only the filtered timepoints
        X_test_filt = X_test_zscored[story_mask]
        
        # Load fMRI data in a more memory-efficient way
        Y_test_filtered_list = []
        offset = 0
        for story, n in zip(test_stories, test_lengths):
            if story in args.filter_stories:
                # Load only this story's data for the voxels we fit
                story_fmri = np.asarray(test_maps[story][:, voxel_indices], dtype=np.float32)
                np.nan_to_num(story_fmri, nan=0.0, copy=False)
                # Z-score this story
                mu = story_fmri.mean(axis=0, keepdims=True)
                sigma = story_fmri.std(axis=0, keepdims=True)
                sigma[sigma < 1e-6] = 1.0
                story_fmri = (story_fmri - mu) / sigma
                Y_test_filtered_list.append(story_fmri)
            offset += n
        
        Y_test_filt = np.concatenate(Y_test_filtered_list, axis=0)
        
        # Compute correlations only on filtered stories
        if weights is not None:
            pred = X_test_filt @ weights
            test_corrs = _col_corr(Y_test_filt, pred)
            print(f"Computed correlations on filtered data: {len(test_corrs)} voxels")
        else:
            print(f"WARNING: weights not available, cannot recompute correlations on filtered data")
            print(f"Falling back to correlations from full test data")
        
        filter_tag = f"_filtered_{'_'.join(args.filter_stories[:2])}"  # shorten tag
    
    # ── Extract top voxels ───────────────────────────────────────────────────
    threshold = np.percentile(test_corrs, 100 - args.top_percentile)
    top_mask = test_corrs >= threshold
    top_corrs = test_corrs[top_mask]
    top_voxel_indices = voxel_indices[top_mask]
    
    print(f"\n=== Top {args.top_percentile}% Voxels ===")
    print(f"Threshold (CC): {threshold:.4f}")
    print(f"Number of voxels: {len(top_corrs)}/{len(test_corrs)} "
          f"({100*len(top_corrs)/len(test_corrs):.1f}%)")
    print(f"Mean CC: {np.mean(top_corrs):.4f}")
    print(f"Median CC: {np.median(top_corrs):.4f}")
    print(f"Min CC: {np.min(top_corrs):.4f}")
    print(f"Max CC: {np.max(top_corrs):.4f}")
    
    # ── Save top voxel information ───────────────────────────────────────────
    top_voxels_data = {
        'subject_id': args.subject_id,
        'method': args.method,
        'top_percentile': args.top_percentile,
        'threshold': float(threshold),
        'voxel_indices': top_voxel_indices,
        'test_correlations': top_corrs,
        'mean_cc': float(np.mean(top_corrs)),
        'median_cc': float(np.median(top_corrs)),
        'n_voxels_total': len(test_corrs),
        'n_top_voxels': len(top_corrs),
    }
    
    if args.filter_stories is not None and len(args.filter_stories) > 0:
        top_voxels_data['filter_stories'] = args.filter_stories
        top_voxels_data['filtered_by_stories'] = True
    
    if weights is not None:
        top_weights = weights[:, top_mask]
        top_voxels_data['weights'] = top_weights
        print(f"Weights shape: {top_weights.shape}")
    
    output_file = os.path.join(output_dir, 
                              f'{args.subject_id}_{args.method}_top{int(args.top_percentile)}{filter_tag}.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(top_voxels_data, f)
    print(f"\nSaved top voxels data to: {output_file}")
    
    # ── Save as JSON for easier inspection ────────────────────────────────────
    json_data = {
        'subject_id': args.subject_id,
        'method': args.method,
        'top_percentile': args.top_percentile,
        'threshold': float(threshold),
        'voxel_indices': top_voxel_indices.tolist(),
        'test_correlations': top_corrs.tolist(),
        'mean_cc': float(np.mean(top_corrs)),
        'median_cc': float(np.median(top_corrs)),
        'n_voxels_total': int(len(test_corrs)),
        'n_top_voxels': int(len(top_corrs)),
    }
    
    if args.filter_stories is not None and len(args.filter_stories) > 0:
        json_data['filter_stories'] = args.filter_stories
        json_data['filtered_by_stories'] = True
    
    json_file = os.path.join(output_dir,
                            f'{args.subject_id}_{args.method}_top{int(args.top_percentile)}{filter_tag}.json')
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved summary to: {json_file}")
    
    # ── Visualize distribution ───────────────────────────────────────────────
    print(f"\nCorrelation distribution (all voxels):")
    percentiles = [0, 5, 25, 50, 75, 95, 99, 100]
    for pct in percentiles:
        val = np.percentile(test_corrs, pct)
        print(f"  {pct:3d}th percentile: {val:7.4f}")
    
    return top_voxels_data


if __name__ == '__main__':
    main()
