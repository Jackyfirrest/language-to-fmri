#!/usr/bin/env python3
"""
Merge partial ridge results from a SLURM array job into a single file.

Usage:
    python merge_ridge_results.py --subject-id s2 --method bow
    python merge_ridge_results.py --subject-id s3 --method glove --n-tasks 10
"""

import argparse
import glob
import os
import pickle
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--subject-id', required=True)
    p.add_argument('--method', required=True)
    p.add_argument('--result-dir', default='../../results')
    p.add_argument('--n-tasks', type=int, default=None,
                   help='Expected number of tasks (for completeness check)')
    return p.parse_args()


def main():
    args = parse_args()
    sid, method = args.subject_id, args.method

    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = args.result_dir if os.path.isabs(args.result_dir) else \
        os.path.normpath(os.path.join(script_dir, args.result_dir))

    metric_dir = os.path.join(result_dir, 'metrics')
    model_dir  = os.path.join(result_dir, 'models', f'{sid}_{method}')

    # ── merge correlation files ───────────────────────────────────────────────
    corr_files = sorted(glob.glob(os.path.join(metric_dir, f'{sid}_{method}_test_corrs_task*.npy')))
    if not corr_files:
        print('No partial corr files found. Nothing to merge.')
        return

    if args.n_tasks and len(corr_files) != args.n_tasks:
        print(f'WARNING: expected {args.n_tasks} task files, found {len(corr_files)}')

    corrs = np.concatenate([np.load(f) for f in corr_files])
    merged_corr_path = os.path.join(metric_dir, f'{sid}_{method}_test_corrs.npy')
    np.save(merged_corr_path, corrs)
    print(f'Merged {len(corr_files)} corr files -> {merged_corr_path}  ({len(corrs)} voxels)')

    # ── merge model files ─────────────────────────────────────────────────────
    model_files = sorted(glob.glob(os.path.join(model_dir, f'{sid}_{method}_model_task*.pkl')))
    if model_files:
        shards = [pickle.load(open(f, 'rb')) for f in model_files]
        merged = {k: shards[0][k] for k in shards[0] if k not in
                  ('voxel_indices', 'test_corrs', 'weights', 'mean_test_cc',
                   'median_test_cc', 'top1pct_test_cc', 'top5pct_test_cc')}
        merged['voxel_indices'] = np.concatenate([s['voxel_indices'] for s in shards])
        merged['test_corrs']    = corrs
        merged['mean_test_cc']  = float(np.mean(corrs))
        merged['median_test_cc']= float(np.median(corrs))
        merged['top1pct_test_cc'] = float(np.percentile(corrs, 99))
        merged['top5pct_test_cc'] = float(np.percentile(corrs, 95))
        if 'weights' in shards[0]:
            merged['weights'] = np.concatenate([s['weights'] for s in shards], axis=1)

        merged_model_path = os.path.join(model_dir, f'{sid}_{method}_model.pkl')
        with open(merged_model_path, 'wb') as f:
            pickle.dump(merged, f)
        print(f'Merged {len(model_files)} model shards -> {merged_model_path}')

    print(f'\n=== Merged results: {sid} {method} ===')
    print(f'  Mean CC   : {np.mean(corrs):.4f}')
    print(f'  Median CC : {np.median(corrs):.4f}')
    print(f'  Top 1% CC : {np.percentile(corrs, 99):.4f}')
    print(f'  Top 5% CC : {np.percentile(corrs, 95):.4f}')
    print(f'  Voxels    : {len(corrs)}')


if __name__ == '__main__':
    main()
