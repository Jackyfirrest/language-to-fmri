#!/usr/bin/env python3
"""
Master script to run the complete finetuned_lora analysis pipeline.

This script:
1. Runs ridge regression with finetuned_lora embeddings
2. Extracts top 5% performing voxels
3. Performs SHAP analysis to identify influential features

Usage:
    python run_complete_pipeline.py --subject-id s2
    python run_complete_pipeline.py --subject-id s3 --skip-ridge
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description='Complete finetuned_lora analysis pipeline'
    )
    p.add_argument('--subject-id', required=True, help='Subject ID (s2, s3)')
    p.add_argument('--skip-ridge', action='store_true',
                   help='Skip ridge regression (if already run)')
    p.add_argument('--skip-top-voxels', action='store_true',
                   help='Skip top voxel extraction')
    p.add_argument('--skip-shap', action='store_true',
                   help='Skip SHAP analysis')
    p.add_argument('--chunk-size', type=int, default=500)
    p.add_argument('--n-alphas', type=int, default=10)
    p.add_argument('--top-percentile', type=float, default=5)
    p.add_argument('--n-top-voxels', type=int, default=None)
    p.add_argument('--save-weights', action='store_true', default=False,
                   help='Save model weights for SHAP analysis')
    return p.parse_args()


def run_command(cmd, step_name):
    """Run a shell command and handle errors."""
    print("\n" + "=" * 80)
    print(f"STEP: {step_name}")
    print("=" * 80)
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nERROR: {step_name} failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"\n✓ {step_name} completed successfully")


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n" + "=" * 80)
    print(f"FINETUNED_LORA ANALYSIS PIPELINE - Subject {args.subject_id}")
    print("=" * 80)
    
    # ── Step 1: Ridge Regression ─────────────────────────────────────────────
    if not args.skip_ridge:
        cmd = [
            sys.executable, 'run_finetuned_lora.py',
            '--subject-id', args.subject_id,
            '--chunk-size', str(args.chunk_size),
            '--n-alphas', str(args.n_alphas),
        ]
        if args.save_weights:
            cmd.append('--save-weights')
        run_command(cmd, "Ridge Regression")
    else:
        print("\n✓ Skipping ridge regression (use --skip-ridge)")
    
    # ── Step 2: Extract Top Voxels ───────────────────────────────────────────
    if not args.skip_top_voxels:
        cmd = [
            sys.executable, 'extract_top_voxels.py',
            '--subject-id', args.subject_id,
            '--method', 'finetuned_lora',
            '--top-percentile', str(args.top_percentile),
        ]
        run_command(cmd, "Extract Top Voxels")
    else:
        print("\n✓ Skipping top voxel extraction (use --skip-top-voxels)")
    
    # ── Step 3: SHAP Analysis ────────────────────────────────────────────────
    if not args.skip_shap:
        cmd = [
            sys.executable, 'shap_analysis.py',
            '--subject-id', args.subject_id,
            '--method', 'finetuned_lora',
            '--top-percentile', str(args.top_percentile),
        ]
        if args.n_top_voxels is not None:
            cmd.extend(['--n-top-voxels', str(args.n_top_voxels)])
        run_command(cmd, "SHAP Analysis")
    else:
        print("\n✓ Skipping SHAP analysis (use --skip-shap)")
    
    # ── Pipeline Complete ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("✓ PIPELINE COMPLETE")
    print("=" * 80)
    print("\nOutput files:")
    print(f"  Metrics: results/metrics/s{args.subject_id[1]}_finetuned_lora_test_corrs.npy")
    print(f"  Models:  results/models/s{args.subject_id[1]}_finetuned_lora/")
    print(f"  Top Voxels: results/top_voxels/")
    print(f"  SHAP Analysis: results/shap_analysis/")


if __name__ == '__main__':
    main()
