#!/usr/bin/env python3
"""
Visualization and summary script for finetuned_lora analysis results.

Generates plots and statistical summaries of:
- Ridge regression correlations
- Top voxel distributions
- SHAP feature importance

Usage:
    python visualize_results.py --subject-id s2 --method finetuned_lora
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
except ImportError:
    print("WARNING: matplotlib not installed. Install with: pip install matplotlib")
    plt = None


def parse_args():
    p = argparse.ArgumentParser(
        description='Visualize finetuned_lora analysis results'
    )
    p.add_argument('--subject-id', required=True, help='Subject ID (s2, s3)')
    p.add_argument('--method', required=True, help='Embedding method')
    p.add_argument('--result-dir', default='../../../results')
    p.add_argument('--output-dir', default='../../../results/figures')
    p.add_argument('--top-percentile', type=float, default=5)
    p.add_argument('--dpi', type=int, default=150)
    return p.parse_args()


def resolve(path, script_dir=None):
    """Resolve relative paths."""
    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    return path if os.path.isabs(path) else \
        os.path.normpath(os.path.join(script_dir, path))


def plot_correlation_distribution(test_corrs, top_mask, output_file, dpi=150):
    """Plot distribution of test correlations."""
    if plt is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax = axes[0]
    ax.hist(test_corrs, bins=50, alpha=0.7, label='All voxels', edgecolor='black')
    ax.hist(test_corrs[top_mask], bins=50, alpha=0.7, label='Top 5%', edgecolor='black')
    ax.set_xlabel('Test Correlation', fontsize=12)
    ax.set_ylabel('Number of Voxels', fontsize=12)
    ax.set_title('Distribution of Ridge Regression Correlations', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Cumulative distribution
    ax = axes[1]
    sorted_corrs = np.sort(test_corrs)
    cumsum = np.arange(1, len(sorted_corrs) + 1) / len(sorted_corrs)
    ax.plot(sorted_corrs, cumsum, linewidth=2, label='All voxels')
    ax.axvline(np.percentile(test_corrs, 95), color='r', linestyle='--', 
               linewidth=2, label='95th percentile (top 5%)')
    ax.set_xlabel('Test Correlation', fontsize=12)
    ax.set_ylabel('Cumulative Fraction', fontsize=12)
    ax.set_title('Cumulative Distribution of Correlations', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation plot: {output_file}")


def plot_shap_importance(top_100_dims, top_20_importances, output_file, dpi=150):
    """Plot SHAP feature importance."""
    if plt is None:
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot top 20 dimensions
    dims_to_plot = top_100_dims[:20]
    importances_to_plot = top_20_importances[:20]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(dims_to_plot)))
    bars = ax.barh(range(len(dims_to_plot)), importances_to_plot, color=colors)
    
    ax.set_yticks(range(len(dims_to_plot)))
    ax.set_yticklabels([f"Dim {d}" for d in dims_to_plot], fontsize=10)
    ax.set_xlabel('SHAP Feature Importance (Mean |SHAP|)', fontsize=12)
    ax.set_title('Top 20 Most Influential Embedding Dimensions\n(SHAP Analysis)', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    # Invert y-axis to show top features at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP importance plot: {output_file}")


def print_summary_statistics(test_corrs, top_corrs):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    print("\nAll Voxels:")
    print(f"  Count: {len(test_corrs)}")
    print(f"  Mean CC: {np.mean(test_corrs):.4f}")
    print(f"  Median CC: {np.median(test_corrs):.4f}")
    print(f"  Std CC: {np.std(test_corrs):.4f}")
    print(f"  Min CC: {np.min(test_corrs):.4f}")
    print(f"  Max CC: {np.max(test_corrs):.4f}")
    
    print("\nTop 5% Voxels:")
    print(f"  Count: {len(top_corrs)}")
    print(f"  Mean CC: {np.mean(top_corrs):.4f}")
    print(f"  Median CC: {np.median(top_corrs):.4f}")
    print(f"  Std CC: {np.std(top_corrs):.4f}")
    print(f"  Min CC: {np.min(top_corrs):.4f}")
    print(f"  Max CC: {np.max(top_corrs):.4f}")
    
    print("\nComparison:")
    print(f"  Top voxels / All voxels mean ratio: {np.mean(top_corrs) / np.mean(test_corrs):.2f}x")
    print(f"  Percentile of top voxels mean: {np.percentile(test_corrs, np.mean(top_corrs) * 100):.1f}")


def main():
    args = parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = resolve(args.result_dir, script_dir)
    output_dir = resolve(args.output_dir, script_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"Visualizing results: {args.subject_id} / {args.method}")
    print("=" * 70)
    
    # ── Load ridge regression results ────────────────────────────────────────
    corr_file = os.path.join(result_dir, 'metrics',
                             f'{args.subject_id}_{args.method}_test_corrs.npy')
    
    if not os.path.exists(corr_file):
        print(f"ERROR: Correlation file not found: {corr_file}")
        sys.exit(1)
    
    test_corrs = np.load(corr_file)
    print(f"Loaded {len(test_corrs)} voxel correlations")
    
    # Extract top voxels
    threshold = np.percentile(test_corrs, 100 - args.top_percentile)
    top_mask = test_corrs >= threshold
    top_corrs = test_corrs[top_mask]
    
    print(f"Top {args.top_percentile}% threshold: {threshold:.4f}")
    print(f"Number of top voxels: {len(top_corrs)}")
    
    # ── Plot correlations ────────────────────────────────────────────────────
    plot_file = os.path.join(output_dir,
                            f'{args.subject_id}_{args.method}_correlations.png')
    plot_correlation_distribution(test_corrs, top_mask, plot_file, dpi=args.dpi)
    
    # ── Load and plot SHAP results ──────────────────────────────────────────
    shap_summary_file = os.path.join(output_dir, '..',
                                     f'shap_analysis/{args.subject_id}_{args.method}_shap_summary.json')
    
    if os.path.exists(shap_summary_file):
        print(f"\nLoading SHAP summary from {shap_summary_file}")
        with open(shap_summary_file) as f:
            shap_summary = json.load(f)
        
        top_100_dims = shap_summary['top_100_dims']
        top_20_importances = shap_summary['top_20_importances']
        
        shap_plot_file = os.path.join(output_dir,
                                     f'{args.subject_id}_{args.method}_shap_importance.png')
        plot_shap_importance(top_100_dims, top_20_importances, shap_plot_file, 
                           dpi=args.dpi)
    else:
        print(f"\nSHAP summary not found: {shap_summary_file}")
        print("Run SHAP analysis first")
    
    # ── Print statistics ────────────────────────────────────────────────────
    print_summary_statistics(test_corrs, top_corrs)
    
    print("\n" + "=" * 70)
    print("Visualizations saved to:")
    print(f"  {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
