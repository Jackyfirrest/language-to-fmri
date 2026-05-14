#!/usr/bin/env python3
"""
Utility to map embedding dimensions to interpretable features.

This script attempts to identify what each embedding dimension represents
by analyzing the top words/features for each dimension.

Note: This requires access to the original embedding vocabulary or 
interpretation data, which may not be available for all embeddings.

Usage:
    python map_embeddings_to_words.py --subject-id s2 --method finetuned_lora
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
from pathlib import Path

try:
    from transformers import AutoTokenizer
except ImportError:
    print("WARNING: transformers not installed. Install with: pip install transformers")


def parse_args():
    p = argparse.ArgumentParser(
        description='Map embedding dimensions to interpretable features'
    )
    p.add_argument('--subject-id', required=True, help='Subject ID (s2, s3)')
    p.add_argument('--method', required=True, help='Embedding method')
    p.add_argument('--result-dir', default='../../../results')
    p.add_argument('--embedding-dir', default='../../../data/embeddings')
    p.add_argument('--output-dir', default='../../../results/embedding_maps')
    p.add_argument('--top-k', type=int, default=20,
                   help='Show top-k words per dimension')
    return p.parse_args()


def analyze_embedding_space(X, top_k=20):
    """Analyze embedding space to find characteristic features per dimension.
    
    For dense embeddings, analyze the distribution of values along each dimension.
    Returns statistics per dimension.
    """
    n_features = X.shape[1]
    
    feature_stats = []
    for dim in range(n_features):
        values = X[:, dim]
        stats = {
            'dimension': int(dim),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75)),
        }
        feature_stats.append(stats)
    
    return feature_stats


def get_bert_tokenizer_info():
    """Get BERT tokenizer for fine-tuned LoRA embeddings.
    
    This assumes the model is based on BERT. Adjust as needed for your specific model.
    """
    try:
        # Typical HuggingFace BERT model
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        return tokenizer
    except Exception as e:
        print(f"WARNING: Could not load BERT tokenizer: {e}")
        return None


def main():
    args = parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    def resolve(path):
        return path if os.path.isabs(path) else \
            os.path.normpath(os.path.join(script_dir, path))
    
    embedding_dir = resolve(args.embedding_dir)
    output_dir = resolve(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"Analyzing embedding space: {args.subject_id} / {args.method}")
    print("=" * 70)
    
    # ── Load embeddings ──────────────────────────────────────────────────────
    def find_embedding(embedding_dir, sid, split, method):
        for ext in ('npy', 'npz'):
            p = os.path.join(embedding_dir, f'{sid}_{split}_{method}_embeddings.{ext}')
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f'No embedding found for {sid}/{split}/{method}')
    
    try:
        train_path = find_embedding(embedding_dir, args.subject_id, 'train', args.method)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    print(f"Loading embeddings from {train_path}")
    
    if train_path.endswith('.npy'):
        X_train = np.load(train_path, mmap_mode='r')
    else:
        # .npz file
        data = np.load(train_path, allow_pickle=False)
        key = 'X' if 'X' in data.files else data.files[0]
        X_train = data[key]
    
    X_train = np.asarray(X_train, dtype=np.float32)
    print(f"Embeddings shape: {X_train.shape}")
    print(f"  Samples: {X_train.shape[0]}")
    print(f"  Dimensions: {X_train.shape[1]}")
    
    # ── Analyze embedding space ──────────────────────────────────────────────
    print("\nAnalyzing embedding space...")
    feature_stats = analyze_embedding_space(X_train, top_k=args.top_k)
    
    # Find most variance-heavy dimensions
    std_values = [s['std'] for s in feature_stats]
    top_std_dims = np.argsort(-np.array(std_values))[:args.top_k]
    
    print(f"\nTop {args.top_k} dimensions by standard deviation:")
    for rank, dim in enumerate(top_std_dims, 1):
        stats = feature_stats[dim]
        print(f"  {rank:2d}. Dim {dim:4d}: "
              f"μ={stats['mean']:7.4f}, σ={stats['std']:7.4f}, "
              f"range=[{stats['min']:7.4f}, {stats['max']:7.4f}]")
    
    # Find dimensions with interesting value patterns
    print(f"\nTop {args.top_k} dimensions by range (max - min):")
    ranges = [s['max'] - s['min'] for s in feature_stats]
    top_range_dims = np.argsort(-np.array(ranges))[:args.top_k]
    
    for rank, dim in enumerate(top_range_dims, 1):
        stats = feature_stats[dim]
        r = stats['max'] - stats['min']
        print(f"  {rank:2d}. Dim {dim:4d}: range={r:7.4f}, "
              f"μ={stats['mean']:7.4f}, σ={stats['std']:7.4f}")
    
    # ── Try to load tokenizer information ─────────────────────────────────
    print("\n" + "-" * 70)
    print("Attempting to identify embedding model type...")
    print("-" * 70)
    
    if 'bert' in args.method.lower() or 'finetuned' in args.method.lower():
        print(f"Method '{args.method}' appears to be BERT-based")
        print("BERT embeddings are {embedding_dim}-dimensional vectors")
        print("Each dimension captures different semantic/syntactic properties")
        print("\nFor fine-tuned models:")
        print("  - Dimensions are typically learned from task-specific data")
        print("  - Not directly interpretable as individual words/features")
        print("  - SHAP analysis (from shap_analysis.py) shows which")
        print("    dimensions are most important for predicting brain activity")
    else:
        print(f"Method '{args.method}' type not recognized")
    
    # ── Save analysis ────────────────────────────────────────────────────────
    output_file = os.path.join(output_dir,
                              f'{args.subject_id}_{args.method}_embedding_stats.json')
    
    output_data = {
        'subject_id': args.subject_id,
        'method': args.method,
        'embedding_shape': list(X_train.shape),
        'all_feature_stats': feature_stats,
        'top_std_dims': top_std_dims.tolist(),
        'top_range_dims': top_range_dims.tolist(),
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved embedding statistics to: {output_file}")
    
    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\n1. For semantic interpretation:")
    print("   - Use SHAP analysis (shap_analysis.py) to identify which")
    print("     dimensions are important for brain predictions")
    print("   - Analyze top voxels to see which dimensions activate them")
    print("\n2. For word-level interpretation:")
    print("   - If you have the original text corpus + word embeddings,")
    print("     you can analyze which words have high/low values in")
    print("     the top dimensions identified by SHAP")
    print("\n3. For model understanding:")
    print("   - Use visualize_results.py to see SHAP feature importance")
    print("   - Compare different embedding methods (word2vec, glove, etc.)")
    print("   - Analyze which semantic features drive brain activity")


if __name__ == '__main__':
    main()
