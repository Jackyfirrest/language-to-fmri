#!/usr/bin/env python3
"""
Wrapper script to run ridge regression for finetuned_lora method.

This script runs ridge regression using the finetuned_lora embeddings
for subjects s2 and s3, then extracts the top 5% performing voxels.

Usage:
    python run_finetuned_lora.py --subject-id s2
    python run_finetuned_lora.py --subject-id s3 --chunk-size 1000
"""

import argparse
import json
import os
import sys
import subprocess

# Import the main run_ridge function
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_ridge import main as run_ridge_main


def parse_args():
    p = argparse.ArgumentParser(
        description='Run ridge regression for finetuned_lora embeddings'
    )
    p.add_argument('--subject-id', required=True, choices=['s2', 's3'],
                   help='Subject ID')
    p.add_argument('--subject', default=None,
                   help='Full subject name (auto-detected if not provided)')
    p.add_argument('--data-path',
                   default='/ocean/projects/mth250011p/shared/215a/final_project/data')
    p.add_argument('--embedding-dir', default='../../../data/embeddings')
    p.add_argument('--split-path', default='../../../data/train_test_split.json')
    p.add_argument('--result-dir', default='../../../results')
    p.add_argument('--chunk-size', type=int, default=500)
    p.add_argument('--n-alphas', type=int, default=10)
    p.add_argument('--cv-frac', type=float, default=0.2)
    p.add_argument('--save-weights', action='store_true', default=False)
    return p.parse_args()


def main():
    args = parse_args()
    
    # Auto-detect subject name from subject-id
    if args.subject is None:
        subject_map = {'s2': 'subject2', 's3': 'subject3'}
        args.subject = subject_map.get(args.subject_id)
        if not args.subject:
            raise ValueError(f"Unknown subject-id: {args.subject_id}")
    
    print("=" * 80)
    print(f"Running ridge regression for {args.subject_id} with finetuned_lora embeddings")
    print("=" * 80)
    
    # Build argv for run_ridge
    argv = [
        '--subject', args.subject,
        '--subject-id', args.subject_id,
        '--method', 'finetuned_lora',
        '--data-path', args.data_path,
        '--embedding-dir', args.embedding_dir,
        '--split-path', args.split_path,
        '--result-dir', args.result_dir,
        '--chunk-size', str(args.chunk_size),
        '--n-alphas', str(args.n_alphas),
        '--cv-frac', str(args.cv_frac),
    ]
    
    if args.save_weights:
        argv.append('--save-weights')
    
    # Replace sys.argv and call run_ridge
    sys.argv = ['run_ridge.py'] + argv
    run_ridge_main()
    
    print("\n" + "=" * 80)
    print("Ridge regression completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
