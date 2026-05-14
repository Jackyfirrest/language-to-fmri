#!/usr/bin/env python3
"""
Setup and validation script for finetuned_lora analysis pipeline.

Checks:
- Python packages installed
- Embedding files available
- Output directories creatable
- fMRI data accessible

Usage:
    python setup_pipeline.py --subject-id s2
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def check_python_packages():
    """Check if required Python packages are available."""
    print("\n" + "=" * 70)
    print("Checking Python Packages")
    print("=" * 70)
    
    packages = {
        'numpy': 'Core numerical computing',
        'scipy': 'Scientific computing (sparse matrices, etc.)',
        'sklearn': 'Scikit-learn for preprocessing',
        'shap': 'SHAP for feature importance (required for shap_analysis.py)',
        'matplotlib': 'Plotting (required for visualize_results.py)',
        'transformers': 'HuggingFace transformers (optional)',
    }
    
    missing = []
    for package, description in packages.items():
        try:
            __import__(package)
            status = "✓"
        except ImportError:
            status = "✗"
            missing.append(package)
        
        required = "(required)" if package in ['numpy', 'scipy', 'sklearn', 'shap'] else "(optional)"
        print(f"  {status} {package:15s} {required:15s} - {description}")
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All required packages installed")
    return True


def check_embeddings(embedding_dir, subject_id):
    """Check if embedding files exist."""
    print("\n" + "=" * 70)
    print("Checking Embedding Files")
    print("=" * 70)
    
    embeddings_needed = [
        f'{subject_id}_train_finetuned_lora_embeddings.npz',
        f'{subject_id}_test_finetuned_lora_embeddings.npz',
    ]
    
    all_exist = True
    for emb_file in embeddings_needed:
        path = os.path.join(embedding_dir, emb_file)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✓ {emb_file:50s} ({size_mb:7.1f} MB)")
        else:
            print(f"  ✗ {emb_file:50s} (NOT FOUND)")
            all_exist = False
    
    if not all_exist:
        print(f"\nMissing embeddings in: {embedding_dir}")
        print("Make sure finetuned_lora embeddings are available")
        return False
    
    print("\n✓ All embedding files found")
    return True


def check_fmri_data(data_path, subject):
    """Check if fMRI data is accessible."""
    print("\n" + "=" * 70)
    print("Checking fMRI Data")
    print("=" * 70)
    
    fmri_dir = os.path.join(data_path, subject)
    
    if not os.path.exists(fmri_dir):
        print(f"  ✗ fMRI directory not found: {fmri_dir}")
        return False
    
    npy_files = list(Path(fmri_dir).glob('*.npy'))
    
    if not npy_files:
        print(f"  ✗ No .npy files found in: {fmri_dir}")
        return False
    
    print(f"  ✓ Found {len(npy_files)} fMRI .npy files")
    total_size_gb = sum(os.path.getsize(f) for f in npy_files) / (1024 ** 3)
    print(f"  ✓ Total fMRI size: {total_size_gb:.1f} GB")
    
    print("\n✓ fMRI data accessible")
    return True


def check_output_directories(result_dir):
    """Check if output directories can be created."""
    print("\n" + "=" * 70)
    print("Checking Output Directories")
    print("=" * 70)
    
    dirs_needed = [
        'metrics',
        'models',
        'top_voxels',
        'shap_analysis',
        'figures',
        'embedding_maps',
    ]
    
    all_ok = True
    for dir_name in dirs_needed:
        dir_path = os.path.join(result_dir, dir_name)
        try:
            os.makedirs(dir_path, exist_ok=True)
            status = "✓"
        except Exception as e:
            status = "✗"
            all_ok = False
            print(f"  {status} {dir_name:20s} - ERROR: {e}")
            continue
        
        print(f"  {status} {dir_name:20s}")
    
    if all_ok:
        print("\n✓ All output directories ready")
    return all_ok


def check_split_file(split_path):
    """Check if train/test split file exists."""
    print("\n" + "=" * 70)
    print("Checking Train/Test Split")
    print("=" * 70)
    
    if not os.path.exists(split_path):
        print(f"  ✗ Split file not found: {split_path}")
        return False
    
    import json
    try:
        with open(split_path) as f:
            split = json.load(f)
        
        n_train = len(split.get('train', []))
        n_test = len(split.get('test', []))
        
        print(f"  ✓ Train stories: {n_train}")
        print(f"  ✓ Test stories: {n_test}")
        print(f"\n✓ Train/test split valid")
        return True
    except Exception as e:
        print(f"  ✗ Error reading split file: {e}")
        return False


def run_test_ridge():
    """Run a quick test of ridge regression."""
    print("\n" + "=" * 70)
    print("Testing Ridge Regression (Optional)")
    print("=" * 70)
    
    response = input("\nRun a quick test with 100 voxels? (y/n): ").strip().lower()
    if response != 'y':
        print("Skipped")
        return True
    
    # TODO: Run a small test
    print("(Test functionality not yet implemented)")
    return True


def main():
    p = argparse.ArgumentParser(
        description='Setup and validate finetuned_lora analysis pipeline'
    )
    p.add_argument('--subject-id', required=True, help='Subject ID (s2, s3)')
    p.add_argument('--subject', default=None, help='Full subject name')
    p.add_argument('--data-path',
                   default='/ocean/projects/mth250011p/shared/215a/final_project/data')
    p.add_argument('--embedding-dir', default='../../../data/embeddings')
    p.add_argument('--split-path', default='../../../data/train_test_split.json')
    p.add_argument('--result-dir', default='../../../results')
    
    args = p.parse_args()
    
    # Auto-detect subject name
    if args.subject is None:
        subject_map = {'s2': 'subject2', 's3': 'subject3'}
        args.subject = subject_map.get(args.subject_id)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    def resolve(path):
        return path if os.path.isabs(path) else \
            os.path.normpath(os.path.join(script_dir, path))
    
    embedding_dir = resolve(args.embedding_dir)
    split_path = resolve(args.split_path)
    result_dir = resolve(args.result_dir)
    
    print("\n" + "=" * 70)
    print("FINETUNED_LORA ANALYSIS PIPELINE - SETUP VALIDATION")
    print("=" * 70)
    print(f"\nSubject ID: {args.subject_id}")
    print(f"Subject: {args.subject}")
    print(f"Data path: {args.data_path}")
    print(f"Embedding dir: {embedding_dir}")
    print(f"Result dir: {result_dir}")
    
    # Run all checks
    results = {
        'Python packages': check_python_packages(),
        'Embeddings': check_embeddings(embedding_dir, args.subject_id),
        'Train/test split': check_split_file(split_path),
        'fMRI data': check_fmri_data(args.data_path, args.subject),
        'Output directories': check_output_directories(result_dir),
    }
    
    # Summary
    print("\n" + "=" * 70)
    print("SETUP VALIDATION SUMMARY")
    print("=" * 70)
    
    for check_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8s} {check_name}")
    
    all_pass = all(results.values())
    
    if all_pass:
        print("\n" + "=" * 70)
        print("✓ ALL CHECKS PASSED - READY TO RUN PIPELINE")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Run ridge regression:")
        print(f"   python run_finetuned_lora.py --subject-id {args.subject_id} --save-weights")
        print("\n2. Or run complete pipeline:")
        print(f"   python run_complete_pipeline.py --subject-id {args.subject_id} --save-weights")
        print("\n3. Check documentation:")
        print("   - QUICKSTART.md (quick examples)")
        print("   - README_FINETUNED_LORA.md (detailed guide)")
        return 0
    else:
        print("\n" + "=" * 70)
        print("✗ SOME CHECKS FAILED - FIX ISSUES BEFORE RUNNING PIPELINE")
        print("=" * 70)
        print("\nSee errors above for details")
        return 1


if __name__ == '__main__':
    sys.exit(main())
