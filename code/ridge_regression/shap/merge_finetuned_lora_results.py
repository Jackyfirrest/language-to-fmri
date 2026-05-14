#!/usr/bin/env python3
"""
Convenience wrapper to merge parallel finetuned_lora ridge results.

This keeps the original merge script untouched while providing a method-specific
entry point that matches the new SLURM array workflow.

Usage:
    python merge_finetuned_lora_results.py --subject-id s3
    python merge_finetuned_lora_results.py --subject-id s2 --n-tasks 10
"""

import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description='Merge finetuned_lora ridge shards.')
    p.add_argument('--subject-id', required=True, help='Subject ID, e.g. s2 or s3')
    p.add_argument('--n-tasks', type=int, default=None,
                   help='Expected number of SLURM array tasks')
    p.add_argument('--result-dir', default='../../../results')
    return p.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    merge_script = os.path.join(os.path.dirname(script_dir), 'merge_ridge_results.py')

    cmd = [
        sys.executable,
        merge_script,
        '--subject-id', args.subject_id,
        '--method', 'finetuned_lora',
        '--result-dir', args.result_dir,
    ]
    if args.n_tasks is not None:
        cmd.extend(['--n-tasks', str(args.n_tasks)])

    os.execv(sys.executable, cmd)


if __name__ == '__main__':
    main()
