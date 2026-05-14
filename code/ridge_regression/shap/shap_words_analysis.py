#!/usr/bin/env python3
"""
SHAP analysis: measure influential words for brain voxel prediction.

Workflow:
1. Load merged ridge model and test embeddings.
2. For each test story, compute top 2 voxels by correlation.
3. For each voxel, generate voxel response plot (observed vs predicted).
4. Compute word influence scores from top embedding dimensions.
5. Generate word influence bar plots and rankings.
6. Save all outputs in LIME-compatible format.

Usage:
    python shap_words_analysis.py \
      --subject-id s3 \
      --method finetuned_lora \
      --stories onapproachtopluto marryamanwholoveshismother \
      --embedding-dir data/embeddings \
      --split-path data/train_test_split.json \
      --dims-csv results/shap_analysis/s3_finetuned_lora_shap_filtered_onapproachtopluto_dimension_influence.csv \
      --output-dir results/shap_analysis
"""

import argparse
import csv
import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Add code dir to path for ridge_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--subject-id', required=True)
    p.add_argument('--subject', default=None, help='Full subject name (auto-detected if not provided)')
    p.add_argument('--method', default='finetuned_lora')
    p.add_argument('--stories', nargs='+', required=True)
    p.add_argument('--data-path', default='/ocean/projects/mth250011p/shared/215a/final_project/data')
    p.add_argument('--embedding-dir', default='data/embeddings')
    p.add_argument('--embedding-dir', default='../../../data/embeddings')
    p.add_argument('--split-path', default='../../../data/train_test_split.json')
    p.add_argument('--result-dir', default='../../../results')
    p.add_argument('--dims-csv', required=True, help='Path to dimension_influence.csv')
    p.add_argument('--output-dir', default='../../../results/shap_analysis')
    p.add_argument('--top-k-words', type=int, default=100)
    p.add_argument('--top-k-voxels', type=int, default=2, help='Number of voxels per story to visualize')
    p.add_argument('--delays', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    return p.parse_args()


def read_top_dims(csv_path, k=100):
    """Read top dimension indices from CSV."""
    dims = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dims.append(int(row['feature_dim']))
            if len(dims) >= k:
                break
    return dims


def load_embeddings(embedding_dir, sid, method, split='test'):
    """Load test embeddings from .npz file."""
    path = os.path.join(embedding_dir, f'{sid}_{split}_{method}_embeddings.npz')
    data = np.load(path)
    return data['X'].astype(np.float32)


def load_raw_text(data_path):
    """Load raw text with word timing info."""
    path = os.path.join(data_path, 'raw_text.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_split(split_path):
    """Load train/test split."""
    with open(split_path, 'r') as f:
        return json.load(f)


def map_words_to_trs(story_data, delays, trim_start=6, trim_end=4):
    """
    Map each word to its corresponding TR index (accounting for delays).
    
    Returns: word_to_trs[word_idx] = set of TR indices that include this word
    """
    word_times = np.asarray(story_data.data_times, dtype=np.float32)
    tr_times = np.asarray(story_data.tr_times, dtype=np.float32)
    
    # Trim TRs as in preprocessing
    trimmed_tr_times = tr_times[trim_start:len(tr_times)-trim_end]
    
    word_to_trs = {}
    for word_idx, wtime in enumerate(word_times):
        # Find TRs where this word is "active" (considering delays)
        trs_for_word = set()
        for tr_idx, tr_time in enumerate(trimmed_tr_times):
            for delay in delays:
                # Word at time wtime contributes to TR at tr_time + delay
                if abs(wtime - (tr_time + delay)) < 1.0:  # within 1 second
                    trs_for_word.add(tr_idx + delay)
        word_to_trs[word_idx] = trs_for_word
    
    return word_to_trs


def compute_word_influence_scores(
    story_name,
    raw_text,
    X_test_embeddings,
    test_slices,
    top_dims,
    delays,
):
    """
    For each word in the story, compute an influence score based on
    how strongly it activates the top influential dimensions.
    """
    if story_name not in test_slices:
        return None
    
    sl = test_slices[story_name]
    X_story = X_test_embeddings[sl]  # shape: (n_trs, n_dims)
    
    story_data = raw_text[story_name]
    words = story_data.data
    word_times = np.asarray(story_data.data_times, dtype=np.float32)
    
    # Map words to TR indices
    word_to_trs = map_words_to_trs(story_data, delays)
    
    # For each word, aggregate activation across top dimensions
    word_scores = np.zeros(len(words), dtype=np.float32)
    
    for word_idx in range(len(words)):
        tr_indices = word_to_trs.get(word_idx, set())
        if not tr_indices:
            continue
        
        tr_list = sorted(list(tr_indices))
        tr_list = [t for t in tr_list if 0 <= t < X_story.shape[0]]
        
        if not tr_list:
            continue
        
        # Average absolute activation across those TRs and top dims
        for dim in top_dims:
            if dim < X_story.shape[1]:
                activations = X_story[tr_list, dim]
                word_scores[word_idx] += float(np.abs(activations).mean())
    
    # Normalize
    if word_scores.max() > 0:
        word_scores = word_scores / (word_scores.max() + 1e-8)
    
    return word_scores, words


def aggregate_word_influence(stories, raw_text, X_test_embeddings, test_slices, top_dims, delays):
    """Compute word influence for all stories and aggregate."""
    all_words = []
    all_scores = []
    
    for story in stories:
        result = compute_word_influence_scores(story, raw_text, X_test_embeddings, test_slices, top_dims, delays)
        if result is None:
            continue
        
        scores, words = result
        for word, score in zip(words, scores):
            # Clean up word (remove subword markers)
            clean_word = word.replace('##', '')
            if clean_word and len(clean_word) > 1:
                all_words.append(clean_word)
                all_scores.append(float(score))
    
    if not all_words:
        return [], []
    
    # Aggregate by unique word
    word_dict = {}
    for word, score in zip(all_words, all_scores):
        if word not in word_dict:
            word_dict[word] = []
        word_dict[word].append(score)
    
    # Average score per word
    agg_words = []
    agg_scores = []
    for word in sorted(word_dict.keys()):
        scores = word_dict[word]
        avg_score = float(np.mean(scores))
        agg_words.append(word)
        agg_scores.append(avg_score)
    
    # Sort by score
    ranked_idx = np.argsort(-np.array(agg_scores))
    ranked_words = [agg_words[i] for i in ranked_idx]
    ranked_scores = [agg_scores[i] for i in ranked_idx]
    
    return ranked_words, ranked_scores


def save_words_plot(out_path: str, words: list, scores: list, title: str, max_rows: int = 20) -> None:
    """Save bar plot of top words with influence scores (LIME-style)."""
    plot_data = list(zip(words[:max_rows], scores[:max_rows]))
    if not plot_data:
        return
    
    # Reverse for display (top at top)
    plot_data = plot_data[::-1]
    plot_words, plot_scores = zip(*plot_data)
    
    # Color by positive/negative
    colors = ["#54A24B" if val > 0.5 else "#E45756" for val in plot_scores]
    
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(plot_words, plot_scores, color=colors)
    ax.axvline(0.5, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP influence score")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def load_merged_model(result_dir: str, sid: str, method: str):
    """Load merged ridge model."""
    model_path = os.path.join(result_dir, 'models', f'{sid}_{method}', f'{sid}_{method}_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found: {model_path}')
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def load_fmri_data(data_path, subject, test_stories, voxel_indices):
    """Load fMRI test data for specified voxels."""
    fmri_dir = os.path.join(data_path, subject)
    print(f'    fMRI dir: {fmri_dir}')
    
    # Load all fMRI data and concatenate
    all_data = []
    test_lengths = []
    for i, story in enumerate(test_stories):
        fmri_file = os.path.join(fmri_dir, f'{story}.npy')
        print(f'      Loading {story} ({i+1}/{len(test_stories)})...', flush=True)
        story_data = np.load(fmri_file)  # shape: (n_trs, n_voxels_total)
        print(f'        Loaded: {story_data.shape}', flush=True)
        test_lengths.append(story_data.shape[0])
        all_data.append(story_data)
    
    # Concatenate all stories
    print(f'    Concatenating {len(all_data)} arrays...', flush=True)
    Y_all = np.vstack(all_data)
    print(f'    Concatenated shape: {Y_all.shape}', flush=True)
    
    # Extract only the voxels we care about
    print(f'    Extracting {len(voxel_indices)} voxels...', flush=True)
    Y_test = Y_all[:, voxel_indices]
    print(f'    Final shape: {Y_test.shape}', flush=True)
    
    return Y_test, test_lengths, None


def save_response_plot(out_path: str, pred_signal: np.ndarray, truth_signal: np.ndarray, voxel_id: int, story: str, cc: float) -> None:
    """Save voxel response plot (observed vs predicted with peak window)."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    trs = np.arange(pred_signal.shape[0])
    ax.plot(trs, truth_signal, label="Observed", color="#4C78A8", linewidth=1.4, alpha=0.8)
    ax.plot(trs, pred_signal, label="Predicted", color="#E45756", linewidth=1.6)
    
    # Highlight top 5 TRs by prediction
    top_trs = np.argsort(-np.abs(pred_signal))[:5]
    peak_min, peak_max = int(np.min(top_trs)), int(np.max(top_trs))
    ax.axvspan(peak_min, peak_max, color="#F3A712", alpha=0.25)
    
    ax.set_xlabel("TR within story")
    ax.set_ylabel("Voxel response (z-scored)")
    ax.set_title(f"{story} | voxel {voxel_id} | CC={cc:.3f}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    
    # Auto-detect subject name
    if args.subject is None:
        subject_map = {'s2': 'subject2', 's3': 'subject3'}
        args.subject = subject_map.get(args.subject_id)
    
    print('Loading data...')
    top_dims = read_top_dims(args.dims_csv, k=args.top_k_words)
    print(f'Top {len(top_dims)} dimensions loaded')
    
    raw_text = load_raw_text(args.data_path)
    split = load_split(args.split_path)
    test_stories = split['test']
    
    fmri_dir = os.path.join(args.data_path, args.subject)
    story_lengths = []
    for story in test_stories:
        fmri_file = os.path.join(fmri_dir, f'{story}.npy')
        story_lengths.append(int(np.load(fmri_file, mmap_mode='r').shape[0]))
    
    print('Loading ridge model...')
    model_obj = load_merged_model(args.result_dir, args.subject_id, args.method)
    voxel_indices = np.asarray(model_obj['voxel_indices'], dtype=np.int32)
    weights = model_obj.get('weights', model_obj.get('coef_', None))
    if weights is None:
        raise ValueError('Model has no weights or coef_ attribute')
    print(f'Model weights shape: {weights.shape}')
    
    print('Loading test embeddings...')
    X_test = load_embeddings(args.embedding_dir, args.subject_id, args.method, split='test')
    print(f'Test embeddings shape: {X_test.shape}')
    
    # Compute story slices from fMRI row counts so embeddings and voxels stay aligned
    test_slices = {}
    offset = 0
    for story, n_samples in zip(test_stories, story_lengths):
        test_slices[story] = slice(offset, offset + n_samples)
        offset += n_samples
    
    if offset != X_test.shape[0]:
        raise ValueError(
            f'Embedding rows ({X_test.shape[0]}) do not match summed fMRI rows ({offset}). '
            'Check that the embeddings were generated with the same trim/alignment.'
        )
    
    # Load fMRI data once
    print('Noting: fMRI data will be loaded per-story on-demand')
    # Don't load all fMRI data upfront - load per story
    
    os.makedirs(args.output_dir, exist_ok=True)
    summary_rows = []
    
    for story in args.stories:
        print(f'\nProcessing {story}...')
        
        # Get word influence scores
        result = compute_word_influence_scores(
            story, raw_text, X_test, test_slices, top_dims, args.delays
        )
        if result is None:
            print(f'  WARNING: Could not process {story}')
            continue
        
        word_scores, words = result
        ranked_idx = np.argsort(-word_scores)[:args.top_k_words]
        top_words = [words[i].replace('##', '') for i in ranked_idx]
        top_scores = [word_scores[i] for i in ranked_idx]
        
        # Save word influence outputs
        out_csv = os.path.join(args.output_dir, f'{args.subject_id}_{story}_shap_words.csv')
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['rank', 'word', 'shap_influence'])
            writer.writeheader()
            for rank, (word, score) in enumerate(zip(top_words, top_scores), 1):
                writer.writerow({'rank': rank, 'word': word, 'shap_influence': float(score)})
        
        out_txt = os.path.join(args.output_dir, f'{args.subject_id}_{story}_shap_words.txt')
        with open(out_txt, 'w') as f:
            for word in top_words:
                f.write(word + '\n')
        
        plot_file = os.path.join(args.output_dir, f'{args.subject_id}_{story}_shap_words.png')
        save_words_plot(
            plot_file,
            top_words,
            top_scores,
            title=f'{story} | Top {len(top_words)} influential words',
            max_rows=20
        )
        
        print(f'  Saved word analysis: {out_csv}, {out_txt}, {plot_file}')
        print(f'  Top 5 words: {", ".join(top_words[:5])}')
        
        # Get story embeddings and fMRI data
        print(f'\n  Loading fMRI for story: {story}')
        sl = test_slices[story]
        fmri_file = os.path.join(args.data_path, args.subject, f'{story}.npy')
        fmri_data = np.load(fmri_file)[:, voxel_indices]
        Y_story = fmri_data
        
        # Compute story embeddings
        sl = test_slices[story]
        X_story = X_test[sl]
        
        # Compute predictions for this story only (much smaller matrix mult)
        print(f'  Computing predictions for story ({X_story.shape[0]} samples)...')
        pred_story = np.asarray(X_story, dtype=np.float32) @ np.asarray(weights, dtype=np.float32)
        print(f'  Predictions shape: {pred_story.shape}')
        
        # Compute correlation per voxel
        print(f'  Computing voxel correlations for {Y_story.shape[1]} voxels...')
        voxel_corrs = np.zeros(Y_story.shape[1], dtype=np.float32)
        for v in range(Y_story.shape[1]):
            y = Y_story[:, v]
            p = pred_story[:, v]
            y_z = (y - y.mean()) / np.maximum(y.std(), 1e-8)
            p_z = (p - p.mean()) / np.maximum(p.std(), 1e-8)
            cc = np.corrcoef(y_z, p_z)
            voxel_corrs[v] = cc[0, 1] if cc.shape == (2, 2) else 0.0
        
        # Top K voxels
        top_vox_idx = np.argsort(-voxel_corrs)[:min(args.top_k_voxels, len(voxel_corrs))]
        
        voxel_summary = []
        for rank, vox_idx in enumerate(top_vox_idx, 1):
            voxel_id = int(voxel_indices[vox_idx])
            voxel_cc = float(voxel_corrs[vox_idx])
            
            # Create response plot
            response_file = os.path.join(args.output_dir, f'{args.subject_id}_{story}_voxel{voxel_id:05d}_response.png')
            save_response_plot(
                response_file,
                pred_story[:, vox_idx],
                Y_story[:, vox_idx],
                voxel_id,
                story,
                voxel_cc
            )
            
            voxel_summary.append({
                'story': story,
                'rank': rank,
                'voxel_id': voxel_id,
                'correlation': voxel_cc,
                'response_plot': os.path.abspath(response_file),
            })
            
            print(f'    Voxel {rank}: id={voxel_id}, cc={voxel_cc:.3f}')
        
        # Add to summary
        top_5_words = ', '.join(top_words[:5])
        summary_rows.append({
            'story': story,
            'top_5_words': top_5_words,
            'top_voxels': len(voxel_summary),
            'words_csv': os.path.abspath(out_csv),
            'words_plot': os.path.abspath(plot_file),
        })
    
    # Save summary
    if summary_rows:
        summary_csv = os.path.join(args.output_dir, f'{args.subject_id}_shap_summary.csv')
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(summary_csv, index=False)
        print(f'\nSaved summary: {summary_csv}')
        total_plots = len(summary_rows) + len(summary_rows) * args.top_k_voxels
        print(f'Total plots generated: {total_plots} ({len(summary_rows)} word + {len(summary_rows)*args.top_k_voxels} voxel response)')


if __name__ == '__main__':
    main()
