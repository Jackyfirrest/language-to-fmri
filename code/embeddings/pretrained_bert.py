#!/usr/bin/env python3
"""
Generate full pretrained BERT embeddings for the Huth Lab stories on Bridges2.

Pipeline
--------
1. Load raw_text.pkl and train/test split
2. Extract one BERT embedding per original word
3. Apply preprocess_embeddings() (downsample / trim / delay)
4. Stack stories into train/test matrices
5. Save compressed .npz files for each subject

Usage
-----
Run on Bridges2 (GPU recommended):

    srun -p GPU --gres=gpu:1 --cpus-per-task=4 --mem=20G --time=04:00:00 --pty bash
    conda activate stat214

Copy the file and run it on the project root directory (same level as `code/` and `data/`):

    python pretrained_bert.py \
        --data-path /ocean/projects/mth250011p/shared/215a/final_project/data \
        --split-json data/train_test_split.json \
        --output-dir data/embeddings

Optional:
    --batch-size 4        # reduce if GPU memory is limited
    --layer last4_mean    # use last 4 layers instead of last layer
    --pooling mean        # subword → word aggregation method
    --save-metadata       # save simple metadata (optional)

Output
------
s2_train_pretrained_bert_embeddings.npz
s2_test_pretrained_bert_embeddings.npz
s3_train_pretrained_bert_embeddings.npz
s3_test_pretrained_bert_embeddings.npz

Notes
-----
- Only final split-level matrices are saved (no story-level outputs).
- X is identical across subjects (derived from raw_text timing),
  but subject-specific filenames are kept for consistency with other pipelines.
"""

import os
import sys
import json
import math
import pickle
import random
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    """Split a list into consecutive chunks of size chunk_size."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pretrained BERT embeddings and save final train/test matrices."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/ocean/projects/mth250011p/shared/215a/final_project/data",
        help="Path containing raw_text.pkl and subject folders."
    )
    parser.add_argument(
        "--split-json",
        type=str,
        default="data/train_test_split.json",
        help="Path to train_test_split.json."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/embeddings",
        help="Directory to save final .npz files."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google-bert/bert-base-uncased",
        help="Hugging Face model name."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        help="Hugging Face cache directory."
    )
    parser.add_argument(
        "--max-words-per-chunk",
        type=int,
        default=128,
        help="Maximum number of original words per chunk (used only without sliding window)."
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=256,
        help="Sliding window size in words. Set to 0 to disable sliding window."
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Stride in words between consecutive sliding windows."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of word chunks per BERT forward pass."
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="last_hidden_state",
        choices=["last_hidden_state", "last4_mean"],
        help="Which BERT layer representation to use."
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "first"],
        help="How to pool subword token vectors back to one vector per original word."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Compute device."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--save-metadata",
        action="store_true",
        help="Save a small JSON metadata file for each output split."
    )
    parser.add_argument(
        "--method-name",
        type=str,
        default="pretrained_bert",
        help="Name used in output filenames: {sid}_{split}_{method-name}_embeddings.npz"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# BERT loading
# ---------------------------------------------------------------------
def load_bert(model_name: str, cache_dir: str, device: str):
    """Load tokenizer and model."""
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        cache_dir=cache_dir,
    )

    print(f"Loading model: {model_name}")
    model = AutoModel.from_pretrained(
        model_name,
        cache_dir=cache_dir,
    )

    model.eval()
    model.to(device)

    hidden_size = model.config.hidden_size
    print(f"Hidden size: {hidden_size}")
    print(f"Using device: {device}")
    return tokenizer, model, hidden_size


# ---------------------------------------------------------------------
# Word-level BERT extraction
# ---------------------------------------------------------------------
def words_to_bert_embeddings(
    words: List[str],
    tokenizer,
    model,
    device: str,
    max_words_per_chunk: int = 128,
    batch_size: int = 4,
    layer: str = "last_hidden_state",
    pooling: str = "mean",
) -> np.ndarray:
    """
    Return one contextual embedding per original word.

    Pipeline:
    1. Split words into manageable chunks.
    2. Tokenize with is_split_into_words=True.
    3. Run BERT.
    4. Map subword token representations back to original words.
    5. Pool subwords into a single vector per original word.
    """
    if len(words) == 0:
        raise ValueError("Received an empty word list.")

    word_chunks = chunk_list(words, max_words_per_chunk)
    story_word_vectors = []

    with torch.no_grad():
        for batch_start in range(0, len(word_chunks), batch_size):
            batch_chunks = word_chunks[batch_start: batch_start + batch_size]

            encoding = tokenizer(
                batch_chunks,
                is_split_into_words=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                return_attention_mask=True,
                return_token_type_ids=True,
            )

            # Save fast-tokenizer metadata before moving tensors to device.
            batch_encodings = encoding.encodings
            encoding = {k: v.to(device) for k, v in encoding.items()}

            if layer == "last4_mean":
                outputs = model(**encoding, output_hidden_states=True)
                stacked = torch.stack(outputs.hidden_states[-4:], dim=0)  # (4, B, T, H)
                token_reps = stacked.mean(dim=0)  # (B, T, H)
            else:
                outputs = model(**encoding)
                token_reps = outputs.last_hidden_state  # (B, T, H)

            token_reps = token_reps.detach().cpu()

            for batch_index, chunk_words in enumerate(batch_chunks):
                reps_i = token_reps[batch_index]  # (T, H)
                word_ids = batch_encodings[batch_index].word_ids

                word_to_token_positions: Dict[int, List[int]] = {}
                for token_pos, word_id in enumerate(word_ids):
                    if word_id is None:
                        continue
                    word_to_token_positions.setdefault(word_id, []).append(token_pos)

                chunk_vectors = []
                for word_idx in range(len(chunk_words)):
                    token_positions = word_to_token_positions.get(word_idx, [])

                    if len(token_positions) == 0:
                        # This should be rare, but use zeros to preserve alignment.
                        vec = torch.zeros(reps_i.shape[-1], dtype=reps_i.dtype)
                    else:
                        subword_vectors = reps_i[token_positions]  # (n_subwords, H)
                        if pooling == "first":
                            vec = subword_vectors[0]
                        else:
                            vec = subword_vectors.mean(dim=0)

                    chunk_vectors.append(vec.numpy())

                chunk_vectors = np.vstack(chunk_vectors).astype(np.float32)

                if chunk_vectors.shape[0] != len(chunk_words):
                    raise RuntimeError(
                        f"Word/subword remapping failed: got {chunk_vectors.shape[0]} "
                        f"vectors for {len(chunk_words)} words."
                    )

                story_word_vectors.append(chunk_vectors)

    final_vectors = np.vstack(story_word_vectors).astype(np.float32)

    if final_vectors.shape[0] != len(words):
        raise RuntimeError(
            f"Final word-level embedding count mismatch: got {final_vectors.shape[0]} "
            f"rows for {len(words)} words."
        )

    return final_vectors


def words_to_bert_embeddings_sliding(
    words: List[str],
    tokenizer,
    model,
    device: str,
    window_size: int = 256,
    stride: int = 128,
    batch_size: int = 4,
    layer: str = "last4_mean",
    pooling: str = "mean",
) -> np.ndarray:
    """
    Return one contextual embedding per original word using a sliding window.

    Each word is embedded in the context of up to window_size surrounding words.
    Words in overlapping regions get embeddings from multiple windows averaged
    together, giving boundary words access to context from both sides.

    Window size of 256 words stays safely under BERT's 512 token limit
    for typical English text (~1.3 tokens/word on average).
    """
    if len(words) == 0:
        raise ValueError("Received an empty word list.")

    n_words = len(words)
    hidden_size = model.config.hidden_size
    emb_sum = np.zeros((n_words, hidden_size), dtype=np.float64)
    emb_count = np.zeros(n_words, dtype=np.int32)

    # build list of (start, end) windows
    windows, starts = [], []
    pos = 0
    while pos < n_words:
        end = min(pos + window_size, n_words)
        windows.append(words[pos:end])
        starts.append(pos)
        if end == n_words:
            break
        pos += stride

    model.eval()
    with torch.no_grad():
        for b in range(0, len(windows), batch_size):
            batch_windows = windows[b: b + batch_size]
            batch_starts  = starts[b: b + batch_size]

            encoding = tokenizer(
                batch_windows,
                is_split_into_words=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                return_attention_mask=True,
                return_token_type_ids=True,
            )
            batch_encodings = encoding.encodings
            encoding = {k: v.to(device) for k, v in encoding.items()}

            if layer == "last4_mean":
                outputs = model(**encoding, output_hidden_states=True)
                stacked = torch.stack(outputs.hidden_states[-4:], dim=0)
                token_reps = stacked.mean(dim=0).detach().cpu()
            else:
                outputs = model(**encoding)
                token_reps = outputs.last_hidden_state.detach().cpu()

            for bi, (chunk_words, word_offset) in enumerate(zip(batch_windows, batch_starts)):
                reps_i   = token_reps[bi]
                word_ids = batch_encodings[bi].word_ids

                word_to_tokens: Dict[int, List[int]] = {}
                for tok_pos, wid in enumerate(word_ids):
                    if wid is not None:
                        word_to_tokens.setdefault(wid, []).append(tok_pos)

                for local_idx in range(len(chunk_words)):
                    global_idx = word_offset + local_idx
                    positions  = word_to_tokens.get(local_idx, [])
                    if positions:
                        svecs = reps_i[positions]
                        vec   = svecs[0] if pooling == "first" else svecs.mean(dim=0)
                    else:
                        vec = torch.zeros(hidden_size, dtype=reps_i.dtype)
                    emb_sum[global_idx]   += vec.numpy()
                    emb_count[global_idx] += 1

    emb_count = np.maximum(emb_count, 1)
    return (emb_sum / emb_count[:, None]).astype(np.float32)


def build_story_level_bert_vectors(
    raw_text: Dict,
    stories: List[str],
    tokenizer,
    model,
    device: str,
    max_words_per_chunk: int,
    batch_size: int,
    layer: str,
    pooling: str,
    window_size: int = 0,
    stride: int = 128,
) -> Dict[str, np.ndarray]:
    """Generate one word-level BERT matrix per story.

    If window_size > 0, uses sliding window extraction for better context.
    Otherwise falls back to non-overlapping chunks.
    """
    bert_vectors: Dict[str, np.ndarray] = {}
    total = len(stories)
    use_sliding = window_size > 0

    for idx, story in enumerate(stories, start=1):
        words = raw_text[story].data
        print(f"[{idx}/{total}] '{story}' (n_words={len(words)}, "
              f"{'sliding window' if use_sliding else 'chunks'})")

        if use_sliding:
            X_story = words_to_bert_embeddings_sliding(
                words=words, tokenizer=tokenizer, model=model, device=device,
                window_size=window_size, stride=stride,
                batch_size=batch_size, layer=layer, pooling=pooling,
            )
        else:
            X_story = words_to_bert_embeddings(
                words=words, tokenizer=tokenizer, model=model, device=device,
                max_words_per_chunk=max_words_per_chunk,
                batch_size=batch_size, layer=layer, pooling=pooling,
            )

        bert_vectors[story] = X_story
        print(f"    word-level shape: {X_story.shape}", flush=True)

    return bert_vectors


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    # Make local project code importable, same idea as the notebook / word2vec script.
    if "code" not in sys.path:
        sys.path.append("code")
    sys.path.append(os.path.abspath(".."))

    from preprocessing_utils import preprocess_embeddings

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Configuration")
    print("=" * 80)
    print(f"data_path           : {args.data_path}")
    print(f"split_json          : {args.split_json}")
    print(f"output_dir          : {args.output_dir}")
    print(f"model_name          : {args.model_name}")
    print(f"cache_dir           : {args.cache_dir}")
    print(f"max_words_per_chunk : {args.max_words_per_chunk}")
    print(f"batch_size          : {args.batch_size}")
    print(f"layer               : {args.layer}")
    print(f"pooling             : {args.pooling}")
    print(f"window_size         : {args.window_size}")
    print(f"stride              : {args.stride}")
    print(f"device              : {args.device}")
    print(f"save_metadata       : {args.save_metadata}")
    print("=" * 80)

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path does not exist: {args.data_path}")
    if not os.path.exists(os.path.join(args.data_path, "raw_text.pkl")):
        raise FileNotFoundError(
            f"raw_text.pkl not found under data path: {args.data_path}"
        )
    if not os.path.exists(args.split_json):
        raise FileNotFoundError(f"Split JSON not found: {args.split_json}")

    print("Loading raw text...")
    with open(os.path.join(args.data_path, "raw_text.pkl"), "rb") as f:
        raw_text = pickle.load(f)

    print("Loading train/test split...")
    with open(args.split_json, "r") as f:
        split = json.load(f)

    train_stories = split["train"]
    test_stories = split["test"]
    all_stories = train_stories + test_stories

    print(f"Number of train stories: {len(train_stories)}")
    print(f"Number of test stories : {len(test_stories)}")
    print(f"Number of all stories  : {len(all_stories)}")

    tokenizer, model, hidden_size = load_bert(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        device=args.device,
    )

    print("\nGenerating word-level BERT embeddings for all stories...")
    bert_vectors = build_story_level_bert_vectors(
        raw_text=raw_text,
        stories=all_stories,
        tokenizer=tokenizer,
        model=model,
        device=args.device,
        max_words_per_chunk=args.max_words_per_chunk,
        batch_size=args.batch_size,
        layer=args.layer,
        pooling=args.pooling,
        window_size=args.window_size,
        stride=args.stride,
    )

    example_story = all_stories[0]
    print("\nSanity check")
    print(f"Example story              : {example_story}")
    print(f"Original number of words   : {len(raw_text[example_story].data)}")
    print(f"Word-level embedding shape : {bert_vectors[example_story].shape}")
    print(f"Second row, first 8 dims    : {bert_vectors[example_story][1][:8]}")

    subjects = ["subject2", "subject3"]
    subject_ids = {"subject2": "s2", "subject3": "s3"}

    print("\nGenerating final train/test matrices...")
    for subject in subjects:
        sid = subject_ids[subject]
        print(f"\nProcessing {subject} ({sid})...")

        for split_name, stories in [("train", train_stories), ("test", test_stories)]:
            print(f"  Preprocessing split: {split_name}")
            processed = preprocess_embeddings(
                stories=stories,
                word_vectors=bert_vectors,
                wordseqs=raw_text,
            )

            X = np.vstack([processed[story] for story in stories]).astype(np.float32)
            out_path = os.path.join(
                args.output_dir,
                f"{sid}_{split_name}_{args.method_name}_embeddings.npz"
            )
            np.savez_compressed(out_path, X=X)

            print(f"  Saved: {out_path}")
            print(f"  Shape: {X.shape}")

            if args.save_metadata:
                meta = {
                    "subject": subject,
                    "subject_id": sid,
                    "split": split_name,
                    "stories": stories,
                    "shape": list(X.shape),
                    "model_name": args.model_name,
                    "hidden_size": hidden_size,
                    "layer": args.layer,
                    "pooling": args.pooling,
                    "max_words_per_chunk": args.max_words_per_chunk,
                    "batch_size": args.batch_size,
                    "data_path": args.data_path,
                }
                meta_path = os.path.join(
                    args.output_dir,
                    f"{sid}_{split_name}_{args.method_name}_embeddings.json"
                )
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
                print(f"  Saved metadata: {meta_path}")

    print("\nDone.")
    print(f"Output directory: {args.output_dir}")
    print("Saved files:")
    for filename in sorted(os.listdir(args.output_dir)):
        print(f"  {os.path.join(args.output_dir, filename)}")


if __name__ == "__main__":
    main()