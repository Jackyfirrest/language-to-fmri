#!/usr/bin/env python3
"""
Created By: Gongyao
Fine-tune BERT using LoRA 

The repo structure has been changed very frequently lately, so I decided to code it this way so you can also parse arguments in the terminal to avoid path problems
    Here is an Example:
    (for data path, enter path to the pkl podcast text file)
    (split json is for train test split json file)
    python finetuned_bert_lora.py \
        --data-path /ocean/projects/mth250011p/shared/215a/final_project/data \
        --split-json data/train_test_split.json \
        --output-dir data/embeddings \
        --epochs 3
"""

import os
import sys
import json
import pickle
import random
import argparse
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from transformers import AutoTokenizer, BertForMaskedLM
from peft import LoraConfig, get_peft_model, PeftModel, TaskType


#Utilities

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

#Here are arguments you can pss in terminal and their default settings
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT with LoRA and extract embeddings.")
    parser.add_argument("--data-path", type=str, default="/ocean/projects/mth250011p/shared/215a/final_project/data")
    parser.add_argument("--split-json", type=str, default="../data/train_test_split.json")
    parser.add_argument("--output-dir", type=str, default="../data/embeddings")
    parser.add_argument("--model-name", type=str, default="google-bert/bert-base-uncased")
    parser.add_argument("--max-words-per-chunk", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--layer", type=str, default="last_hidden_state",
                        choices=["last_hidden_state", "last4_mean", "mid4_mean", "all_mean"])
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "first"])
    parser.add_argument("--epochs", type=int, default=8, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for LoRA")
    parser.add_argument("--window-size", type=int, default=256,
                        help="Sliding window size in words. Set to 0 to use non-overlapping chunks.")
    parser.add_argument("--stride", type=int, default=128,
                        help="Stride in words between consecutive sliding windows.")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip LoRA fine-tuning and load weights from --checkpoint-dir instead.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/lora_epoch8",
                        help="Path to saved PEFT checkpoint to load when --skip-training is set.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="Fraction of training chunks held out for validation loss tracking.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--whole-word-masking", action="store_true",
                        help="Mask all subword tokens of a word together instead of random subwords.")
    parser.add_argument("--mlm-prob", type=float, default=0.15,
                        help="Fraction of tokens selected for MLM masking.")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha scaling.")
    parser.add_argument("--lora-target-ffn", action="store_true",
                        help="Also apply LoRA to FFN intermediate and output dense layers.")
    parser.add_argument("--train-window-size", type=int, default=0,
                        help="Use sliding windows of this size for MLM training chunks. "
                             "0 = non-overlapping chunks of --max-words-per-chunk.")
    parser.add_argument("--train-stride", type=int, default=128,
                        help="Stride for sliding training windows (used when --train-window-size > 0).")
    parser.add_argument("--method-name", type=str, default="finetuned_lora",
                        help="Name used in output filenames: {sid}_{split}_{method-name}_embeddings.npz")
    return parser.parse_args()


#Masking words and fitting LoRA
class PodcastDataset(Dataset):
    def __init__(self, word_chunks, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.word_chunks = word_chunks  # list of word lists, not joined strings
        self.max_length = max_length

    def __len__(self):
        return len(self.word_chunks)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.word_chunks[idx],
            is_split_into_words=True,  # match inference tokenization exactly
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # word_ids maps each token position to its original word index (None for specials)
        word_ids = encoding.encodings[0].word_ids
        word_ids_tensor = torch.tensor(
            [wid if wid is not None else -1 for wid in word_ids], dtype=torch.long
        )
        return {'input_ids': encoding['input_ids'].squeeze(0),
                'word_ids': word_ids_tensor}

def mask_tokens(input_ids, vocab_size, mask_token_id, pad_token_id,
                cls_token_id, sep_token_id, mlm_prob=0.15,
                word_ids=None):
    """MLM masking. If word_ids is provided, masks all subwords of a word together
    (whole-word masking); otherwise masks individual subword tokens at random."""
    labels = input_ids.clone()
    special_tokens_mask = (
        (input_ids == pad_token_id) |
        (input_ids == cls_token_id) |
        (input_ids == sep_token_id)
    )

    if word_ids is not None:
        # Whole-word masking: decide per word, apply to all its subword tokens
        probability_matrix = torch.zeros(labels.shape)
        # word_ids shape: (batch, seq_len); -1 means special token
        for b in range(input_ids.shape[0]):
            unique_words = word_ids[b][word_ids[b] >= 0].unique()
            for wid in unique_words:
                if torch.bernoulli(torch.tensor(mlm_prob)):
                    token_positions = (word_ids[b] == wid).nonzero(as_tuple=True)[0]
                    probability_matrix[b, token_positions] = 1.0
        masked_indices = probability_matrix.bool()
    else:
        probability_matrix = torch.full(labels.shape, mlm_prob)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

    # Never mask special tokens regardless
    masked_indices &= ~special_tokens_mask
    labels[~masked_indices] = -100

    # 80% replace with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id

    # 10% replace with random token
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]

    return input_ids, labels

def train_bert(model, dataloader, val_dataloader, tokenizer, epochs=3, lr=5e-4, device='cuda',
               model_name="google-bert/bert-base-uncased", lora_r=32, lora_alpha=64,
               target_ffn=False, whole_word_masking=False, mlm_prob=0.15):
    print("Setting up LoRA configuration...")
    target_modules = ["query", "key", "value", "dense"]
    if target_ffn:
        target_modules += ["intermediate.dense", "output.dense"]
        print(f"  FFN layers included in LoRA targets")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_ckpt_dir = os.path.join("checkpoints", "lora_best")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            input_ids = batch['input_ids']
            wids = batch.get('word_ids') if whole_word_masking else None
            inputs, labels = mask_tokens(
                input_ids.clone(),
                vocab_size=tokenizer.vocab_size,
                mask_token_id=tokenizer.mask_token_id,
                pad_token_id=tokenizer.pad_token_id,
                cls_token_id=tokenizer.cls_token_id,
                sep_token_id=tokenizer.sep_token_id,
                mlm_prob=mlm_prob,
                word_ids=wids,
            )

            inputs = inputs.to(device)
            labels = labels.to(device)
            attention_mask = (inputs != tokenizer.pad_token_id).long().to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            optimizer.step()

            total_loss += outputs.loss.item()
            progress_bar.set_postfix({'loss': f"{outputs.loss.item():.4f}"})

        scheduler.step()
        avg_train_loss = total_loss / len(dataloader)

        # validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids']
                wids = batch.get('word_ids') if whole_word_masking else None
                inputs, labels = mask_tokens(
                    input_ids.clone(),
                    vocab_size=tokenizer.vocab_size,
                    mask_token_id=tokenizer.mask_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    cls_token_id=tokenizer.cls_token_id,
                    sep_token_id=tokenizer.sep_token_id,
                    mlm_prob=mlm_prob,
                    word_ids=wids,
                )
                inputs = inputs.to(device)
                labels = labels.to(device)
                attention_mask = (inputs != tokenizer.pad_token_id).long().to(device)
                val_loss += model(input_ids=inputs, attention_mask=attention_mask, labels=labels).loss.item()
        avg_val_loss = val_loss / len(val_dataloader)

        print(
            f"Epoch {epoch+1}  train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}"
            f"  lr={scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )

        # save per-epoch checkpoint so a wall-time kill doesn't lose all progress
        ckpt_dir = os.path.join("checkpoints", f"lora_epoch{epoch+1}")
        model.save_pretrained(ckpt_dir)
        print(f"Checkpoint saved to {ckpt_dir}", flush=True)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(best_ckpt_dir)
            print(f"  → New best val_loss={best_val_loss:.4f}, saved to {best_ckpt_dir}", flush=True)

    # load best weights before returning — reload from a clean base to avoid
    # applying the adapter on top of already-LoRA-modified linear layers
    print(f"\nLoading best checkpoint (val_loss={best_val_loss:.4f}) from {best_ckpt_dir}")
    clean_base = BertForMaskedLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(clean_base, best_ckpt_dir)
    model.to(device)
    return model


# BERT extraction. word-level
def words_to_bert_embeddings(words: List[str], tokenizer, model, device: str,
                             max_words_per_chunk: int, batch_size: int, 
                             layer: str, pooling: str) -> np.ndarray:
    if len(words) == 0:
        raise ValueError("Received an empty word list.")

    word_chunks = chunk_list(words, max_words_per_chunk)
    story_word_vectors = []

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(word_chunks), batch_size):
            batch_chunks = word_chunks[batch_start: batch_start + batch_size]
            encoding = tokenizer(
                batch_chunks, is_split_into_words=True, return_tensors="pt",
                padding=True, truncation=True, max_length=512,
                return_attention_mask=True, return_token_type_ids=True,
            )

            batch_encodings = encoding.encodings
            encoding = {k: v.to(device) for k, v in encoding.items()}

            need_hidden = layer in ("last4_mean", "mid4_mean", "all_mean")
            outputs = model(**encoding, output_hidden_states=need_hidden)
            if layer == "last4_mean":
                token_reps = torch.stack(outputs.hidden_states[-4:]).mean(0).cpu()
            elif layer == "mid4_mean":
                token_reps = torch.stack(outputs.hidden_states[5:9]).mean(0).cpu()
            elif layer == "all_mean":
                token_reps = torch.stack(outputs.hidden_states[1:]).mean(0).cpu()
            else:
                token_reps = outputs.last_hidden_state.cpu()

            for batch_index, chunk_words in enumerate(batch_chunks):
                reps_i = token_reps[batch_index]
                word_ids = batch_encodings[batch_index].word_ids

                word_to_token_positions: Dict[int, List[int]] = {}
                for token_pos, word_id in enumerate(word_ids):
                    if word_id is not None:
                        word_to_token_positions.setdefault(word_id, []).append(token_pos)

                chunk_vectors = []
                for word_idx in range(len(chunk_words)):
                    token_positions = word_to_token_positions.get(word_idx, [])
                    if len(token_positions) == 0:
                        vec = torch.zeros(reps_i.shape[-1], dtype=reps_i.dtype)
                    else:
                        subword_vectors = reps_i[token_positions]
                        vec = subword_vectors[0] if pooling == "first" else subword_vectors.mean(dim=0)
                    chunk_vectors.append(vec.numpy())

                story_word_vectors.append(np.vstack(chunk_vectors).astype(np.float32))

    return np.vstack(story_word_vectors).astype(np.float32)

def words_to_bert_embeddings_sliding(
    words: List[str], tokenizer, model, device: str,
    window_size: int = 256, stride: int = 128,
    batch_size: int = 4, layer: str = "last4_mean", pooling: str = "mean",
) -> np.ndarray:
    """Sliding-window BERT extraction: overlapping windows averaged at boundaries."""
    if len(words) == 0:
        raise ValueError("Received an empty word list.")

    n_words = len(words)
    hidden_size = model.config.hidden_size
    emb_sum   = np.zeros((n_words, hidden_size), dtype=np.float64)
    emb_count = np.zeros(n_words, dtype=np.int32)

    # Build overlapping windows
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
                batch_windows, is_split_into_words=True, return_tensors="pt",
                padding=True, truncation=True, max_length=512,
                return_attention_mask=True, return_token_type_ids=True,
            )
            batch_encodings = encoding.encodings
            encoding = {k: v.to(device) for k, v in encoding.items()}

            need_hidden = layer in ("last4_mean", "mid4_mean", "all_mean")
            outputs = model(**encoding, output_hidden_states=need_hidden)
            if layer == "last4_mean":
                token_reps = torch.stack(outputs.hidden_states[-4:]).mean(0).detach().cpu()
            elif layer == "mid4_mean":
                token_reps = torch.stack(outputs.hidden_states[5:9]).mean(0).detach().cpu()
            elif layer == "all_mean":
                token_reps = torch.stack(outputs.hidden_states[1:]).mean(0).detach().cpu()
            else:
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


def build_story_level_bert_vectors(raw_text, stories, tokenizer, model, args):
    bert_vectors = {}
    use_sliding = getattr(args, "window_size", 0) > 0
    for idx, story in enumerate(stories, 1):
        words = raw_text[story].data
        print(f"[{idx}/{len(stories)}] Extracting '{story}' "
              f"({'sliding window' if use_sliding else 'chunks'})...")
        if use_sliding:
            X_story = words_to_bert_embeddings_sliding(
                words=words, tokenizer=tokenizer, model=model, device=args.device,
                window_size=args.window_size, stride=args.stride,
                batch_size=args.batch_size, layer=args.layer, pooling=args.pooling,
            )
        else:
            X_story = words_to_bert_embeddings(
                words=words, tokenizer=tokenizer, model=model, device=args.device,
                max_words_per_chunk=args.max_words_per_chunk, batch_size=args.batch_size,
                layer=args.layer, pooling=args.pooling,
            )
        bert_vectors[story] = X_story
        print(f"    shape: {X_story.shape}", flush=True)
    return bert_vectors


#Here is the pipeline
def main():
    args = parse_args()
    seed_everything(args.seed)

    # inject path for preprocessing_utils
    if "code" not in sys.path:
        sys.path.append("code")
    sys.path.append(os.path.abspath(".."))
    from preprocessing_utils import preprocess_embeddings

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading raw text and splits from {args.data_path}...")
    with open(os.path.join(args.data_path, "raw_text.pkl"), "rb") as f:
        raw_text = pickle.load(f)
    with open(args.split_json, "r") as f:
        split = json.load(f)

    train_stories = split["train"]
    test_stories = split["test"]
    all_stories = train_stories + test_stories

    print("Loading BERT and Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    mlm_model = BertForMaskedLM.from_pretrained(args.model_name)

    if args.skip_training:
        # Load saved LoRA checkpoint — skip training entirely
        print(f"\nSkipping training. Loading LoRA checkpoint from: {args.checkpoint_dir}")
        fine_tuned_peft_model = PeftModel.from_pretrained(mlm_model, args.checkpoint_dir)
        fine_tuned_peft_model.to(args.device)
    else:
        # unsupervised data for training — keep as word lists to match inference tokenization
        print("\nPreparing training data...")
        all_chunks = []
        use_sliding_train = args.train_window_size > 0
        for story in train_stories:
            words = raw_text[story].data
            if use_sliding_train:
                pos = 0
                while pos < len(words):
                    end = min(pos + args.train_window_size, len(words))
                    all_chunks.append(words[pos:end])
                    if end == len(words):
                        break
                    pos += args.train_stride
            else:
                all_chunks.extend(chunk_list(words, args.max_words_per_chunk))

        random.shuffle(all_chunks)
        n_val = max(1, int(len(all_chunks) * args.val_frac))
        val_chunks   = all_chunks[:n_val]
        train_chunks = all_chunks[n_val:]
        max_len = args.train_window_size if use_sliding_train else args.max_words_per_chunk
        print(f"  {len(train_chunks)} train chunks, {n_val} val chunks "
              f"({'sliding win=' + str(max_len) if use_sliding_train else 'non-overlap, size=' + str(max_len)})")

        dataset     = PodcastDataset(train_chunks, tokenizer, max_length=max_len + 2)
        val_dataset = PodcastDataset(val_chunks,   tokenizer, max_length=max_len + 2)
        dataloader     = DataLoader(dataset,     batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        print("\nStarting LoRA Fine-Tuning...")
        fine_tuned_peft_model = train_bert(mlm_model, dataloader, val_dataloader, tokenizer,
                                           epochs=args.epochs, lr=args.lr, device=args.device,
                                           model_name=args.model_name,
                                           lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                                           target_ffn=args.lora_target_ffn,
                                           whole_word_masking=args.whole_word_masking,
                                           mlm_prob=args.mlm_prob)

    # peft wraps the model, base_model.model gives BertForMaskedLM, .bert gives the transformer
    core_bert_model = fine_tuned_peft_model.base_model.model.bert

    # use fine-tuned model to generate embeddings
    print("\nGenerating word-level embeddings using Fine-Tuned Model...")
    bert_vectors = build_story_level_bert_vectors(raw_text, all_stories, tokenizer, core_bert_model, args)

    subject_ids = {"subject2": "s2", "subject3": "s3"}

    # embeddings are story-level (not subject-specific): compute once, save for each subject
    print("\nPreprocessing and Saving Final Matrices...")
    for split_name, stories in [("train", train_stories), ("test", test_stories)]:
        processed = preprocess_embeddings(stories=stories, word_vectors=bert_vectors, wordseqs=raw_text)
        X = np.vstack([processed[story] for story in stories]).astype(np.float32)
        for sid in subject_ids.values():
            out_path = os.path.join(args.output_dir, f"{sid}_{split_name}_{args.method_name}_embeddings.npz")
            np.savez_compressed(out_path, X=X)
            print(f"Saved: {out_path} | Shape: {X.shape}")

    print("\nPipeline Complete!")

if __name__ == "__main__":
    main()
