import numpy as np
import pickle
import json
import os
import sys
import gensim.downloader as api  # gensim is a library for working with word embeddings

# add code/ to path so we can import our preprocessing_utils
sys.path.append('code')
from preprocessing_utils import preprocess_embeddings

# Config
 
DATA_PATH = '/scratch/users/s214/lab3'
 
SUBJECTS = ['subject2', 'subject3']
SUBJECT_IDS = {'subject2': 's2', 'subject3': 's3'}
 
OUTPUT_DIR = 'data/embeddings'
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# Word2Vec embedding dimension
# Google's pretrained model always produces 300-dimensional vectors (this is fixed) by the model and cannot be changed
# Compare to BoW where dimension = vocab size (12000)
EMBED_DIM = 300

# Load the pretrained Word2Vec model
 
# "word2vec-google-news-300" is a model that Google trained on 100 billion words of Google News text

# What gensim.downloader does:
# 1. Downloads the model file (1.6 GB) from the internet on first run
# 2. Caches it locally so future runs are instant
# 3. Returns a KeyedVectors object, essentially a big dictionary where keys are words (strings) and values are 300-dim numpy arrays

# So after this line, word2vec_model['cat'] gives a (300,) array representing the word "cat" in 300-dimensional semantic space
 
print("Loading pretrained Word2Vec model")
word2vec_model = api.load('word2vec-google-news-300')
print(f"Model loaded. Vocabulary size: {len(word2vec_model)} words")

# Load data
 
# raw_text is a dict of {story_name: DataSequence}
# DataSequence contains:
#   .data       = list of words in the story e.g. ['alright', 'so', 'thank', ...]
#   .data_times = timestamp (seconds) for each word — when was this word spoken
#   .tr_times   = fMRI scan timestamps (every 2 seconds) — when each brain scan happened
# These timestamps are what allow us to align word embeddings to fMRI timepoints later
 
print("\nLoading raw text")
with open(f'{DATA_PATH}/raw_text.pkl', 'rb') as f:
    raw_text = pickle.load(f)

print("Loading train/test split")
# we use the exact same train/test split as bow.py
# this is crucial: if each method used a different split, we couldn't fairly compare
with open('data/train_test_split.json', 'r') as f:
    split = json.load(f)
 
train_stories = split['train'] # 81 stories used to fit the ridge model
test_stories  = split['test']  # 20 stories used to evaluate the ridge model
all_stories   = train_stories + test_stories  # all 101 stories


# Generate Word2Vec vectors
 
# In Word2Vec, each word maps to a dense 300-dimensional vector of real numbers that was learned by training on billions of words of text
#   e.g. "cat" -> [0.21, -0.13, 0.87, ..., 0.04] (length 300, all floats)

# The values in the vector encode semantic meaning: words with similar meanings end up with similar vectors, so the distance between vectors reflects meaning

# Why dense? With BoW, almost every entry is 0 (one word out of 12000 vocab)
# With Word2Vec, all 300 entries carry real information, no wasted zeros
 
print("\nGenerating Word2Vec vectors for all stories")
 
# Helper function: look up the Word2Vec vector for a single word.
# We separate this logic into its own function so the main loop stays readable.

def get_word_vector(word, model, embed_dim):
    """Return the Word2Vec vector for a word, or a zero vector if unknown.
 
    Word2Vec was trained on formal news text, so some words from the podcast
    stories won't be in its vocabulary — informal words like "gonna", "uh",
    filler sounds, or rare proper nouns. We can't just skip those words because
    we need one vector per word to keep the sequence aligned with word timestamps.
    Instead, we return a zero vector of the right size as a neutral placeholder.
 
    Args:
        word:      the word string to look up e.g. 'cat'
        model:     the loaded gensim KeyedVectors object
        embed_dim: expected vector length (300 for Google News Word2Vec)
 
    Returns:
        numpy array of shape (embed_dim,) — either the real vector or zeros
    """
    if word in model:
        # word is in the vocabulary: return its learned 300-dim vector
        return model[word]
    else:
        # word is unknown: return zeros so we don't crash and the sequence stays aligned
        return np.zeros(embed_dim, dtype=np.float32)


# Now build the embedding matrix for each story
# For each story we produce an array of shape (n_words, 300):
#   - one row per word in the story
#   - each row is the 300-dim Word2Vec vector for that word

word2vec_vectors = {}
for story in all_stories:
    words = raw_text[story].data # list of word strings for this story

    # look up the vector for every word and stack them into a 2D array
    # np.vstack turns a list of 1D arrays (each shape (300,)) into a 2D array (n_words, 300)
    vectors = np.vstack([
        get_word_vector(word, word2vec_model, EMBED_DIM)
        for word in words
    ])
    # vectors is now shape (n_words, 300) — a dense float32 array
    word2vec_vectors[story] = vectors.astype(np.float32)  # ensure float32 for memory efficiency



# Quick sanity check: print the shape for one story to verify it looks right
example_story = 'adollshouse'
print(f"Example shape for {example_story}: {word2vec_vectors[example_story].shape}")
print(f"  (rows = words in story, cols = 300 Word2Vec dimensions)")
print(f"  stored as dense array: {isinstance(word2vec_vectors[example_story], np.ndarray)}")
 
# Count how many words across all stories fell back to the zero vector (were unknown)
# This is useful to report because many zeros could hurt model quality
total_words   = sum(len(raw_text[s].data) for s in all_stories)
unknown_words = sum(
    sum(1 for word in raw_text[s].data if word not in word2vec_model)
    for s in all_stories
)
print(f"\nOut-of-vocabulary words: {unknown_words} / {total_words} "
      f"({100 * unknown_words / total_words:.1f}%) -> replaced with zero vectors")


# Preprocess and save per subject
for subject in SUBJECTS:
    sid = SUBJECT_IDS[subject]   # 's2' or 's3'
    print(f"\nProcessing {subject} ({sid})...")
 
    for split_name, stories in [('train', train_stories), ('test', test_stories)]:
        print(f"  {split_name}: {len(stories)} stories...")
 
        # Run the full preprocessing pipeline from preprocessing_utils.py:
        #   Step 1 — Downsample (word-rate -> TR-rate):
        #     (n_words, 300) -> (256, 300) using Lanczos interpolation over all tr_times
        #   Step 2 — Trim [10:-5]:
        #     aligns to pre-trimmed Y: (256, 300) -> (241, 300)
        #   Step 3 — Delay (hemodynamic lag):
        #     (241, 300) -> (241, 1200)  [300 * 4 delay copies]

        processed = preprocess_embeddings(
            stories=stories,
            word_vectors=word2vec_vectors,
            wordseqs=raw_text,
        )
        # processed is a dict: {story: array of shape (241, 1200)}
 
        # Stack all stories in this split into one big matrix along the time axis.
        # np.vstack concatenates along axis=0 (rows), so we go from
        #   a list of arrays each shaped (T'_i, 1200)
        #   to one array shaped (sum of all T'_i, 1200)
        # This is the final X matrix for ridge regression: rows=time, cols=features
        X = np.vstack([processed[story] for story in stories])
        print(f"  X shape ({split_name}): {X.shape}")
        # rows = total timepoints across all stories in this split
        # cols = 300 (Word2Vec dims) * 4 (delay copies) = 1200
 
        # Save following the naming convention: {sid}_{split}_{method}_embeddings.npz
        # .npz is numpy's compressed archive format — smaller files than .npy
        # the array is stored under the key 'X' and retrieved later with data['X']
        out_path = os.path.join(OUTPUT_DIR, f'{sid}_{split_name}_word2vec_embeddings.npz')
        np.savez_compressed(out_path, X=X)
        print(f"  Saved to {out_path}")
 
print("\nWord2Vec embeddings done")
print("Files saved to data/embeddings/ as .npz (compressed)")

print(f"  Word2Vec feature columns: {EMBED_DIM} * 4 = {EMBED_DIM * 4}")
