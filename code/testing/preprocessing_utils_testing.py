
# Test script to verify preprocessing_utils.py is working correctly
# Run from repo root: python code/testing/preprocessing_utils_testing.py

import numpy as np
import pickle
import sys
import os

sys.path.append('code')
from preprocessing_utils import preprocess_embeddings, load_fmri, trim_fmri

# Config
DATA_PATH = '/scratch/users/s214/lab3'
# DATA_PATH = '/ocean/projects/mth250011p/shared/215a/final_project/data'  # Bridges

# Load raw text
print("Loading raw text...")
with open(f'{DATA_PATH}/raw_text.pkl', 'rb') as f:
    raw_text = pickle.load(f)

# Test on one story 
test_stories = ['adollshouse']

# Test 1: load_fmri
print("\n[Test 1] load_fmri...")
fmri = load_fmri(test_stories, 'subject2', DATA_PATH)
print(f"  fMRI shape (pre-trimmed, no further trim): {fmri['adollshouse'].shape}")
print(f"  Expected: (241, 94251)")

# Test 2: preprocess_embeddings with dummy BoW vectors
print("\n[Test 2] preprocess_embeddings with dummy BoW...")
words = raw_text['adollshouse'].data
vocab = list(set(words))
word_to_idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)

bow_vectors = {'adollshouse': np.array([
    [1 if word_to_idx[w] == i else 0 for i in range(vocab_size)]
    for w in words
], dtype=np.float32)}

processed = preprocess_embeddings(test_stories, bow_vectors, raw_text)
print(f"  Embedding shape after preprocessing: {processed['adollshouse'].shape}")
print(f"  Expected: (241, {vocab_size * 4})")

# Test 3: X and Y time dimensions match
print("\n[Test 3] X and Y time dimensions match...")
fmri_T = fmri['adollshouse'].shape[0]
emb_T = processed['adollshouse'].shape[0]
assert fmri_T == emb_T, f"MISMATCH: fMRI T={fmri_T}, embedding T={emb_T}"
print(f"  SUCCESS: both have T'={fmri_T} timepoints")

print("\nAll tests passed!")