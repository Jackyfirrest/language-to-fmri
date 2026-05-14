"""
GloVe embedding for fMRI language encoding.

Uses the pre-trained GloVe 840B 300-d vectors.
Download from:
  https://nlp.stanford.edu/data/glove.840B.300d.zip  (~2.0 GB compressed)

Place the decompressed text file at:
  lab3/data/raw/glove.840B.300d.txt

Alternatively the smaller 6B / 100-d set works too:
  https://nlp.stanford.edu/data/glove.6B.zip  → glove.6B.100d.txt  (100-d)

The pipeline mirrors bow.py and word2vec.py.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from preprocessing import downsample_word_vectors, make_delayed

DEFAULT_GLOVE_PATH = os.path.join(
    os.path.dirname(__file__), "../../data/raw/glove.840B.300d.txt"
)
GLOVE_DIM = 300  # change to 100 if using the 6B/100d file


def load_glove(glove_path: str = DEFAULT_GLOVE_PATH) -> dict:
    """Load GloVe vectors from a plain-text file into a dict {word: np.ndarray}."""
    print(f"Loading GloVe from {glove_path} …")
    embeddings = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec  = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vec
    print(f"Loaded {len(embeddings):,} GloVe vectors (dim={next(iter(embeddings.values())).shape[0]})")
    return embeddings


def get_glove_vectors(wordseqs: dict, glove: dict, dim: int = GLOVE_DIM) -> dict:
    """Look up each word token in the GloVe dictionary.

    OOV words receive a zero vector.

    Returns:
        {story: np.ndarray of shape (num_words, dim)}
    """
    word_vectors = {}
    oov_total = 0
    for story, ds in wordseqs.items():
        vecs = []
        for word in ds.data:
            w = word.lower()
            if w in glove:
                vecs.append(glove[w])
            else:
                vecs.append(np.zeros(dim, dtype=np.float32))
                oov_total += 1
        word_vectors[story] = np.array(vecs, dtype=np.float32)
    if oov_total:
        print(f"GloVe OOV tokens: {oov_total}")
    return word_vectors


def process_glove(stories_train, stories_test, wordseqs,
                  glove_path=DEFAULT_GLOVE_PATH, dim=GLOVE_DIM,
                  trim_start=5, trim_end=10, delays=range(1, 5)):
    """Full GloVe pipeline: embed → downsample → trim → lag."""
    glove = load_glove(glove_path)

    all_stories = list(set(stories_train) | set(stories_test))
    word_vectors = get_glove_vectors(
        {s: wordseqs[s] for s in all_stories}, glove, dim
    )

    downsampled = downsample_word_vectors(all_stories, word_vectors, wordseqs)

    def _trim_and_lag(stories):
        mats = []
        for story in stories:
            ds = downsampled[story]
            trimmed = ds[trim_start: len(ds) - trim_end]
            lagged  = make_delayed(trimmed, list(delays))
            mats.append(lagged)
        return np.vstack(mats)

    X_train = _trim_and_lag(stories_train)
    X_test  = _trim_and_lag(stories_test)
    return X_train, X_test


if __name__ == "__main__":
    import pickle
    wordseqs   = pickle.load(open(sys.argv[1], "rb"))
    train_list = sys.argv[2].split(",")
    test_list  = sys.argv[3].split(",")
    out_prefix = sys.argv[4]
    glove_path = sys.argv[5] if len(sys.argv) > 5 else DEFAULT_GLOVE_PATH

    X_train, X_test = process_glove(train_list, test_list, wordseqs, glove_path)
    np.save(f"{out_prefix}_train_glove_embeddings.npy", X_train)
    np.save(f"{out_prefix}_test_glove_embeddings.npy",  X_test)
    print(f"Saved GloVe embeddings: train {X_train.shape}, test {X_test.shape}")
