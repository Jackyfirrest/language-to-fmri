import numpy as np
import sys
import os

# import provided preprocessing functions from GSI
# make_delayed: creates lagged copies of embeddings to model hemodynamic delay
# downsample_word_vectors: aligns word-rate embeddings to fMRI TR-rate using Lanczos interpolation

sys.path.append(os.path.join(os.path.dirname(__file__), 'provided'))
from preprocessing import downsample_word_vectors, make_delayed

# TR (repetition time): how often the fMRI scanner takes a brain "snapshot": every 2 seconds
TR = 2

# Trim X embeddings to align with pre-trimmed Y (fMRI already has edge TRs removed).
# Per lab instructions: trim first 5 TRs (scanner warmup) and last 10 TRs (BOLD lag).
# 256 - 5 - 10 = 241, which equals the number of rows in the pre-trimmed fMRI.
TRIM_START = 5
TRIM_END = 10

# Hemodynamic delays: brain blood-oxygen response peaks 4-6 seconds after hearing a word
# By creating copies of the embeddings shifted by 1,2,3,4 seconds, we let the ridge model
# figure out which delay best predicts each voxel's response
# This multiplies embedding dimension by 4 (one copy per delay)
DELAYS = [1, 2, 3, 4]


def trim_fmri(fmri_data, trim_start=TRIM_START, trim_end=TRIM_END, tr=TR):
    """Optionally trim additional edge TRs from fMRI data.

    Y is already pre-trimmed during dataset preprocessing, so this is NOT called
    in the main pipeline. Use only if you want to remove additional edge TRs
    (e.g., residual boundary artifacts), and apply the same trim to X.

    Args:
        fmri_data: array of shape (T, V) — timepoints x voxels
        trim_start: TRs to remove from beginning
        trim_end: TRs to remove from end
        tr: repetition time in seconds (how often fMRI scans)

    Returns:
        trimmed fmri array of shape (T', V)
    """
    start_idx = trim_start  # 5 TRs
    end_idx = trim_end   # 10 TRs
    return fmri_data[start_idx:-end_idx]


def preprocess_embeddings(stories, word_vectors, wordseqs,
                          trim_start=TRIM_START, trim_end=TRIM_END,
                          tr=TR, delays=DELAYS):
    """Full preprocessing pipeline to turn word embeddings into model features.

    The problem: word embeddings are at word-rate (3 words/sec, thousands of words),
    but fMRI is at TR-rate (1 scan/2sec, ~250 timepoints). We need to align them.

    Pipeline:
        1. Downsample: word-rate embeddings -> TR-rate using Lanczos interpolation
           (1656 words, embed_dim) -> (256 TRs, embed_dim)

        2. Trim [5:-10]: removes warmup/lag TRs to align with pre-trimmed Y
           (256 TRs, embed_dim) -> (241 TRs, embed_dim)

        3. Delay: create 4 shifted copies to model hemodynamic lag
           (241 TRs, embed_dim) -> (241 TRs, embed_dim * 4)

    Y (fMRI) is already pre-trimmed to 241 rows and is NOT trimmed again.
    After this pipeline, X and Y have matching time dimensions for ridge regression.

    Args:
        stories: list of story names to process e.g. ['adollshouse', 'avatar']
        word_vectors: dict of {story: array (n_words, embed_dim)}
                      one embedding vector per word in the story
        wordseqs: dict of {story: DataSequence} — this is raw_text from the pkl file
                  DataSequence contains .data_times (word timestamps) and
                  .tr_times (fMRI scan timestamps) needed for downsampling
        trim_start: TRs to trim from start (default 5)
        trim_end: TRs to trim from end (default 10)
        tr: repetition time in seconds
        delays: list of delay values in seconds for make_delayed

    Returns:
        dict of {story: array of shape (T', embed_dim * len(delays))}
        T' = 241 trimmed timepoints, matches pre-trimmed fMRI
    """

    # convert sparse matrices to dense one story at a time
    # we can't keep all stories dense at once (memory) but one at a time is fine
    # lanczosinterp2D inside downsample_word_vectors requires dense arrays
    dense_word_vectors = {}
    for story in stories:
        wv = word_vectors[story]
        if hasattr(wv, 'toarray'):  # check if sparse
            dense_word_vectors[story] = wv.toarray().astype(np.float32)
        else:
            dense_word_vectors[story] = wv


    # step 1: downsample using dense vectors
    # Lanczos interpolation maps each word embedding to the nearest TR timepoint
    # uses word timestamps (data_times) and TR timestamps (tr_times) from DataSequence
    # result: one embedding vector per TR instead of one per word
    downsampled = downsample_word_vectors(stories, dense_word_vectors, wordseqs)

    # step 2: trim [10:-5] to align with pre-trimmed Y, then add delays
    processed = {}
    for story in stories:
        arr = downsampled[story]              # shape: (256, embed_dim)
        arr = arr[trim_start:-trim_end]       # [5:-10] -> shape: (241, embed_dim)
        arr = make_delayed(arr, delays)       # shape: (241, embed_dim * 4)
        processed[story] = arr
    return processed


def load_fmri(stories, subject, data_path):
    """Load fMRI data for a list of stories.

    Each story's fMRI file is a (T, V) matrix:
        T = number of fMRI timepoints (varies by story length)
        V = number of voxels = 94251 (fixed, whole brain)

    Y is already pre-trimmed (edge TRs removed during dataset preprocessing),
    so no further trimming is applied here.

    Args:
        stories: list of story names e.g. ['adollshouse', 'avatar']
        subject: subject folder name e.g. 'subject2' or 'subject3'
        data_path: root path to data directory

    Returns:
        dict of {story: array of shape (T, V)}
        T matches the T' from preprocess_embeddings output (e.g. 241 for adollshouse)
    """
    fmri = {}
    for story in stories:
        path = os.path.join(data_path, subject, f'{story}.npy')
        fmri[story] = np.load(path)   # shape: (T, V) — already pre-trimmed
    return fmri


# after preprocess_embeddings and load_fmri, for every story you end up with:
# X (embeddings): (241, embed_dim * 4)  <- features for ridge regression
# Y (fMRI):       (241, 94251)          <- targets for ridge regression (pre-trimmed)