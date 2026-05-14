import numpy as np
import pickle
import json
import os
import sys
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp

# add code/ to path so we can import our preprocessing_utils
sys.path.append('code')
from preprocessing_utils import preprocess_embeddings

# Config 
DATA_PATH = '/scratch/users/s214/lab3'

# the text/embeddings are the same for both subjects (same stories), but fMRI lengths
# differ slightly per subject so we preprocess separately per subject
SUBJECTS = ['subject2', 'subject3']

# short names used in output filenames following naming convention
SUBJECT_IDS = {'subject2': 's2', 'subject3': 's3'}

# where to save the processed embeddings
OUTPUT_DIR = 'data/embeddings'
os.makedirs(OUTPUT_DIR, exist_ok=True) 

# load the data

# raw_text is a dict of {story_name: DataSequence}
# DataSequence contains:
# .data = list of words in the story
# .data_times = timestamp (in seconds) for each word
# .tr_times = fMRI scan timestamps (every 2 seconds)

print("Loading raw text")
with open(f'{DATA_PATH}/raw_text.pkl', 'rb') as f:
    raw_text = pickle.load(f)

print("Loading train/test split")
# load the fixed train/test split we created in eda.py
# using the same split ensures fair comparison across all embedding methods
with open('data/train_test_split.json', 'r') as f:
    split = json.load(f)

train_stories = split['train'] # 81 stories used to fit the ridge model
test_stories = split['test'] # 20 stories used to evaluate the ridge model
all_stories = train_stories + test_stories # all 101 stories combined

# Build vocabulary and BoW vectors using sklearn 
# CountVectorizer handles vocabulary building efficiently
# it also uses sparse matrices internally, only storing non-zero values
# this is critical for BoW because each word vector has only 1 non-zero out of 12000 vocab entries -> storing 12000 zeros per word is wasteful

print("\nBuilding vocabulary and generating BoW vectors")

# CountVectorizer expects full strings, not lists of words so we join each story's word list into one string
# e.g. ['the', 'dog', 'ate'] -> 'the dog ate'
# we do this for ALL stories so vocabulary covers train AND test
story_texts = {story: ' '.join(raw_text[story].data) for story in all_stories}

# fit vocabulary on all stories — builds the word->index mapping internally
# analyzer='word' means it tokenizes by whitespace (our words are already clean)

vectorizer = CountVectorizer(analyzer='word')
vectorizer.fit(story_texts.values())
vocab_size = len(vectorizer.vocabulary_)
print(f"Vocabulary size: {vocab_size} unique words")

# generate one-hot vector per WORD for each story
# we transform each word individually as its own "document"
# result is a sparse matrix of shape (n_words, vocab_size)
# sparse means: only the single 1 is stored, not the thousands of zeros

bow_vectors = {}
for story in all_stories:
    words = raw_text[story].data   # list of words e.g. ['alright', 'thank', ...]
    # transform each word as its own document -> sparse (n_words, vocab_size)
    # unknown words (not in vocab) become all-zero rows — shouldn't happen since we built vocab from all stories
    vectors = vectorizer.transform(words)  # stays sparse, no .toarray() yet
    bow_vectors[story] = vectors

print(f"Example shape for adollshouse: {bow_vectors['adollshouse'].shape}")
print(f"  stored as sparse: {sp.issparse(bow_vectors['adollshouse'])}")



# Preprocess and save per subject

# now we apply the full preprocessing pipeline per subject: downsample -> crop -> trim -> delay

# why per subject?
#   the TEXT is the same for both subjects (same stories, same words)
#   BUT the fMRI recordings have slightly different lengths per subject
#   so we need to crop the embeddings to match each subject's fMRI length

# we also save train and test separately because:
#  - ridge model is FIT on train embeddings + train fMRI
#  - ridge model is EVALUATED on test embeddings + test fMRI

for subject in SUBJECTS:
    sid = SUBJECT_IDS[subject] # s2 or s3
    print(f"\nProcessing {subject} ({sid})...")

    for split_name, stories in [('train', train_stories), ('test', test_stories)]:
        print(f"  {split_name}: {len(stories)} stories...")

        # run the full preprocessing pipeline (defined in preprocessing_utils.py):
        #   1. downsample: (n_words, vocab_size) -> (256, vocab_size)
        #   2. trim [10:-5]: align to pre-trimmed Y -> (241, vocab_size)
        #   3. delay: create 4 shifted copies -> (241, vocab_size * 4)
        # returns a dict of {story: processed_array}

        processed = preprocess_embeddings(
            stories=stories,
            word_vectors=bow_vectors,
            wordseqs=raw_text,
        )

        # stack all stories into one big matrix along the time axis
        # e.g. if each story has 234 timepoints and we have 81 train stories:
        # (81 * 234, vocab_size * 4) -> roughly (18954, vocab_size * 4)
        # this is our final X matrix for ridge regression

        X = np.vstack([processed[story] for story in stories])
        print(f"  X shape ({split_name}): {X.shape}")
        # rows = total timepoints across all stories in this split
        # cols = vocab_size * 4 (4 delay copies)

        # save following naming convention: {subject}_{split}_{method}_embeddings.npy
        # e.g. s2_train_bow_embeddings.npy
        out_path = os.path.join(OUTPUT_DIR, f'{sid}_{split_name}_bow_embeddings.npz')
        np.savez_compressed(out_path, X=X)
        print(f"  Saved to {out_path}")
    
print("\nBoW embeddings done!")
print("Files saved to data/embeddings/ as .npz (compressed)")



        




