import numpy as np
import pickle
import os
import random
import json

# Data path 
DATA_PATH = '/scratch/users/s214/lab3'
# DATA_PATH = '/ocean/projects/mth250011p/shared/215a/final_project/data'  # Bridges

# Load one story to check shape
data = np.load(f'{DATA_PATH}/subject2/adollshouse.npy')
# print(data.shape)
# (241, 94251)

# Load raw text
with open(f'{DATA_PATH}/raw_text.pkl', 'rb') as f:
    raw_text = pickle.load(f)

# print(type(raw_text))
# <class 'dict'>
# print(raw_text.keys())

# Get valid stories (intersection of fMRI data and text data)
fmri_stories = set(f.replace('.npy', '') for f in os.listdir(f'{DATA_PATH}/subject2'))
text_stories = set(raw_text.keys())
valid_stories = fmri_stories & text_stories

# print(f"Valid stories: {len(valid_stories)}")
# print(sorted(valid_stories))

# just to be sure let's check that subject 3 has the same stories as subject 2

# fmri_stories_s3 = set(f.replace('.npy', '') for f in os.listdir(f'{DATA_PATH}/subject3'))
# print(f"Subject2 stories: {len(fmri_stories)}")
# print(f"Subject3 stories: {len(fmri_stories_s3)}")
# print(f"Same stories: {fmri_stories == fmri_stories_s3}")

# lets check what the raw_text actually contains for one story
story = 'adollshouse'
# print(type(raw_text[story]))
# output: <class 'ridge_utils.DataSequence.DataSequence'>
# since the object is DataSequence object from ridge_utils, it means that raw_text has already been processed into a custom object
# print(raw_text[story])
# output: <ridge_utils.DataSequence.DataSequence object at 0x767db8159ef0>

# let's inspect further
story_data = raw_text['adollshouse']

# inspect all attributes
print(dir(story_data))

# these are the most likely useful ones based on preprocessing.py
print(story_data.data_times)   # word timestamps
print(story_data.tr_times)     # fMRI TR timestamps  
print(story_data.data)         # actual word data

# story_data.data = list of words (the actual text)
# story_data.data_times = timestamp for each word (in seconds)
# story_data.tr_times = fMRI scan timestamps (every 2 seconds)

# lets check how many words are in one story vs how many TRs
story_data = raw_text['adollshouse']
print(f"Number of words: {len(story_data.data)}")
print(f"Number of TRs: {len(story_data.tr_times)}")
print(f"Number of fMRI timepoints: {data.shape[0]}")


# Train/test split
# random.seed(42)
# valid_stories = sorted(valid_stories)  # sort before shuffle for reproducibility
# random.shuffle(valid_stories)

# n_test = 20
# train_stories = valid_stories[n_test:]
# test_stories = valid_stories[:n_test]

# print(f"Train: {len(train_stories)} stories")
# print(f"Test: {len(test_stories)} stories")

# # Save split
# split = {'train': train_stories, 'test': test_stories}
# with open('data/train_test_split.json', 'w') as f:
#     json.dump(split, f, indent=2)

# print("Train/test split saved to data/train_test_split.json")