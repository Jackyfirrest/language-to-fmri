import numpy as np
import json

# load the split to know how many stories
with open('data/train_test_split.json') as f:
    split = json.load(f)

train_stories = split['train']  # 81
test_stories = split['test']    # 20

# load and check each file
files = [
    'data/embeddings/s2_train_bow_embeddings.npz',
    'data/embeddings/s2_test_bow_embeddings.npz',
    'data/embeddings/s3_train_bow_embeddings.npz',
    'data/embeddings/s3_test_bow_embeddings.npz',
]

for path in files:
    X = np.load(path)['X']
    print(f"{path.split('/')[-1]}: {X.shape}")

# Results look good:

# All 4 files load correctly ✓
# Columns = 48884 = 12221 vocab × 4 delays ✓ consistent across all files
# s2 and s3 have identical shapes ✓
# Train (26463 rows) >> test (6808 rows) ✓ makes sense, 81 vs 20 stories