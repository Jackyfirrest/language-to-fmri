""" import numpy as np
import pickle
import sys
sys.path.append('code')

DATA_PATH = '/scratch/users/s214/lab3'

with open(f'{DATA_PATH}/raw_text.pkl', 'rb') as f:
    raw_text = pickle.load(f)

story = 'adollshouse'
ds = raw_text[story]

# check tr_times
print(f"tr_times length: {len(ds.tr_times)}")
print(f"tr_times first 10: {ds.tr_times[:10]}")
print(f"tr_times last 5: {ds.tr_times[-5:]}")

# check fMRI shape
fmri = np.load(f'{DATA_PATH}/subject2/{story}.npy')
print(f"\nfMRI raw shape: {fmri.shape}")
print(f"fMRI timepoints: {fmri.shape[0]}")

# the key comparison
print(f"\ntr_times length: {len(ds.tr_times)}")
print(f"fMRI timepoints: {fmri.shape[0]}")
print(f"difference: {len(ds.tr_times) - fmri.shape[0]}") """


""" import numpy as np
import pickle
import sys
sys.path.append('code')

DATA_PATH = '/scratch/users/s214/lab3'

with open(f'{DATA_PATH}/raw_text.pkl', 'rb') as f:
    raw_text = pickle.load(f)

# check multiple stories to see if 15 is consistent or varies
stories_to_check = ['adollshouse', 'avatar', 'sweetaspie', 'life', 'tetris']

print("story | tr_times | fMRI | difference | negative TRs")
print("-" * 65)
for story in stories_to_check:
    ds = raw_text[story]
    fmri = np.load(f'{DATA_PATH}/subject2/{story}.npy')
    
    tr_len = len(ds.tr_times)
    fmri_len = fmri.shape[0]
    diff = tr_len - fmri_len
    negative_trs = np.sum(ds.tr_times < 0)  # count negative TR timestamps
    
    print(f"{story:15} | {tr_len:8} | {fmri_len:4} | {diff:10} | {negative_trs}")

# also check: what are the exact negative tr_times for adollshouse
ds = raw_text['adollshouse']
print(f"\nadollshouse negative tr_times: {ds.tr_times[ds.tr_times < 0]}")
print(f"adollshouse first positive tr_times: {ds.tr_times[ds.tr_times >= 0][:5]}")

# check what the last 15 tr_times look like
ds = raw_text['adollshouse']
fmri = np.load(f'{DATA_PATH}/subject2/adollshouse.npy')

print(f"\nLast 20 tr_times: {ds.tr_times[-20:]}")
print(f"fMRI last timepoint corresponds to TR: {ds.tr_times[fmri.shape[0]-1]}")
print(f"tr_times at fMRI length index: {ds.tr_times[fmri.shape[0]]}")
print(f"\nFirst 10 tr_times: {ds.tr_times[:10]}")
print(f"tr_times[5] (first positive): {ds.tr_times[5]}") """


""" import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for gandalf
import matplotlib.pyplot as plt

DATA_PATH = '/scratch/users/s214/lab3'

# load one story fMRI
fmri = np.load(f'{DATA_PATH}/subject2/adollshouse.npy')

# pick a few voxels to plot — use mean across voxels to see overall signal
mean_signal = fmri.mean(axis=1)  # average across all voxels, shape (241,)

# mark the trim boundaries
trim_start = 5
trim_end = 10

fig, ax = plt.subplots(figsize=(12, 4))

# plot full signal
ax.plot(mean_signal, color='gray', alpha=0.5, label='full signal')

# highlight first 5 TRs in red
ax.plot(range(trim_start), mean_signal[:trim_start], 
        color='red', linewidth=2, label=f'first {trim_start} TRs (trimmed)')

# highlight last 10 TRs in red
ax.plot(range(len(mean_signal)-trim_end, len(mean_signal)), 
        mean_signal[-trim_end:],
        color='orange', linewidth=2, label=f'last {trim_end} TRs (trimmed)')

# highlight kept region in blue
ax.plot(range(trim_start, len(mean_signal)-trim_end), 
        mean_signal[trim_start:-trim_end],
        color='blue', linewidth=2, label='kept region')

ax.axvline(x=trim_start, color='red', linestyle='--', alpha=0.7)
ax.axvline(x=len(mean_signal)-trim_end, color='orange', linestyle='--', alpha=0.7)

ax.set_xlabel('TR (timepoint)')
ax.set_ylabel('mean BOLD signal across voxels')
ax.set_title('adollshouse — fMRI signal showing trimmed boundary TRs')
ax.legend()

plt.tight_layout()
plt.savefig('figs/fmri_trim_visualization.png', dpi=150)
print("saved to figs/fmri_trim_visualization.png")

# also print the actual values so we can see numerically
print(f"\nFirst 5 TR values (trimmed): {mean_signal[:5]}")
print(f"Next 5 TR values (kept):     {mean_signal[5:10]}")
print(f"Last 10 TR values (trimmed): {mean_signal[-10:]}")
print(f"Prev 5 TR values (kept):     {mean_signal[-15:-10]}") """


import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('code')

DATA_PATH = '/scratch/users/s214/lab3'

with open(f'{DATA_PATH}/raw_text.pkl', 'rb') as f:
    raw_text = pickle.load(f)

# check multiple stories to see if boundary noise is consistent
stories_to_check = ['adollshouse', 'avatar', 'sweetaspie', 'life', 'tetris']
trim_start = 5
trim_end = 10

fig, axes = plt.subplots(len(stories_to_check), 1, figsize=(14, 3 * len(stories_to_check)))

print("story | mean first 5 TRs | mean kept region | mean last 10 TRs | start outlier? | end outlier?")
print("-" * 95)

for i, story in enumerate(stories_to_check):
    fmri = np.load(f'{DATA_PATH}/subject2/{story}.npy')
    mean_signal = fmri.mean(axis=1)

    first_5 = mean_signal[:trim_start].mean()
    last_10 = mean_signal[-trim_end:].mean()
    kept = mean_signal[trim_start:-trim_end].mean()
    kept_std = mean_signal[trim_start:-trim_end].std()

    # is boundary more than 2 std from kept region mean?
    start_outlier = abs(first_5 - kept) > 2 * kept_std
    end_outlier = abs(last_10 - kept) > 2 * kept_std

    print(f"{story:15} | {first_5:16.4f} | {kept:16.4f} | {last_10:16.4f} | {str(start_outlier):14} | {str(end_outlier)}")

    ax = axes[i]
    ax.plot(mean_signal, color='gray', alpha=0.4)
    ax.plot(range(trim_start), mean_signal[:trim_start], color='red', linewidth=2)
    ax.plot(range(len(mean_signal)-trim_end, len(mean_signal)), mean_signal[-trim_end:], color='orange', linewidth=2)
    ax.plot(range(trim_start, len(mean_signal)-trim_end), mean_signal[trim_start:-trim_end], color='blue', linewidth=2)
    ax.axvline(x=trim_start, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=len(mean_signal)-trim_end, color='orange', linestyle='--', alpha=0.5)
    ax.set_title(story)
    ax.set_ylabel('mean BOLD')
    ax.set_xlabel('TR')

# add legend to first plot only
axes[0].plot([], [], color='red', label='first 5 TRs (trimmed)')
axes[0].plot([], [], color='orange', label='last 10 TRs (trimmed)')
axes[0].plot([], [], color='blue', label='kept region')
axes[0].legend(loc='upper right')

plt.suptitle('fMRI boundary TR signal across 5 stories — subject2', fontsize=13)
plt.tight_layout()
plt.savefig('figs/fmri_trim_all_stories.png', dpi=150)
print("\nsaved to figs/fmri_trim_all_stories.png")