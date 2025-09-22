"""
CREATE EEG WINDOWS FOR SEQ2SEQ INPUT
------------------------------------
Input:
  processed/EEG_segments/subX/BlockY/classYY_clipZZ.npy
    Shape = [400,62]

Process:
  - Apply sliding window:
      window_size = 100
      overlap     = 50
  - From 400 samples → 7 windows.
  - Rearrange into [7,62,100].

Output:
  processed/EEG_windows/BlockY/classYY_clipZZ.npy
    Shape = [7,62,100]
"""

# Pseudocode steps:
# 1. Load [400,62] EEG segment
# 2. Create sliding windows with numpy slicing
# 3. Stack windows → [7,62,100]
# 4. Save into EEG_windows folder

import os
import numpy as np
from tqdm import tqdm

# parameters
window_size = 100
overlap = 50
step = window_size - overlap  # 50
num_windows = (400 - window_size) // step + 1  # 7
batch_size = 64  # adjust based on RAM

# paths
in_dir = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments/"
out_dir = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_windows/"
os.makedirs(out_dir, exist_ok=True)

def make_windows(seg):
    windows = []
    for start in range(0, 400 - window_size + 1, step):
        win = seg[start:start+window_size, :].T  # [62,100]
        windows.append(win)
    return np.stack(windows, axis=0)  # [7,62,100]

# collect available subjects
subjects = [s for s in sorted(os.listdir(in_dir)) if os.path.isdir(os.path.join(in_dir, s))]

# display subject options in "0: subX" format
print("\nAvailable subjects to process:")
for idx, subj in enumerate(subjects):
    print(f"{idx}: {subj}")

# ask user
choices = input("\nEnter subject indices to process (comma separated): ")
choices = [int(c.strip()) for c in choices.split(",") if c.strip().isdigit()]

# process selected subjects
for idx in choices:
    subj = subjects[idx]
    subj_path = os.path.join(in_dir, subj)

    for block in sorted(os.listdir(subj_path)):
        block_path = os.path.join(subj_path, block)
        if not os.path.isdir(block_path):
            continue

        out_block = os.path.join(out_dir, subj, block)
        os.makedirs(out_block, exist_ok=True)

        fnames = [f for f in os.listdir(block_path) if f.endswith(".npy")]
        for i in tqdm(range(0, len(fnames), batch_size), desc=f"{subj} {block}"):
            batch = fnames[i:i+batch_size]

            segs = [np.load(os.path.join(block_path, f)) for f in batch]
            results = [make_windows(seg) for seg in segs]

            for fname, windows in zip(batch, results):
                save_path = os.path.join(out_block, fname)
                np.save(save_path, windows)

print("\nProcessing complete.")

