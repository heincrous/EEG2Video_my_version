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

# paths
in_dir = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments/"
out_dir = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_windows/"
os.makedirs(out_dir, exist_ok=True)

# walk through all subject folders
for subj in os.listdir(in_dir):
    subj_path = os.path.join(in_dir, subj)
    if not os.path.isdir(subj_path):
        continue

    for block in os.listdir(subj_path):
        block_path = os.path.join(subj_path, block)
        out_block = os.path.join(out_dir, subj, block)
        os.makedirs(out_block, exist_ok=True)

        for fname in tqdm(os.listdir(block_path), desc=f"{subj} {block}"):
            if not fname.endswith(".npy"):
                continue

            seg = np.load(os.path.join(block_path, fname))  # [400,62]

            # sliding windows
            windows = []
            for start in range(0, 400 - window_size + 1, step):
                win = seg[start:start+window_size, :].T  # [62,100]
                windows.append(win)
            windows = np.stack(windows, axis=0)  # [7,62,100]

            # save
            save_path = os.path.join(out_block, fname)
            np.save(save_path, windows)