"""
CREATE EEG WINDOWS FOR SEQ2SEQ INPUT (SUBJECT-LEVEL ARRAYS)
-----------------------------------------------------------
Input:
  processed/EEG_segments/subX.npy
    Shape = [7,40,5,400,62]

Process:
  - Apply sliding window (size=100, overlap=50).
  - Each 2s clip [400,62] → [7,62,100].
  - Preserve subject-level structure.

Output:
  processed/EEG_windows/subX.npy
    Shape = [7,40,5,7,62,100]
"""

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

def make_windows(seg):
    """
    seg: [400,62]
    return: [7,62,100]
    """
    windows = []
    for start in range(0, 400 - window_size + 1, step):
        win = seg[start:start+window_size, :].T  # [62,100]
        windows.append(win)
    return np.stack(windows, axis=0)  # [7,62,100]

# collect available subjects
subjects = [f for f in sorted(os.listdir(in_dir)) if f.endswith(".npy")]

print("\nAvailable subjects to process:")
for idx, subj in enumerate(subjects):
    print(f"{idx}: {subj}")

# ask user
choices = input("\nEnter subject indices to process (comma separated): ")
choices = [int(c.strip()) for c in choices.split(",") if c.strip().isdigit()]

# process selected subjects
for idx in choices:
    subj_file = subjects[idx]
    subj_name = subj_file.replace(".npy", "")
    subj_path = os.path.join(in_dir, subj_file)

    print(f"\nProcessing {subj_name}...")

    data = np.load(subj_path)  # [7,40,5,400,62]
    out_array = np.zeros((7,40,5,num_windows,62,100), dtype=np.float32)

    for b in range(7):
        for c in tqdm(range(40), desc=f"{subj_name} Block{b+1}"):
            for k in range(5):
                seg = data[b,c,k]  # [400,62]
                windows = make_windows(seg)  # [7,62,100]
                out_array[b,c,k] = windows

    out_path = os.path.join(out_dir, f"{subj_name}.npy")
    np.save(out_path, out_array)
    print(f"Saved {subj_name} → {out_path}, shape {out_array.shape}")

print("\nProcessing complete.")
