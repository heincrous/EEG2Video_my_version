"""
SEGMENT RAW EEG INTO CLIP-ALIGNED SEGMENTS
------------------------------------------
Input:
  raw/EEG/subX.npy, subX_session2.npy ...
  Sampling rate = 200 Hz
  Channels      = 62
  Video clip length = 2 s → 400 samples

Process:
  - Use GT_LABEL (7x40) to align EEG stream with video class order per block.
  - Each block has 40 classes × 5 clips = 200 clips.
  - For each video clip, cut out the corresponding [400,62] EEG slice.
  - Store under Block and true class folder.

Output:
  processed/EEG_segments/subX/BlockY/classYY_clipZZ.npy
    Shape = [400,62]
"""

import os
import sys
import numpy as np
from tqdm import tqdm

fre = 200
segment_len = 2 * fre  # 400 samples
channels = 62
clips_per_class = 5

# paths
raw_dir = "/content/drive/MyDrive/EEG2Video_data/raw/EEG/"
out_dir = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments/"

# import GT_LABEL directly from core_files/gt_label.py
repo_root = "/content/EEG2Video_my_version"
sys.path.append(os.path.join(repo_root, "core_files"))
from gt_label import GT_LABEL   # GT_LABEL shape (7,40), values 0–39

# list subject files
all_files = [f for f in os.listdir(raw_dir) if f.endswith(".npy")]
print("Available subject files:")
for i, f in enumerate(all_files):
    print(f"{i}: {f}")

# user chooses which subjects
chosen = input("Enter indices of subjects to process (comma separated, 'all' for all): ").strip()
if chosen.lower() == "all":
    selected_files = all_files
else:
    idxs = [int(x) for x in chosen.split(",")]
    selected_files = [all_files[i] for i in idxs]

# process each subject
for subj_file in selected_files:
    subj_name = subj_file.replace(".npy", "")
    eeg_data = np.load(os.path.join(raw_dir, subj_file))  # shape [7,62,T]

    for block_id in range(7):
        now_data = eeg_data[block_id]  # [62,T]

        for order_idx in tqdm(range(40), desc=f"{subj_name} Block {block_id+1}"):
            true_class = GT_LABEL[block_id, order_idx]  # 0–39

            for clip_id in range(clips_per_class):
                # flatten index within block: 0..199
                flat_idx = order_idx * clips_per_class + clip_id
                start_idx = flat_idx * segment_len
                end_idx = start_idx + segment_len

                eeg_slice = now_data[:, start_idx:end_idx]  # [62,400]
                eeg_slice = eeg_slice.T  # → [400,62]

                out_path = os.path.join(
                    out_dir,
                    subj_name,
                    f"Block{block_id+1}",
                    f"class{true_class:02d}_clip{clip_id+1:02d}.npy"
                )
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.save(out_path, eeg_slice)

    print(f"Finished {subj_name}")
