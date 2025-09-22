"""
SEGMENT RAW EEG INTO CLIP-ALIGNED SEGMENTS
------------------------------------------
Input:
  raw/EEG/subX.npy, subX_session2.npy ...
  Sampling rate = 200 Hz
  Channels      = 62
  Video clip length = 2 s → 400 samples

Process:
  - Use GT_LABEL to align EEG stream with video clip order.
  - For each video clip, cut out the corresponding [400,62] EEG slice.
  - Ensure each slice is stored under the correct Block and Class.

Output:
  processed/EEG_segments/subX/BlockY/classYY_clipZZ.npy
    Shape = [400,62]
"""

# Pseudocode steps:
# 1. Load raw EEG file (subX.npy or subX_session2.npy)
# 2. For each video clip in GT_LABEL:
#     a. Find start index in EEG stream
#     b. Slice 400 samples × 62 channels
#     c. Verify shape = [400,62]
#     d. Save as processed/EEG_segments/subX/BlockY/classYY_clipZZ.npy
#3. Repeat for all clips and subjects

import os
import numpy as np
from tqdm import tqdm

fre = 200
segment_len = 2 * fre  # 400 samples
channels = 62

# paths
raw_dir = "/content/drive/MyDrive/EEG2Video_data/raw/EEG/"
out_dir = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments/"

# load GT_LABEL (shape [7,40,5]) – zero-indexed
repo_root = "/content/EEG2Video_my_version"
gt_label_path = os.path.join(repo_root, "core_files", "gt_label.npy")
GT_LABEL = np.load(gt_label_path)

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
    os.makedirs(out_dir, exist_ok=True)

    eeg_data = np.load(os.path.join(raw_dir, subj_file))  # shape [7,62,T]

    # container for subject: [7,40,5,62,400]
    save_data = np.empty((0, 40, 5, channels, segment_len))

    for block_id in range(7):
        now_data = eeg_data[block_id]  # [62,T]
        block_data = np.empty((0, 5, channels, segment_len))

        for class_id in tqdm(range(40), desc=f"{subj_name} Block {block_id+1}"):
            class_data = np.empty((0, channels, segment_len))
            for clip_id in range(5):
                start_idx = GT_LABEL[block_id, class_id, clip_id]
                end_idx = start_idx + segment_len

                eeg_slice = now_data[:, start_idx:end_idx]          # [62,400]
                eeg_slice = eeg_slice.reshape(1, channels, segment_len)
                class_data = np.concatenate((class_data, eeg_slice))
            block_data = np.concatenate((block_data, class_data.reshape(1, 5, channels, segment_len)))

        save_data = np.concatenate((save_data, block_data.reshape(1, 40, 5, channels, segment_len)))

    # save subject-level .npy
    save_path = os.path.join(out_dir, subj_name + ".npy")
    np.save(save_path, save_data)
    print(f"Saved {save_path}, shape={save_data.shape}")
