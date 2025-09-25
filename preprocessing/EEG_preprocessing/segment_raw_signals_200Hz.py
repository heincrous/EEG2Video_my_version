"""
SEGMENT EEG INTO CLIP-ALIGNED SEGMENTS (SUBJECT-LEVEL ARRAYS)
-------------------------------------------------------------
Input:
  raw/EEG/subX.npy, subX_session2.npy ...
  Sampling rate = 200 Hz
  Channels      = 62
  Block length  = 40 classes × (3 s hint + 5×2 s clips) = 520 s
  Each block per channel = 104000 samples

Process:
  - Use GT_LABEL (7×40) to align EEG with video class order per block.
  - For each class slot: skip 3 s hint (600 samples),
    then extract 5×2 s clips (5×400 samples).
  - Collect into subject-level array shaped (7,40,5,400,62).

Output:
  processed/EEG_segments/subX.npy
    Shape = [7,40,5,400,62]
"""

import os
import sys
import numpy as np
from tqdm import tqdm

# parameters
fre = 200
segment_len = 2 * fre   # 400 samples (2 s)
hint_len    = 3 * fre   # 600 samples (3 s)
channels = 62
clips_per_class = 5

# paths
raw_dir = "/content/drive/MyDrive/EEG2Video_data/raw/EEG/"
out_dir = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments/"
os.makedirs(out_dir, exist_ok=True)

# import GT_LABEL
repo_root = "/content/EEG2Video_my_version"
sys.path.append(os.path.join(repo_root, "core_files"))
from gt_label import GT_LABEL   # shape (7,40), values 0–39

# list subject files
all_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".npy")])
print("Available subject files:")
for i, f in enumerate(all_files):
    print(f"{i}: {f}")

# user chooses subjects
chosen = input("Enter indices of subjects to process (comma separated, 'all' for all): ").strip()
if chosen.lower() == "all":
    selected_files = all_files
else:
    idxs = [int(x) for x in chosen.split(",")]
    selected_files = [all_files[i] for i in idxs]

# process each subject
for subj_file in selected_files:
    subj_name = subj_file.replace(".npy", "")
    eeg_data = np.load(os.path.join(raw_dir, subj_file))  # [7,62,104000]

    subj_array = np.zeros((7, 40, 5, segment_len, channels), dtype=np.float32)

    for block_id in range(7):
        now_data = eeg_data[block_id]  # [62,T_block], T_block ≈ 104000
        l = 0

        for order_idx in tqdm(range(40), desc=f"{subj_name} Block {block_id+1}"):
            true_class = GT_LABEL[block_id, order_idx]

            # skip 3 s hint
            l += hint_len

            for clip_id in range(clips_per_class):
                start_idx = l
                end_idx   = l + segment_len

                if end_idx > now_data.shape[1]:
                    raise ValueError(
                        f"{subj_name} Block {block_id+1}: insufficient samples "
                        f"for class {true_class}, clip {clip_id}. "
                        f"Available {now_data.shape[1]}, needed {end_idx}"
                    )

                eeg_slice = now_data[:, start_idx:end_idx]  # [62,400]
                eeg_slice = eeg_slice.T  # → [400,62]

                subj_array[block_id, true_class, clip_id] = eeg_slice
                l = end_idx

        expected_len = (hint_len + clips_per_class*segment_len) * 40
        if l != expected_len:
            print(
                f"Warning: {subj_name} Block {block_id+1} ended at {l}, "
                f"expected {expected_len}."
            )

    # ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{subj_name}.npy")
    np.save(out_path, subj_array)
    print(f"Saved {subj_name} → {out_path}, shape {subj_array.shape}")
