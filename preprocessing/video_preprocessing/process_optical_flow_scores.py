# ==========================================
# PROCESS OPTICAL FLOW SCORES
# ==========================================
# Input:
#   EEG2Video_data/raw/meta-info/All_video_optical_flow_score.npy
#   core_files/gt_label.py   (true class order per block)
#
# Process:
#   - Raw file shape: (7, 200) = 7 blocks × 40 classes × 5 clips
#   - Each class appears once per order in GT_LABEL[block, order]
#   - Rearrange scores → [blocks, classes, 5 clips]
#   - Save organized array for EEG-VP benchmark
#
# Output:
#   EEG2Video_data/processed/meta-info/All_video_optical_flow_score_byclass.npy
# ==========================================

import os
import numpy as np

from core.gt_label import GT_LABEL   # shape (7,40), values 0–39


# ==========================================
# CONFIG
# ==========================================
config = {
    "drive_root": "/content/drive/MyDrive/EEG2Video_data",
}

raw_dir       = os.path.join(config["drive_root"], "raw", "meta-info")
processed_dir = os.path.join(config["drive_root"], "processed", "meta-info")
os.makedirs(processed_dir, exist_ok=True)

scores_path = os.path.join(raw_dir, "All_video_optical_flow_score.npy")
save_path   = os.path.join(processed_dir, "All_video_optical_flow_score_byclass.npy")


# ==========================================
# LOAD RAW SCORES
# ==========================================
scores = np.load(scores_path)   # shape (7,200)
print("Loaded scores shape:", scores.shape)


# ==========================================
# REORGANIZE SCORES
# ==========================================
num_blocks, num_slots = scores.shape
num_classes           = 40
clips_per_class       = 5

organized = np.zeros((num_blocks, num_classes, clips_per_class), dtype=np.float32)

for block_id in range(num_blocks):
    for order_idx in range(num_classes):
        cls = GT_LABEL[block_id, order_idx]
        start = order_idx * clips_per_class
        end   = start + clips_per_class
        organized[block_id, cls, :] = scores[block_id, start:end]

print("Organized shape:", organized.shape)  # (7,40,5)
print("Example block0, class0 scores:", organized[0, 0])


# ==========================================
# SAVE
# ==========================================
np.save(save_path, organized)
print(f"Saved organized scores to {save_path}")
print("\nProcessing complete.")
