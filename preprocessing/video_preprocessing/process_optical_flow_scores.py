# ==========================================
# Organize Optical Flow Scores by GT Label
# ==========================================
import os
import numpy as np

# === Repo imports ===
from core.gt_label import GT_LABEL   # shape (7,40), values 0–39

# === Path ===
drive_root = "/content/drive/MyDrive/EEG2Video_data"
scores_path = os.path.join(drive_root, "raw", "meta-info", "All_video_optical_flow_score.npy")

# === Load ===
scores = np.load(scores_path)   # shape (7,200)
print("Scores shape:", scores.shape)

# === Reorganize into [blocks, classes, 5 clips] ===
num_blocks, num_slots = scores.shape
clips_per_class = 5
num_classes = 40

organized = np.zeros((num_blocks, num_classes, clips_per_class), dtype=np.float32)

for block_id in range(num_blocks):
    for order_idx in range(num_classes):
        cls = GT_LABEL[block_id, order_idx]
        # each order has exactly 5 clips
        start = order_idx * clips_per_class
        end   = start + clips_per_class
        organized[block_id, cls, :] = scores[block_id, start:end]

print("Organized shape:", organized.shape)  # (7,40,5)
print("Example block0, class0 scores:", organized[0,0])
