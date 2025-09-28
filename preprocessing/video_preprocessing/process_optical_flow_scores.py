# ==========================================
# Process Optical Flow Scores
# ==========================================
import os
import numpy as np

from core.gt_label import GT_LABEL   # shape (7,40), values 0–39

# paths
drive_root   = "/content/drive/MyDrive/EEG2Video_data"
raw_dir      = os.path.join(drive_root, "raw", "meta-info")
processed_dir= os.path.join(drive_root, "processed", "meta-info")
os.makedirs(processed_dir, exist_ok=True)

scores_path  = os.path.join(raw_dir, "All_video_optical_flow_score.npy")
save_path    = os.path.join(processed_dir, "All_video_optical_flow_score_byclass.npy")

# load
scores = np.load(scores_path)   # shape (7,200)
print("Scores shape:", scores.shape)

# reorganize into [blocks, classes, 5 clips]
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

# save
np.save(save_path, organized)
print(f"Saved organized scores to {save_path}")
print("\nProcessing complete.")
