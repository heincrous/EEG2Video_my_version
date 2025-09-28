# ==========================================
# SPLIT BLOCK-LEVEL MP4 INTO CLIP-LEVEL MP4
# ==========================================
# Input:
#   EEG2Video_data/raw/Video/BlockY_full.mp4 or Xth_10min.mp4
#   core_files/gt_label.py   (true class order per block)
#
# Process:
#   - Each block = 40 classes × 5 clips = 200 clips
#   - Each class has: 3 s hint (72 frames) + 5 × 2 s clips (5 × 48 frames)
#   - GT_LABEL[block, order] gives true class ID (0–39) for slot
#   - Extract exact 2 s (48 frames) per clip
#   - Save as MP4 with classYY_clipZZ.mp4 naming
#
# Output:
#   EEG2Video_data/processed/Video_mp4/BlockY/classYY_clipZZ.mp4
# ==========================================

# === Standard libraries ===
import os
import re

# === Third-party libraries ===
import cv2
from tqdm import tqdm

# === Repo imports ===
from core.gt_label import GT_LABEL   # shape (7,40), values 0–39


# ==========================================
# CONFIGURATION (EDITABLE PARAMETERS)
# ==========================================
config = {
    "fps":             24,
    "clip_seconds":    2,
    "hint_seconds":    3,
    "clips_per_class": 5,
    "resize_w":        512,
    "resize_h":        288,

    "drive_root": "/content/drive/MyDrive/EEG2Video_data",
}

# Derived values
config["clip_len"] = config["clip_seconds"] * config["fps"]  # 48 frames
config["hint_len"] = config["hint_seconds"] * config["fps"]  # 72 frames

# Paths
in_dir  = os.path.join(config["drive_root"], "raw", "Video")
out_dir = os.path.join(config["drive_root"], "processed", "Video_mp4")
os.makedirs(out_dir, exist_ok=True)


# ==========================================
# Helper: list video files
# ==========================================
def list_video_files(directory):
    return sorted([f for f in os.listdir(directory) if f.endswith(".mp4")])


# ==========================================
# Main processing loop
# ==========================================
all_files = list_video_files(in_dir)
print("Available block videos:")
for i, f in enumerate(all_files):
    print(f"[{i}] {f}")

chosen = input("Enter indices of block videos to process (comma separated, 'all' for all): ").strip()
if chosen.lower() == "all":
    selected_files = all_files
else:
    idxs = [int(x) for x in chosen.split(",")]
    selected_files = [all_files[i] for i in idxs]

for fname in selected_files:
    block_name = os.path.splitext(fname)[0]

    # extract block number (works for 1st_10min, Block1_full, etc.)
    match = re.search(r"(\d+)", block_name)
    if not match:
        raise ValueError(f"Could not parse block number from {block_name}")
    block_id = int(match.group(1)) - 1  # zero-based 0–6

    block_out_dir = os.path.join(out_dir, f"Block{block_id+1}")
    os.makedirs(block_out_dir, exist_ok=True)

    video_path = os.path.join(in_dir, fname)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    saved_count = 0

    for order_idx in tqdm(range(40), desc=f"Processing {fname}"):
        true_class = GT_LABEL[block_id, order_idx]
        slot_start = order_idx * (config["hint_len"] + config["clips_per_class"] * config["clip_len"])

        for clip_id in range(config["clips_per_class"]):
            start_idx = slot_start + config["hint_len"] + clip_id * config["clip_len"]
            end_idx   = start_idx + config["clip_len"]

            if end_idx > total_frames:
                print(f"Warning: Block{block_id+1} class{true_class} clip{clip_id+1} exceeds video length")
                continue

            out_path = os.path.join(block_out_dir, f"class{true_class:02d}_clip{clip_id+1:02d}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, config["fps"], (config["resize_w"], config["resize_h"]))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

            for fidx in range(config["clip_len"]):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (config["resize_w"], config["resize_h"]), interpolation=cv2.INTER_LINEAR)
                writer.write(frame)

            writer.release()
            saved_count += 1

    cap.release()
    print(f"Finished splitting {fname} → {block_out_dir}, saved {saved_count} clips")

print("\nProcessing complete.")
