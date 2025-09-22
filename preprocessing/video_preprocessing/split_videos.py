"""
SPLIT BLOCK-LEVEL MP4 INTO CLIP-LEVEL MP4 (GT-LABEL DRIVEN)
------------------------------------------------------------
Input:
  raw/Video/BlockY_full.mp4 or Xth_10min.mp4
  core_files/gt_label.py   (true class order per block)

Process:
  - Each block = 40 classes × 5 clips = 200 clips
  - Each class has: 3s hint (72 frames) + 5×2s clips (5×48 frames)
  - GT_LABEL[block, order] gives true class ID (0–39) for slot
  - Extract exact 2s (48 frames) per clip
  - Save as MP4 with classYY_clipZZ.mp4 naming

Output:
  processed/Video_mp4/BlockY/classYY_clipZZ.mp4
"""

import os
import sys
import cv2
from tqdm import tqdm
import re

# paths
repo_root = "/content/EEG2Video_my_version"
in_dir = "/content/drive/MyDrive/EEG2Video_data/raw/Video/"
out_dir = "/content/drive/MyDrive/EEG2Video_data/processed/Video_mp4/"

# import GT_LABEL
sys.path.append(os.path.join(repo_root, "core_files"))
from gt_label import GT_LABEL   # shape (7,40), values 0–39

os.makedirs(out_dir, exist_ok=True)

# parameters
fps = 24
clip_len = 2 * fps   # 48 frames
hint_len = 3 * fps   # 72 frames
clips_per_class = 5

# list all block videos
all_files = [f for f in os.listdir(in_dir) if f.endswith(".mp4")]
print("Available block videos:")
for i, f in enumerate(all_files):
    print(f"{i}: {f}")

chosen = input("Enter indices of block videos to process (comma separated, 'all' for all): ").strip()
if chosen.lower() == "all":
    selected_files = all_files
else:
    idxs = [int(x) for x in chosen.split(",")]
    selected_files = [all_files[i] for i in idxs]

# process
for fname in selected_files:
    block_name = os.path.splitext(fname)[0]

    # extract the first number in the name (works for 1st_10min, Block1_full, etc.)
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
        true_class = GT_LABEL[block_id, order_idx]  # 0–39
        slot_start = order_idx * (hint_len + clips_per_class * clip_len)

        for clip_id in range(clips_per_class):
            start_idx = slot_start + hint_len + clip_id * clip_len
            end_idx = start_idx + clip_len

            if end_idx > total_frames:
                print(f"Warning: Block{block_id+1} class{true_class} clip{clip_id+1} exceeds video length")
                continue

            out_path = os.path.join(
                block_out_dir,
                f"class{true_class:02d}_clip{clip_id+1:02d}.mp4"
            )
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (512, 288))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

            for fidx in range(clip_len):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (512, 288), interpolation=cv2.INTER_LINEAR)
                writer.write(frame)

            writer.release()
            saved_count += 1

    cap.release()
    print(f"Finished splitting {fname} → {block_out_dir}, saved {saved_count} clips")
