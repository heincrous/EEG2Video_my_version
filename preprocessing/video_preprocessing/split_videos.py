"""
SPLIT BLOCK-LEVEL MP4 INTO CLIP-LEVEL MP4 (GT-LABEL DRIVEN)
------------------------------------------------------------
Input:
  raw/Video/BlockY_full.mp4
  core_files/gt_label.npy   (ground truth alignment)

Process:
  - Load GT_LABEL[block, class, clip] → frame index of clip start
  - Skip 3s hint frames (encoded in GT_LABEL offsets)
  - Extract exact 2s (48 frames at 24 fps) per clip
  - Save each as its own MP4

Output:
  processed/Video_mp4/BlockY/classYY_clipZZ.mp4
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

# paths
repo_root = "/content/EEG2Video_my_version"
in_dir = "/content/drive/MyDrive/EEG2Video_data/raw/Video/"
out_dir = "/content/drive/MyDrive/EEG2Video_data/processed/Video_mp4/"
gt_label_path = os.path.join(repo_root, "core_files", "gt_label.npy")

os.makedirs(out_dir, exist_ok=True)

# parameters
fps = 24
clip_len = 2 * fps   # 48 frames

# load GT_LABEL
GT_LABEL = np.load(gt_label_path)   # shape [7,40,5], entries = frame start indices

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
    block_name = os.path.splitext(fname)[0]   # e.g. Block1_full
    block_id = int(''.join([c for c in block_name if c.isdigit()])) - 1  # zero-index

    block_out_dir = os.path.join(out_dir, f"Block{block_id+1}")
    os.makedirs(block_out_dir, exist_ok=True)

    video_path = os.path.join(in_dir, fname)
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for class_id in tqdm(range(40), desc=f"Processing {fname}"):
        for clip_id in range(5):
            start_idx = GT_LABEL[block_id, class_id, clip_id]
            end_idx = start_idx + clip_len

            # safety check
            if end_idx > total_frames:
                print(f"Warning: clip {class_id}_{clip_id} exceeds video length in {fname}")
                continue

            out_path = os.path.join(
                block_out_dir,
                f"class{class_id:02d}_clip{clip_id+1:02d}.mp4"
            )
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (512, 288))

            # jump to start index
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

            for fidx in range(clip_len):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (512, 288), interpolation=cv2.INTER_LINEAR)
                writer.write(frame)

            writer.release()

    cap.release()
    print(f"Finished splitting {fname} → {block_out_dir}")
