"""
SPLIT BLOCK-LEVEL MP4 INTO CLIP-LEVEL MP4
-----------------------------------------
Input:
  raw/Video/BlockY_full.mp4
    ~10 min recording per block

Process:
  - Each block contains 40 classes × 5 clips.
  - Skip 3s hint before each class.
  - Extract 5 × 2s video clips per class (24 fps).
  - Save each clip as its own MP4.

Output:
  processed/Video_mp4/BlockY/classYY_clipZZ.mp4
    2s duration, 24 fps
"""

import os
import cv2
from tqdm import tqdm

# paths
in_dir = "/content/drive/MyDrive/EEG2Video_data/raw/Video/"
out_dir = "/content/drive/MyDrive/EEG2Video_data/processed/Video_mp4/"
os.makedirs(out_dir, exist_ok=True)

# video parameters
fps = 24
clip_len = 2 * fps   # 2 seconds = 48 frames
hint_len = 3 * fps   # 3 seconds = 72 frames

# list all block videos
all_files = [f for f in os.listdir(in_dir) if f.endswith(".mp4")]
print("Available block videos:")
for i, f in enumerate(all_files):
    print(f"{i}: {f}")

# user chooses
chosen = input("Enter indices of block videos to process (comma separated, 'all' for all): ").strip()
if chosen.lower() == "all":
    selected_files = all_files
else:
    idxs = [int(x) for x in chosen.split(",")]
    selected_files = [all_files[i] for i in idxs]

# process selected block videos
for fname in selected_files:
    block_name = os.path.splitext(fname)[0]   # e.g. "Block1_full"
    block_id = ''.join([c for c in block_name if c.isdigit()])  # extract block number

    block_out_dir = os.path.join(out_dir, f"Block{block_id}")
    os.makedirs(block_out_dir, exist_ok=True)

    video_path = os.path.join(in_dir, fname)
    cap = cv2.VideoCapture(video_path)

    frame_idx = 0
    class_id = 0

    # loop over 40 classes
    for class_id in tqdm(range(40), desc=f"Processing {fname}"):
        # skip 3s hint
        for _ in range(hint_len):
            ret, _ = cap.read()
            frame_idx += 1
            if not ret:
                break

        # extract 5 × 2s clips
        for clip_id in range(5):
            out_path = os.path.join(block_out_dir, f"class{class_id:02d}_clip{clip_id+1:02d}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (512, 288))

            for _ in range(clip_len):
                ret, frame = cap.read()
                frame_idx += 1
                if not ret:
                    break
                frame = cv2.resize(frame, (512, 288), interpolation=cv2.INTER_LINEAR)
                writer.write(frame)

            writer.release()

    cap.release()
    print(f"Finished splitting {fname} → {block_out_dir}")