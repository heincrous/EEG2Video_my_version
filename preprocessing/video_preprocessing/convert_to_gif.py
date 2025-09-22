"""
CREATE GIFS FROM MP4 CLIPS
---------------------------
Input:
  processed/Video_mp4/BlockY/classYY_clipZZ.mp4

Process:
  - Convert to GIF format
  - Maintain 24 fps

Output:
  processed/Video_gif/BlockY/classYY_clipZZ.gif
"""

# This script is mainly for visualization and debugging

import os
import imageio
import cv2
from tqdm import tqdm

# paths
in_dir = "/content/drive/MyDrive/EEG2Video_data/processed/Video_mp4/"
out_dir = "/content/drive/MyDrive/EEG2Video_data/processed/Video_gif/"
os.makedirs(out_dir, exist_ok=True)

fps = 24

# walk through all blocks
for block in os.listdir(in_dir):
    block_path = os.path.join(in_dir, block)
    if not os.path.isdir(block_path):
        continue

    out_block = os.path.join(out_dir, block)
    os.makedirs(out_block, exist_ok=True)

    for fname in tqdm(os.listdir(block_path), desc=f"Converting {block}"):
        if not fname.endswith(".mp4"):
            continue

        video_path = os.path.join(block_path, fname)
        gif_path = os.path.join(out_block, fname.replace(".mp4", ".gif"))

        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # cv2 uses BGR
            frames.append(frame)

        cap.release()

        # write GIF at 24 fps
        imageio.mimsave(gif_path, frames, fps=fps)
