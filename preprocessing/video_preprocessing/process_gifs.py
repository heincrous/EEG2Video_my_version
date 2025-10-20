# ==========================================
# CONVERT CLIP-LEVEL MP4 INTO GIFS
# ==========================================
# Input:
#   EEG2Video_data/processed/Video_mp4/BlockY/classYY_clipZZ.mp4
#
# Process:
#   - Each MP4 clip (2 s) converted into a 6-frame GIF
#   - 6 frames are sampled evenly across each clip
#   - GIFs saved at 3 frames per second (≈ 2 s duration)
#
# Output:
#   EEG2Video_data/processed/Video_gif/BlockY/classYY_clipZZ.gif
# ==========================================

import os
import cv2
import torch
import numpy as np
import torchvision
import imageio
from einops import rearrange
from tqdm import tqdm


# ==========================================
# CONFIG
# ==========================================
config = {
    "fps":          3,        # GIF frame rate
    "frames_out":   6,        # number of frames per GIF
    "resize_w":     512,
    "resize_h":     288,
    "drive_root":   "/content/drive/MyDrive/EEG2Video_data",
}

# paths
in_root  = os.path.join(config["drive_root"], "processed", "Video_mp4")
out_root = os.path.join(config["drive_root"], "processed", "Video_gif")
os.makedirs(out_root, exist_ok=True)


# ==========================================
# Helper: sample frames from video
# ==========================================
def sample_frames(video_path, n_frames, resize_w, resize_h):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, n_frames, dtype=int)

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)

    cap.release()
    return torch.tensor(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0  # [T,3,H,W]


# ==========================================
# Helper: save GIF grid
# ==========================================
def save_gif_grid(video_tensor, path, fps=3, rescale=False, n_rows=1):
    # video_tensor: [T,3,H,W]
    frames = rearrange(video_tensor, "t c h w -> t 1 c h w")  # add batch dim
    outputs = []

    for x in frames:
        grid = torchvision.utils.make_grid(x, nrow=n_rows)
        grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            grid = (grid + 1.0) / 2.0
        frame = (grid * 255).cpu().numpy().astype(np.uint8)
        outputs.append(frame)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


# ==========================================
# Main processing loop
# ==========================================
for block in sorted(os.listdir(in_root)):
    block_in = os.path.join(in_root, block)
    if not os.path.isdir(block_in):
        continue

    block_out = os.path.join(out_root, block)
    os.makedirs(block_out, exist_ok=True)

    mp4_files = [f for f in os.listdir(block_in) if f.endswith(".mp4")]

    for mp4_file in tqdm(mp4_files, desc=f"Converting {block}"):
        mp4_path = os.path.join(block_in, mp4_file)
        gif_path = os.path.join(block_out, mp4_file.replace(".mp4", ".gif"))

        video_tensor = sample_frames(mp4_path, config["frames_out"], config["resize_w"], config["resize_h"])
        if video_tensor.shape[0] == 0:
            continue

        save_gif_grid(video_tensor, gif_path, fps=config["fps"], rescale=False, n_rows=1)

print("\nGIF conversion complete.")
