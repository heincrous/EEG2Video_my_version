"""
ENCODE MP4 CLIPS INTO VIDEO LATENTS (VAE)
------------------------------------------
Input:
  processed/Video_mp4/BlockY/classYY_clipZZ.mp4
    Each clip = 48 frames @ 24 fps, 512x512

Process:
  - Load clip frames
  - Downsample to 6 frames (3 FPS, matches EEG2Video paper)
  - Encode each frame with Stable Diffusion VAE
  - Latent per frame = [4,36,64]
  - Stack into sequence [6,4,36,64]

Output:
  processed/Video_latents/BlockY/classYY_clipZZ.npy
    Shape = [6,4,36,64]
"""

import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from diffusers import AutoencoderKL

# paths
in_dir = "/content/drive/MyDrive/EEG2Video_data/processed/Video_mp4/"
out_dir = "/content/drive/MyDrive/EEG2Video_data/processed/Video_latents/"
os.makedirs(out_dir, exist_ok=True)

# load pretrained Stable Diffusion VAE (v1-4 to match training)
device = "cuda" if torch.cuda.is_available() else "cpu"
vae = AutoencoderKL.from_pretrained(
    "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4",
    subfolder="vae"
).to(device, dtype=torch.float16)
vae.eval()

# parameters
target_size = (512, 512)
fps = 24
subsample = True   # always subsample to 6 frames
batch_size = 64     # number of frames per VAE forward pass

# -------------------------------------------------
# List available blocks to process
# -------------------------------------------------
all_blocks = sorted([b for b in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, b))])
print("Available blocks:")
for i, b in enumerate(all_blocks):
    print(f"{i}: {b}")

chosen = input("Enter indices of blocks to process (comma separated, 'all' for all): ").strip()
if chosen.lower() == "all":
    selected_blocks = all_blocks
else:
    idxs = [int(x) for x in chosen.split(",")]
    selected_blocks = [all_blocks[i] for i in idxs]

# -------------------------------------------------
# Process selected blocks
# -------------------------------------------------
for block in selected_blocks:
    block_path = os.path.join(in_dir, block)
    out_block = os.path.join(out_dir, block)
    os.makedirs(out_block, exist_ok=True)

    for fname in tqdm(os.listdir(block_path), desc=f"Encoding {block}"):
        if not fname.endswith(".mp4"):
            continue

        video_path = os.path.join(block_path, fname)
        cap = cv2.VideoCapture(video_path)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            print(f"Warning: no frames in {video_path}, skipping")
            continue

        # convert to tensor [B,3,512,512]
        frames = np.stack(frames, axis=0).astype(np.float32) / 255.0
        frames = torch.from_numpy(frames).permute(0,3,1,2).to(device, dtype=torch.float16)

        # subsample to 6 frames (3 FPS, 2s clip)
        if subsample and frames.shape[0] == 48:
            idxs = np.linspace(0, 47, 6, dtype=int)  # 6 evenly spaced frames
            frames = frames[idxs]

        # encode frames in batches
        with torch.no_grad():
            latents = []
            frames = frames.to(memory_format=torch.channels_last)
            for i in range(0, frames.shape[0], batch_size):
                batch = frames[i:i+batch_size] * 2 - 1  # scale to [-1,1]
                latent = vae.encode(batch).latent_dist.sample()
                latent = latent * 0.18215
                latents.append(latent)
            latents = torch.cat(latents, dim=0)  # [6,4,64,64]

        # resize spatial dimension to [36,64] (to match paper)
        latents = torch.nn.functional.interpolate(latents, size=(36,64), mode="bilinear")

        # save
        latents = latents.cpu().numpy()
        save_path = os.path.join(out_block, fname.replace(".mp4",".npy"))
        np.save(save_path, latents)

    print(f"Finished encoding {block} → {out_block}")
