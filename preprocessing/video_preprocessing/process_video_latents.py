"""
ENCODE MP4 CLIPS INTO VIDEO LATENTS (SUBJECT-LEVEL ARRAY)
---------------------------------------------------------
Input:
  processed/Video_mp4/BlockY/classYY_clipZZ.mp4
    Each clip = 48 frames @ 24 fps, 512x288

Process:
  - Load frames, resize to 512x288 (no distortion)
  - Downsample to 6 frames (3 FPS, matches EEG2Video paper)
  - Encode each frame with Stable Diffusion VAE
  - Latent per frame = [4,36,64]
  - Stack into clip latent [6,4,36,64]
  - Collect all blocks → [7,40,5,6,4,36,64]

Output:
  processed/Video_latents/Video_latents.npy
    Shape = [7,40,5,6,4,36,64]
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

# load pretrained Stable Diffusion VAE
device = "cuda" if torch.cuda.is_available() else "cpu"
vae = AutoencoderKL.from_pretrained(
    "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4",
    subfolder="vae"
).to(device, dtype=torch.float16)
vae.eval()

# parameters
target_size = (512, 288)   # correct aspect ratio
fps = 24
subsample = True           # always subsample to 6 frames
batch_size = 32            # frames per VAE forward pass

# -------------------------------------------------
# List available blocks
# -------------------------------------------------
all_blocks = sorted([b for b in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, b))])
print("Available blocks:")
for i, b in enumerate(all_blocks):
    print(f"{i}: {b}")

# -------------------------------------------------
# Allocate subject-level array [7,40,5,6,4,36,64]
# -------------------------------------------------
all_latents = np.zeros((7,40,5,6,4,36,64), dtype=np.float16)

# -------------------------------------------------
# Process all blocks
# -------------------------------------------------
for block_id, block in enumerate(all_blocks):
    block_path = os.path.join(in_dir, block)
    print(f"\nProcessing {block}...")

    mp4_files = sorted([f for f in os.listdir(block_path) if f.endswith(".mp4")])
    for fname in tqdm(mp4_files, desc=f"Encoding {block}"):
        # parse class and clip indices
        parts = fname.replace(".mp4","").split("_")
        cls = int(parts[0].replace("class",""))
        clip = int(parts[1].replace("clip","")) - 1

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

        # convert to tensor [B,3,288,512]
        frames = np.stack(frames, axis=0).astype(np.float32) / 255.0
        frames = torch.from_numpy(frames).permute(0,3,1,2).to(device, dtype=torch.float16)

        # subsample to 6 frames (3 FPS)
        if subsample and frames.shape[0] == 48:
            idxs = np.linspace(0, 47, 6, dtype=int)
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
            latents = torch.cat(latents, dim=0)  # [6,4,36,64]

        all_latents[block_id, cls, clip] = latents.cpu().numpy()

# -------------------------------------------------
# Save subject-level array
# -------------------------------------------------
save_path = os.path.join(out_dir, "Video_latents.npy")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
np.save(save_path, all_latents)
print(f"\nSaved subject-level latents → {save_path}, shape {all_latents.shape}")
