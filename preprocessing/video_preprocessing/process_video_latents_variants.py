# ==========================================
# ENCODE MP4 CLIPS INTO VIDEO LATENTS (with 3 variants)
# ==========================================
# Input:
#   EEG2Video_data/processed/Video_mp4/BlockY/classYY_clipZZ.mp4
#   Each clip = 48 frames @ 24 fps, 512×288
#
# Process:
#   - Load frames, resize to 512×288 (no distortion)
#   - Generate 3 variants by offsetting subsampling (0,4,8)
#   - Each variant: 6 frames @ 3 FPS
#   - Encode each frame with Stable Diffusion VAE
#   - Latent per frame = [4,36,64]
#   - Collect → [7, 40, 15, 6, 4, 36, 64]   (15 clips per class)
#
# Output:
#   EEG2Video_data/processed/Video_latents/Video_latents_variants.npy
# ==========================================

import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from diffusers import AutoencoderKL


# ==========================================
# CONFIG
# ==========================================
config = {
    "fps":           24,
    "resize_w":      512,
    "resize_h":      288,
    "target_frames": 48,   # expected clip length @ 24 fps
    "variants":      [0, 2, 4, 6, 8, 10],  # offsets for subsampling
    "frames_per_clip": 6,
    "batch_size":    32,   # frames per VAE forward pass

    "drive_root": "/content/drive/MyDrive/EEG2Video_data",
    "vae_path":   "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4",
}

# paths
in_dir  = os.path.join(config["drive_root"], "processed", "Video_mp4")
out_dir = os.path.join(config["drive_root"], "processed", "Video_latents")
os.makedirs(out_dir, exist_ok=True)


# ==========================================
# Load pretrained VAE
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
vae = AutoencoderKL.from_pretrained(config["vae_path"], subfolder="vae").to(device, dtype=torch.float32)
vae.eval()


# ==========================================
# List available blocks
# ==========================================
all_blocks = sorted([b for b in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, b))])
print("Available blocks:")
for i, b in enumerate(all_blocks):
    print(f"[{i}] {b}")


# ==========================================
# Allocate block-level array
# ==========================================
all_latents = np.zeros((7, 40, 15, config["frames_per_clip"], 4, 36, 64), dtype=np.float32)


# ==========================================
# Main processing loop
# ==========================================
for block_id, block in enumerate(all_blocks):
    block_path = os.path.join(in_dir, block)
    print(f"\nProcessing {block}...")

    mp4_files = sorted([f for f in os.listdir(block_path) if f.endswith(".mp4")])
    for fname in tqdm(mp4_files, desc=f"Encoding {block}"):
        # parse class and clip indices
        parts = fname.replace(".mp4", "").split("_")
        cls  = int(parts[0].replace("class", ""))
        clip = int(parts[1].replace("clip", "")) - 1

        video_path = os.path.join(block_path, fname)
        cap = cv2.VideoCapture(video_path)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (config["resize_w"], config["resize_h"]), interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) != config["target_frames"]:
            print(f"Warning: {video_path} has {len(frames)} frames, expected {config['target_frames']}. Skipping.")
            continue

        frames = np.stack(frames, axis=0).astype(np.float32) / 255.0
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).to(device, dtype=torch.float32)

        # generate variants
        for v, offset in enumerate(config["variants"]):
            idxs = np.linspace(offset, config["target_frames"] - 1, config["frames_per_clip"], dtype=int)
            frames_sel = frames[idxs]

            # encode frames in batches
            with torch.no_grad():
                latents = []
                frames_sel = frames_sel.to(memory_format=torch.channels_last)
                for i in range(0, frames_sel.shape[0], config["batch_size"]):
                    batch = frames_sel[i:i+config["batch_size"]] * 2 - 1  # scale to [-1,1]
                    latent = vae.encode(batch).latent_dist.sample()   # fp32
                    latent = latent * 0.18215
                    latents.append(latent)
                latents = torch.cat(latents, dim=0)  # [6,4,36,64]

            # save into slot: 5 clips × 3 variants = 15 slots
            variant_idx = clip * len(config["variants"]) + v
            all_latents[block_id, cls, variant_idx] = latents.cpu().numpy().astype(np.float32)


# ==========================================
# Save block-level array
# ==========================================
os.makedirs(out_dir, exist_ok=True)   # ensure directory still exists at save time
save_path = os.path.join(out_dir, "Video_latents_variants.npy")
np.save(save_path, all_latents)
print(f"\nSaved block-level latents → {save_path}, shape {all_latents.shape}")

print("\nProcessing complete.")
