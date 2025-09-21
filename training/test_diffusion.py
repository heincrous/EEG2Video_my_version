import sys
import os
import torch
import numpy as np
import imageio
from tqdm import tqdm

# ---------------- Add repo root ----------------
repo_root = "/content/EEG2Video_my_version"
sys.path.append(repo_root)

# ---------------- Imports ----------------
from pipelines.pipeline_tuneavideo import TuneAVideoPipeline

# ---------------- CONFIG ----------------
OUTPUT_DIR = "/content/drive/MyDrive/EEG2Video_checkpoints/EEG2Video_diffusion_output"
BLIP_CAP_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_captions"
SAVE_DIR = "/content/drive/MyDrive/EEG2Video_inference"
os.makedirs(SAVE_DIR, exist_ok=True)
VIDEO_LENGTH = 6  # frames

# ---------------- INLINE GIF SAVING ----------------
def save_videos_grid(videos, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(videos, torch.Tensor):
        videos = videos.cpu().numpy()
    if videos.ndim == 5:  # [B, F, C, H, W]
        for i, video in enumerate(videos):
            frames = []
            for frame in video:
                if frame.shape[0] == 1:
                    frame = frame.squeeze(0)
                else:
                    frame = frame.transpose(1, 2, 0)
                frame = np.clip(frame*255, 0, 255).astype(np.uint8)
                frames.append(frame)
            imageio.mimsave(f"{path}_{i}.gif", frames, fps=5)
    elif videos.ndim == 4:  # [F, C, H, W]
        frames = []
        for frame in videos:
            if frame.shape[0] == 1:
                frame = frame.squeeze(0)
            else:
                frame = frame.transpose(1,2,0)
            frame = np.clip(frame*255, 0, 255).astype(np.uint8)
            frames.append(frame)
        imageio.mimsave(path, frames, fps=5)

# ---------------- LOAD PIPELINE ----------------
pipeline = TuneAVideoPipeline.from_pretrained(OUTPUT_DIR).to("cuda")
pipeline.enable_vae_slicing()
pipeline.unet.half()
pipeline.vae.half()

# ---------------- GET ALL TEST CAPTIONS ----------------
test_captions = []
for block in sorted(os.listdir(BLIP_CAP_DIR)):
    block_dir = os.path.join(BLIP_CAP_DIR, block)
    if not os.path.exists(block_dir):
        continue
    for txt_file in sorted(os.listdir(block_dir)):
        if txt_file.endswith(".txt"):
            test_captions.append(os.path.join(block_dir, txt_file))

print(f"Found {len(test_captions)} captions for inference.")

# ---------------- RUN INFERENCE WITH PROGRESS BAR ----------------
for txt_path in tqdm(test_captions, desc="Generating GIFs"):
    with open(txt_path, "r") as f:
        prompt_text = f.read().strip()

    with torch.no_grad():
        sample = pipeline(prompt_text, generator=None, latents=None, video_length=VIDEO_LENGTH).videos

    save_name = os.path.basename(txt_path).replace(".txt", ".gif")
    save_path = os.path.join(SAVE_DIR, save_name)
    save_videos_grid(sample, save_path)

print(f"\nAll GIFs saved to {SAVE_DIR}")
