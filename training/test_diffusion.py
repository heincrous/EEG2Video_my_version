import sys
import os
import torch
import numpy as np
import imageio

# ---------------- Add repo root ----------------
repo_root = "/content/EEG2Video_my_version"
sys.path.append(repo_root)

# ---------------- Imports ----------------
from pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from core_files.unet import UNet3DConditionModel

# ---------------- CONFIG ----------------
OUTPUT_DIR = "/content/drive/MyDrive/EEG2Video_checkpoints/EEG2Video_diffusion_output"
BLIP_CAP_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_captions"
SAVE_DIR = "/content/drive/MyDrive/EEG2Video_inference"
os.makedirs(SAVE_DIR, exist_ok=True)
VIDEO_LENGTH = 1  # single-frame GIF for fast testing

# ---------------- INLINE GIF SAVING ----------------
def save_videos_grid(videos, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(videos, torch.Tensor):
        videos = videos.cpu().numpy()
    if videos.ndim == 5:
        video = videos[0]  # take first video only
        frames = []
        for frame in video:
            if frame.shape[0] == 1:
                frame = frame.squeeze(0)
            else:
                frame = frame.transpose(1,2,0)
            frame = np.clip(frame*255,0,255).astype(np.uint8)
            frames.append(frame)
        imageio.mimsave(path, frames, fps=5)
    elif videos.ndim == 4:
        frames = []
        for frame in videos:
            if frame.shape[0] == 1:
                frame = frame.squeeze(0)
            else:
                frame = frame.transpose(1,2,0)
            frame = np.clip(frame*255,0,255).astype(np.uint8)
            frames.append(frame)
        imageio.mimsave(path, frames, fps=5)

# ---------------- CLEAR CUDA ----------------
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# ---------------- LOAD UNET ----------------
unet_path = os.path.join(OUTPUT_DIR, "unet")
unet = UNet3DConditionModel.from_pretrained_2d(unet_path).to("cuda", dtype=torch.float16)

# ---------------- LOAD PIPELINE ----------------
pipeline = TuneAVideoPipeline.from_pretrained(
    OUTPUT_DIR,
    unet=unet
).to("cuda", dtype=torch.float16)

pipeline.enable_vae_slicing()
pipeline.enable_attention_slicing()

# ---------------- COLLECT ONE TEST CAPTION ----------------
test_blocks = sorted(os.listdir(BLIP_CAP_DIR))
test_captions = []

for block in test_blocks:
    block_dir = os.path.join(BLIP_CAP_DIR, block)
    if not os.path.exists(block_dir):
        continue
    txt_files = sorted([f for f in os.listdir(block_dir) if f.endswith(".txt")])
    if txt_files:
        test_captions.append((block, txt_files[0], os.path.join(block_dir, txt_files[0])))
        break  # take only one caption

print(f"Testing 1 caption with VIDEO_LENGTH={VIDEO_LENGTH}")

# ---------------- RUN INFERENCE ----------------
block, txt_file, txt_path = test_captions[0]
with open(txt_path, "r") as f:
    prompt_text = f.read().strip()

with torch.autocast("cuda"), torch.no_grad():
    sample = pipeline(prompt_text, video_length=VIDEO_LENGTH).videos

save_name = f"{block}_{txt_file.replace('.txt','.gif')}"
save_path = os.path.join(SAVE_DIR, save_name)
save_videos_grid(sample, save_path)

print(f"\nTest GIF saved to {save_path}")
