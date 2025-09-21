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
from core_files.unet import UNet3DConditionModel
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel

# ---------------- CONFIG ----------------
OUTPUT_DIR = "/content/drive/MyDrive/EEG2Video_checkpoints/EEG2Video_diffusion_output"
BLIP_CAP_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_captions"
TEST_SPLIT_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/Split_4train1test/test/Video_latents"
SAVE_DIR = "/content/drive/MyDrive/EEG2Video_inference"
os.makedirs(SAVE_DIR, exist_ok=True)
VIDEO_LENGTH = 6  # number of frames

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
                    frame = frame.transpose(1,2,0)
                frame = np.clip(frame*255,0,255).astype(np.uint8)
                frames.append(frame)
            imageio.mimsave(f"{path}_{i}.gif", frames, fps=5)
    elif videos.ndim == 4:  # [F, C, H, W]
        frames = []
        for frame in videos:
            if frame.shape[0] == 1:
                frame = frame.squeeze(0)
            else:
                frame = frame.transpose(1,2,0)
            frame = np.clip(frame*255,0,255).astype(np.uint8)
            frames.append(frame)
        imageio.mimsave(path, frames, fps=5)

# ---------------- LOAD UNET MANUALLY ----------------
unet_path = os.path.join(OUTPUT_DIR, "unet")
unet = UNet3DConditionModel.from_pretrained_2d(unet_path).to("cuda")
unet.half()  # FP16 for efficiency

# ---------------- LOAD PIPELINE ----------------
pipeline = TuneAVideoPipeline.from_pretrained(
    OUTPUT_DIR,
    unet=unet  # pass UNet manually
).to("cuda")
pipeline.enable_vae_slicing()

# ---------------- FIND TEST CAPTIONS ----------------
test_blocks = sorted(os.listdir(TEST_SPLIT_DIR))
test_captions = []

for block in test_blocks:
    block_video_dir = os.path.join(TEST_SPLIT_DIR, block)
    block_cap_dir = os.path.join(BLIP_CAP_DIR, block)
    if not os.path.exists(block_cap_dir):
        continue
    video_files = sorted([f for f in os.listdir(block_video_dir) if f.endswith(".npy")])
    for vf in video_files:
        txt_file = vf.replace(".npy",".txt")
        txt_path = os.path.join(block_cap_dir, txt_file)
        if os.path.exists(txt_path):
            test_captions.append((block, vf, txt_path))

print(f"Found {len(test_captions)} test captions matching test video latents")

# ---------------- RUN INFERENCE ----------------
for txt_path in tqdm(test_captions, desc="Generating GIFs"):
    block, vf, txt_path_file = txt_path  # unpack
    with open(txt_path_file, "r") as f:
        prompt_text = f.read().strip()

    with torch.no_grad():
        sample = pipeline(prompt_text, generator=None, latents=None, video_length=VIDEO_LENGTH).videos

    save_name = f"{block}_{vf.replace('.npy','.gif')}"
    save_path = os.path.join(SAVE_DIR, save_name)
    save_videos_grid(sample, save_path)

print(f"\nAll test GIFs saved to {SAVE_DIR}")
