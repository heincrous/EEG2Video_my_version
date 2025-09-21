import sys
import os
import torch
import numpy as np
import imageio

# ---------------- Add repo root to path ----------------
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

# ---------------- Imports ----------------
from pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from core_files.unet import UNet3DConditionModel
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel

# ---------------- CONFIG ----------------
CHECKPOINT_DIR = "./EEG2Video_checkpoints/EEG2Video_diffusion_output"
BLIP_CAP_DIR = "./EEG2Video_data/processed/BLIP_captions"
TEST_SPLIT_DIR = "./EEG2Video_data/processed/Split_4train1test/test/Video_latents"
SAVE_DIR = "./EEG2Video_inference"
os.makedirs(SAVE_DIR, exist_ok=True)
VIDEO_LENGTH = 6  # must match training

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
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
                frames.append(frame)
            imageio.mimsave(f"{path}_{i}.gif", frames, fps=5)
    elif videos.ndim == 4:  # [F, C, H, W]
        frames = []
        for frame in videos:
            if frame.shape[0] == 1:
                frame = frame.squeeze(0)
            else:
                frame = frame.transpose(1, 2, 0)
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            frames.append(frame)
        imageio.mimsave(path, frames, fps=5)

# ---------------- LOAD LOCAL MODELS ----------------
unet = UNet3DConditionModel.from_pretrained_2d(os.path.join(CHECKPOINT_DIR, "unet"))
vae = AutoencoderKL.from_pretrained(os.path.join(CHECKPOINT_DIR, "vae"))
text_encoder = CLIPTextModel.from_pretrained(os.path.join(CHECKPOINT_DIR, "text_encoder"))
scheduler = DDIMScheduler.from_pretrained(os.path.join(CHECKPOINT_DIR, "scheduler"))

pipeline = TuneAVideoPipeline(
    unet=unet,
    vae=vae,
    text_encoder=text_encoder,
    scheduler=scheduler,
    tokenizer=None  # set if needed
)
pipeline.enable_vae_slicing()
pipeline.to("cuda")

# ---------------- CROSS-CHECK TEST CAPTIONS ----------------
test_blocks = sorted(os.listdir(TEST_SPLIT_DIR))
test_captions = []

for block in test_blocks:
    block_video_dir = os.path.join(TEST_SPLIT_DIR, block)
    block_cap_dir = os.path.join(BLIP_CAP_DIR, block)
    if not os.path.exists(block_cap_dir):
        continue
    video_files = sorted([f for f in os.listdir(block_video_dir) if f.endswith(".npy")])
    for vf in video_files:
        txt_file = vf.replace(".npy", ".txt")
        txt_path = os.path.join(block_cap_dir, txt_file)
        if os.path.exists(txt_path):
            test_captions.append((block, vf, txt_path))

print(f"Found {len(test_captions)} test captions matching video latents")

# ---------------- INFERENCE ----------------
for block, vf, txt_path in test_captions:
    with open(txt_path, "r") as f:
        prompt_text = f.read().strip()

    with torch.no_grad():
        sample = pipeline(prompt_text, generator=None, latents=None, video_length=VIDEO_LENGTH).videos

    save_path = os.path.join(SAVE_DIR, f"{block}_{vf.replace('.npy','.gif')}")
    save_videos_grid(sample, save_path)

print(f"\nAll test GIFs saved to {SAVE_DIR}")
