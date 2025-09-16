import os
import torch
import numpy as np
from decord import VideoReader, cpu
from diffusers import AutoencoderKL

# Paths
EEG_PATH = "/content/drive/MyDrive/Data/Raw/SEED-DV/EEG/sub1.npy"
VIDEO_PATH = "/content/drive/MyDrive/Data/Raw/SEED-DV/Video/1st_10min.mp4"

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load EEG (block 1 only, first 512 samples)
eeg_data = np.load(EEG_PATH)  # shape (7, 62, 104000)
block1 = eeg_data[0]
eeg_segment = torch.tensor(block1[:, :512], dtype=torch.float32).unsqueeze(0).to(device)

# 2. Load video (grab first few frames)
vr = VideoReader(VIDEO_PATH, ctx=cpu(0))
frames = [torch.tensor(vr[i].asnumpy()).permute(2,0,1).unsqueeze(0).float()/255.0 for i in range(0,12,2)]
video_tensor = torch.cat(frames).to(device)

# 3. Encode frames into latents with pretrained VAE
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
with torch.no_grad():
    video_tensor = torch.nn.functional.interpolate(video_tensor, (256,256))
    latents = vae.encode(video_tensor).latent_dist.sample() * 0.18215

# Print shapes for debugging
print("EEG segment shape:", eeg_segment.shape)
print("Video tensor shape:", video_tensor.shape)
print("Latents shape:", latents.shape)
