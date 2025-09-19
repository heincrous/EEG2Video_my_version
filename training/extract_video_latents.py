import os
import numpy as np
import torch
from diffusers import AutoencoderKL
from torchvision.io import read_video
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
video_dir = "/content/drive/MyDrive/Data/Raw/Video/"
latent_outdir = "/content/drive/MyDrive/Data/Processed/Video_latents/"
os.makedirs(latent_outdir, exist_ok=True)

# Load pretrained Stable Diffusion VAE
vae = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="vae"
).to(device)
vae.eval()

# Preprocessing transform for frames
transform = transforms.Compose([
    transforms.Resize((288, 512)),  # match EEG2Video downsampled size
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def video_to_latents(video_path, save_path):
    # Load video frames
    video, _, _ = read_video(video_path, pts_unit="sec")   # [T,H,W,C]
    frames = [transform(frame).unsqueeze(0) for frame in video]  # list of [1,3,288,512]
    frames = torch.cat(frames).to(device)                  # [T,3,288,512]

    with torch.no_grad():
        latents = vae.encode(frames).latent_dist.mean      # [T,4,36,64]

    np.save(save_path, latents.cpu().numpy())

# Process all videos
for fname in sorted(os.listdir(video_dir)):
    if fname.endswith(".mp4"):
        in_path = os.path.join(video_dir, fname)
        out_path = os.path.join(latent_outdir, fname.replace(".mp4", ".npy"))
        print(f"Processing {fname} -> {out_path}")
        video_to_latents(in_path, out_path)

print("All videos processed and latents saved.")
