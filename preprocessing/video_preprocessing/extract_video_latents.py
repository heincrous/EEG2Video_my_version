import os
import re
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import imageio
from diffusers import AutoencoderKL
import torch.nn.functional as F

# CONFIG
GIF_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/Video_Gif/"
SAVE_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/Video_latents/"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Stable Diffusion VAE
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae")
vae = vae.to(device)
vae.eval()

os.makedirs(SAVE_DIR, exist_ok=True)

def get_block_folders(directory):
    return sorted([f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))])

def encode_gif_to_latent(gif_path):
    frames = []
    gif = imageio.mimread(gif_path)
    for frame in gif:
        img = Image.fromarray(frame)
        img = img.resize((512, 512))
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        frames.append(img)
    x = torch.cat(frames, dim=0).to(device)  # (frames,3,512,512)
    x = 2.0 * x - 1.0
    with torch.no_grad():
        posterior = vae.encode(x).latent_dist
        latents = posterior.sample() * 0.18215  # (frames,4,64,64)

    # Downsample (4,64,64) -> (4,36,64) for consistency
    latents = F.interpolate(latents, size=(36, 64), mode="bilinear", align_corners=False)

    return latents.cpu().numpy()

# -----------------------------
# Ask user which blocks to process
# -----------------------------
all_blocks = get_block_folders(GIF_DIR)
print("Available blocks:", all_blocks)

user_input = input("Enter blocks to process (comma separated, e.g. 1st_10min,2nd_10min): ")
block_list = [b.strip() for b in user_input.split(",") if b.strip() in all_blocks]

if not block_list:
    raise ValueError("No valid blocks selected!")

processed_count = 0
first_shape_printed = False

for block in block_list:
    block_path = os.path.join(GIF_DIR, block)
    save_block_path = os.path.join(SAVE_DIR, block)
    os.makedirs(save_block_path, exist_ok=True)

    gif_files = sorted([f for f in os.listdir(block_path) if f.endswith(".gif")])

    for gif_file in tqdm(gif_files, desc=f"Encoding {block}"):
        gif_path = os.path.join(block_path, gif_file)
        latents = encode_gif_to_latent(gif_path)

        if not first_shape_printed:
            print(f"Example latent shape for {gif_file}: {latents.shape}")
            first_shape_printed = True

        save_path = os.path.join(save_block_path, gif_file.replace(".gif", ".npy"))
        np.save(save_path, latents)
        processed_count += 1

    print(f"Finished block {block}, saved into {save_block_path}")

print(f"\nSummary: {processed_count} GIF clips processed into latents")