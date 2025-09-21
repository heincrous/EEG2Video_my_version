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
MASTER_LATENTS_FILE = "/content/drive/MyDrive/EEG2Video_data/processed/40classes_latents.pt"
LOG_FILE = "/content/drive/MyDrive/EEG2Video_data/processed/processed_log.txt"
PROCESS_TAG = "[VIDEO_LATENT]"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Stable Diffusion VAE
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae")
vae = vae.to(device)
vae.eval()

os.makedirs(SAVE_DIR, exist_ok=True)

def get_folders_in_directory(directory):
    return sorted(
        [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))],
        key=lambda x: int(re.findall(r'\d+', x)[0])
    )

def load_processed_log():
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, "r") as f:
        return set(line.strip() for line in f.readlines())

def update_processed_log(filename):
    entry = f"{PROCESS_TAG} {filename}"
    if entry not in load_processed_log():
        with open(LOG_FILE, "a") as f:
            f.write(entry + "\n")

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

    # Downsample from (4,64,64) -> (4,36,64) for consistency with model
    latents = F.interpolate(latents, size=(36, 64), mode="bilinear", align_corners=False)

    return latents.cpu()

def append_to_master_latents(new_latents):
    # Create master latent file if not exists
    if os.path.exists(MASTER_LATENTS_FILE):
        master = torch.load(MASTER_LATENTS_FILE)
        updated = torch.cat([master, new_latents.unsqueeze(0)], dim=0)
    else:
        updated = new_latents.unsqueeze(0)
    torch.save(updated, MASTER_LATENTS_FILE)

# -----------------------------
# Clear old master latents at start
# -----------------------------
if os.path.exists(MASTER_LATENTS_FILE):
    os.remove(MASTER_LATENTS_FILE)
    print(f"Old master latent file removed: {MASTER_LATENTS_FILE}")

# Main loop
block_folders = get_folders_in_directory(GIF_DIR)
processed = load_processed_log()
processed_count, skipped_count = 0, 0
first_shape_printed = False

for block in block_folders:
    block_path = os.path.join(GIF_DIR, block)
    save_block_path = os.path.join(SAVE_DIR, block)
    os.makedirs(save_block_path, exist_ok=True)

    gif_files = sorted([f for f in os.listdir(block_path) if f.endswith(".gif")],
                       key=lambda x: int(os.path.splitext(x)[0]))

    for gif_file in tqdm(gif_files, desc=f"Block {block}"):
        entry = f"{PROCESS_TAG} {block}/{gif_file}"
        if entry in processed:
            skipped_count += 1
            continue

        gif_path = os.path.join(block_path, gif_file)
        latents = encode_gif_to_latent(gif_path)

        # Print shape for the first processed file as verification
        if not first_shape_printed:
            print(f"Example latent shape for {gif_file}: {latents.shape}")
            first_shape_printed = True

        # Save per-clip latent
        np.save(os.path.join(save_block_path, gif_file.replace(".gif", ".npy")), latents.numpy())

        # Append to master file
        append_to_master_latents(latents)

        # Update log
        update_processed_log(f"{block}/{gif_file}")
        processed_count += 1

print(f"\nSummary: {processed_count} clips processed, {skipped_count} skipped")
print(f"Master latent file: {MASTER_LATENTS_FILE}")
