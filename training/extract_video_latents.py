# training/extract_video_latents_per_clip.py
import os
import numpy as np
import torch
from diffusers import AutoencoderKL
from torchvision import transforms
import decord

device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
video_root = "/content/drive/MyDrive/Data/Raw/Video"   # contains 1st_10min.mp4 ... 7th_10min.mp4
out_root   = "/content/drive/MyDrive/Data/Processed/Video_latents_per_clip"
os.makedirs(out_root, exist_ok=True)

# Load pretrained Stable Diffusion VAE
vae = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="vae"
).to(device)
vae.eval()

# Preprocessing transform for frames
transform = transforms.Compose([
    transforms.Resize((288, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def encode_frames(frames_tchw):
    with torch.no_grad():
        lat = vae.encode(frames_tchw).latent_dist.mean
    return lat.cpu().numpy().astype("float32")  # [T,4,36,64]

def extract_clip(video_reader, fps, t0, t1, F=6):
    """Extract a 2s clip between t0–t1 sec, sample F frames evenly, encode to latents."""
    i0 = int(round(t0 * fps))
    i1 = int(round(t1 * fps))
    if i1 <= i0:
        i1 = i0 + int(round(2 * fps))  # fallback ~2s span

    idxs = np.linspace(i0, i1 - 1, num=F, dtype=int)

    frames = video_reader.get_batch(idxs).asnumpy()  # [F,H,W,C]
    frames = torch.stack([transform(f).unsqueeze(0) for f in frames])  # [F,3,288,512]
    frames = frames.squeeze(1).to(device)

    latents = encode_frames(frames)  # [F,4,36,64]
    return latents

def process_block(block_idx, fname, F=6):
    path = os.path.join(video_root, fname)
    video_reader = decord.VideoReader(path)
    fps = video_reader.get_avg_fps()

    for c in range(40):
        base = 3 + 13 * c  # skip 3s hint, then 13s per concept
        for i in range(5):
            t0, t1 = base + 2 * i, base + 2 * i + 2
            lat = extract_clip(video_reader, fps, t0, t1, F=F)

            out_dir = os.path.join(out_root, f"block{block_idx+1:02d}", f"class{c:02d}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"clip{i}.npy")

            np.save(out_path, lat)
    print(f"Finished block {block_idx+1}: saved latents to {out_root}")

def main():
    files = {
        0: "1st_10min.mp4",
        1: "2nd_10min.mp4",
        2: "3rd_10min.mp4",
        3: "4th_10min.mp4",
        4: "5th_10min.mp4",
        5: "6th_10min.mp4",
        6: "7th_10min.mp4"
    }
    for b, fname in files.items():
        print(f"Processing block {b+1}: {fname}")
        process_block(b, fname, F=6)
    print("✅ All blocks processed. Latents saved per clip.")

if __name__ == "__main__":
    main()
