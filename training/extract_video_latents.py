import os
import numpy as np
import torch
from diffusers import AutoencoderKL
from torchvision import transforms
import decord
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
video_root = "/content/drive/MyDrive/Data/Raw/Video"
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
    i0 = int(round(t0 * fps))
    i1 = int(round(t1 * fps))
    if i1 <= i0:
        i1 = i0 + int(round(2 * fps))  # fallback ~2s span

    idxs = np.linspace(i0, i1 - 1, num=F, dtype=int)
    frames = video_reader.get_batch(idxs).asnumpy()  # [F,H,W,C]

    proc_frames = []
    for f in frames:
        pil_img = Image.fromarray(f)
        tensor_img = transform(pil_img).unsqueeze(0)
        proc_frames.append(tensor_img)

    frames_torch = torch.cat(proc_frames, dim=0).to(device)  # [F,3,288,512]
    latents = encode_frames(frames_torch)  # [F,4,36,64]
    return latents

def process_block(block_idx, fname, F=6, max_concepts=40):
    path = os.path.join(video_root, fname)
    video_reader = decord.VideoReader(path)
    fps = video_reader.get_avg_fps()

    for c in range(max_concepts):  # all 40 concepts
        base = 3 + 13 * c
        for i in range(5):
            t0, t1 = base + 2 * i, base + 2 * i + 2
            lat = extract_clip(video_reader, fps, t0, t1, F=F)

            out_dir = os.path.join(out_root, f"block{block_idx+1:02d}", f"class{c:02d}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"clip{i}.npy")
            np.save(out_path, lat)

    print(f"Finished block {block_idx+1}: saved {max_concepts*5} clips to {out_root}")

def main():
    files = {
        0: "2nd_10min.mp4",
        1: "3rd_10min.mp4",
        2: "4th_10min.mp4"
        }
    for b, fname in files.items():
        print(f"Processing block {b+1}: {fname}")
        process_block(b, fname, F=6, max_concepts=40)
    print("All requested blocks processed. Latents saved per clip.")

if __name__ == "__main__":
    main()
