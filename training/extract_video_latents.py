import os, math
import numpy as np
import torch
from diffusers import AutoencoderKL
from torchvision.io import read_video
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

video_root = "/content/drive/MyDrive/Data/Raw/Video"            # contains 1st_10min.mp4 ... 7th_10min.mp4
out_root   = "/content/drive/MyDrive/Data/Processed/Video_latents_per_clip"  # we will write per-clip latents here
os.makedirs(out_root, exist_ok=True)

vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device)
vae.eval()

transform = transforms.Compose([
    transforms.Resize((288, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def encode_frames(frames_tchw):
    with torch.no_grad():
        lat = vae.encode(frames_tchw).latent_dist.mean
    return lat.cpu().numpy()  # [T,4,36,64]

def extract_clip(video_tensor, fps, t0, t1, F):
    # video_tensor is [T, H, W, C] in uint8
    # pick frames between t0 and t1 seconds, evenly sampled to F frames
    T = video_tensor.shape[0]
    # Map times to frame indices
    i0 = max(0, min(T-1, int(round(t0 * fps))))
    i1 = max(0, min(T-1, int(round(t1 * fps))))
    if i1 <= i0:
        i1 = min(T-1, i0 + max(1, int(round(2 * fps))))  # fallback to ~2s span
    idxs = np.linspace(i0, i1-1, num=F, dtype=int)
    # transform each chosen frame
    frames = []
    for i in idxs:
        frame = video_tensor[i]            # [H,W,C], uint8
        frame = transform(frame).unsqueeze(0)  # [1,3,288,512]
        frames.append(frame)
    frames = torch.cat(frames, dim=0).to(device)  # [F,3,288,512]
    latents = encode_frames(frames)               # [F,4,36,64]
    return latents

def process_block(block_idx, fname, F=8):
    # block_idx in [0..6] -> 1st_10min.mp4 etc.
    path = os.path.join(video_root, fname)
    video, audio, info = read_video(path, pts_unit="sec")
    fps = info["video_fps"]
    # schedule: per concept c, start at 3 + 13*c, then clips at +2*i, each 2s long
    for c in range(40):
        base = 3 + 13*c
        for i in range(5):
            t0 = base + 2*i
            t1 = t0 + 2.0
            lat = extract_clip(video, fps, t0, t1, F)  # [F,4,36,64]
            out_dir = os.path.join(out_root, f"block{block_idx+1:02d}", f"class{c:02d}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"clip{i}.npy")
            np.save(out_path, lat)

def main():
    files = {0: "1st_10min.mp4", 1: "2nd_10min.mp4", 2: "3rd_10min.mp4",
             3: "4th_10min.mp4", 4: "5th_10min.mp4", 5: "6th_10min.mp4", 6: "7th_10min.mp4"}
    for b, fname in files.items():
        print(f"Processing block {b+1}: {fname}")
        process_block(b, fname, F=8)
    print("Done. Latents saved per clip under Video_latents_per_clip/.")

if __name__ == "__main__":
    main()
