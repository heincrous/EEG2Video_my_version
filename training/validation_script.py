"""
VALIDATE PROCESSED DATASET
--------------------------
Checks shapes and counts for all modalities, both subject-independent
and subject-dependent, including master list files and subject-level
master arrays. Handles partial datasets gracefully.
Also samples a random train/test video with its BLIP caption and embedding,
and inspects frames for MP4, GIF, and latent files.
"""

import os
import numpy as np
import random
import cv2
import imageio
from tqdm import tqdm

base_dir = "/content/drive/MyDrive/EEG2Video_data/processed/"

# modalities
modalities_subindep = ["Video_mp4", "Video_gif", "Video_latents", "BLIP_text", "BLIP_embeddings"]
modalities_subdep = ["EEG_segments", "EEG_windows", "EEG_features"]

# expected trailing shapes with explanations
expected_trailing = {
    "EEG_segments": ((400, 62), "time=400 samples (2s @200Hz), channels=62"),
    "EEG_windows": ((7, 62, 100), "windows=7 (overlapping), channels=62, samples=100"),
    "EEG_features": ((310,), "features=310 (DE/PSD features per band×channel)"),
    "Video_latents": ((4, 36, 64), "channels=4 (VAE latent), height=36, width=64, frames=N (time dimension)"),
    "BLIP_embeddings": ((77, 512), "sequence_length=77 tokens, embedding_dim=512"),
}

def explain_shape(modality, shape):
    exp, note = expected_trailing.get(modality, (None, None))
    return f"{shape} → {note}" if note else str(shape)

def check_npy(path, modality):
    """Check numpy shape and compare with expected."""
    try:
        arr = np.load(path)
        exp, note = expected_trailing.get(modality, (None, None))
        if exp and arr.shape[-len(exp):] != exp:
            print(f"[{modality}] Shape mismatch: {path}, got {arr.shape}, expected *{exp}")
    except Exception as e:
        print(f"[{modality}] Error loading {path}: {e}")

def verify_master_lists(modality):
    """Check train/test master list files."""
    for split in ["train", "test"]:
        list_path = os.path.join(base_dir, modality, f"{split}_list.txt")
        if not os.path.exists(list_path):
            print(f"[{modality}] Missing {split}_list.txt")
            continue
        with open(list_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        print(f"[{modality}] {split}_list.txt → {len(lines)} entries")
        for path in lines[:2]:
            print(f"  sample: {path}")

def check_subject_master(path, modality):
    """Check subject-level arrays before expansion."""
    try:
        arr = np.load(path)
        print(f"[{modality}] {os.path.basename(path)} shape {explain_shape(modality, arr.shape)}")
    except Exception as e:
        print(f"[{modality}] Error loading {path}: {e}")

def count_files(modality):
    """Count train/test files and check npy shapes."""
    for split in ["train", "test"]:
        split_dir = os.path.join(base_dir, modality, split)
        if not os.path.exists(split_dir):
            continue
        count = 0
        for root, _, files in os.walk(split_dir):
            for fname in files:
                if fname.endswith((".npy", ".mp4", ".gif", ".txt")):
                    count += 1
                    if fname.endswith(".npy"):
                        check_npy(os.path.join(root, fname), modality)
        print(f"[{modality}] {split} set has {count} files")

def inspect_video(path):
    """Check frame count and resolution of one MP4."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[Video_mp4] Could not open {path}")
        return
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"[Video_mp4] {os.path.basename(path)} → frames={frame_count}, resolution={w}x{h}")

def inspect_gif(path):
    """Check frame count of one GIF."""
    try:
        gif = imageio.mimread(path)
        print(f"[Video_gif] {os.path.basename(path)} → frames={len(gif)}, resolution={gif[0].shape[1]}x{gif[0].shape[0]}")
    except Exception as e:
        print(f"[Video_gif] Error reading {path}: {e}")

def inspect_latents(path):
    """Check shape of one latent tensor."""
    try:
        arr = np.load(path)
        print(f"[Video_latents] {os.path.basename(path)} shape {explain_shape('Video_latents', arr.shape)}")
    except Exception as e:
        print(f"[Video_latents] Error reading {path}: {e}")

def sample_video_and_caption(split="train"):
    """Pick random video and show caption + embedding."""
    video_dir = os.path.join(base_dir, "Video_mp4", split)
    blip_text_dir = os.path.join(base_dir, "BLIP_text", split)
    blip_emb_dir = os.path.join(base_dir, "BLIP_embeddings", split)

    all_videos = []
    for root, _, files in os.walk(video_dir):
        for f in files:
            if f.endswith(".mp4"):
                all_videos.append(os.path.join(root, f))
    if not all_videos:
        print(f"No videos found in {video_dir}")
        return
    chosen_video = random.choice(all_videos)
    rel_path = os.path.relpath(chosen_video, video_dir)

    txt_path = os.path.join(blip_text_dir, rel_path).replace(".mp4", ".txt")
    emb_path = os.path.join(blip_emb_dir, rel_path).replace(".mp4", ".npy")

    print(f"[{split.upper()}] Video: {chosen_video}")
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            firstline = f.readline().strip()
        print(f"[{split.upper()}] Caption: {firstline}")
    else:
        print(f"[{split.upper()}] Missing BLIP caption for {rel_path}")

    if os.path.exists(emb_path):
        try:
            emb = np.load(emb_path)
            print(f"[{split.upper()}] BLIP embedding shape: {explain_shape('BLIP_embeddings', emb.shape)}")
        except Exception as e:
            print(f"[{split.upper()}] Error loading embedding {emb_path}: {e}")
    else:
        print(f"[{split.upper()}] Missing BLIP embedding for {rel_path}")

# validate subject-independent modalities
for mod in modalities_subindep:
    print(f"\n=== Checking {mod} ===")
    count_files(mod)
    verify_master_lists(mod)

# validate subject-dependent modalities
for mod in modalities_subdep:
    print(f"\n=== Checking {mod} ===")
    mod_dir = os.path.join(base_dir, mod)
    for fname in os.listdir(mod_dir):
        if fname.endswith(".npy") and not os.path.isdir(os.path.join(mod_dir, fname)):
            check_subject_master(os.path.join(mod_dir, fname), mod)
    count_files(mod)
    verify_master_lists(mod)

# sample inspection of one MP4, one GIF, one latent
print("\n=== Inspecting one MP4/GIF/Latent ===")
mp4_dir = os.path.join(base_dir, "Video_mp4", "train")
gif_dir = os.path.join(base_dir, "Video_gif", "train")
lat_dir = os.path.join(base_dir, "Video_latents", "train")

for d, fn, func in [
    (mp4_dir, ".mp4", inspect_video),
    (gif_dir, ".gif", inspect_gif),
    (lat_dir, ".npy", inspect_latents),
]:
    found = False
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith(fn):
                func(os.path.join(root, f))
                found = True
                break
        if found:
            break

# random samples
print("\n=== Random sample from TRAIN ===")
sample_video_and_caption("train")
print("\n=== Random sample from TEST ===")
sample_video_and_caption("test")
