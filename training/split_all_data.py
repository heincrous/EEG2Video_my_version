import os
import random
import shutil
import json
from glob import glob
import numpy as np
import torch

# CONFIG
BASE_DIR = "/content/drive/MyDrive/EEG2Video_data/processed"
MODALITIES = ["EEG_segments", "Video_latents", "BLIP_embeddings", "Meta_info"]
SAVE_DIR = os.path.join(BASE_DIR, "Split")
TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)

os.makedirs(SAVE_DIR, exist_ok=True)
for split in ["train", "test"]:
    for modality in MODALITIES:
        os.makedirs(os.path.join(SAVE_DIR, split, modality), exist_ok=True)

def get_class_id(filename):
    parts = filename.split("/")
    for p in parts:
        if p.startswith("class"):
            return p
    return None

# collect all clips by class
clips_by_class = {}
for modality in MODALITIES:
    modality_dir = os.path.join(BASE_DIR, modality)
    files = glob(os.path.join(modality_dir, "Block*/*/*.npy")) + glob(os.path.join(modality_dir, "Block*/*/*.npz"))
    for f in files:
        rel = os.path.relpath(f, modality_dir)  # BlockX/classYY_clipZZ.npy
        clip_id = os.path.splitext(rel)[0]      # remove extension
        class_id = get_class_id(rel)
        if class_id is None:
            continue
        clips_by_class.setdefault(class_id, set()).add(clip_id)

train_set, test_set = [], []
for class_id, clips in clips_by_class.items():
    clips = sorted(list(clips))
    random.shuffle(clips)
    n_train = int(len(clips) * TRAIN_RATIO)
    train_set.extend(clips[:n_train])
    test_set.extend(clips[n_train:])

def copy_clip_files(clip_id, split):
    for modality in MODALITIES:
        modality_dir = os.path.join(BASE_DIR, modality)
        dest_dir = os.path.join(SAVE_DIR, split, modality)
        src_candidates = glob(os.path.join(modality_dir, clip_id + ".*"))
        for src in src_candidates:
            dst = os.path.join(dest_dir, os.path.basename(src))
            shutil.copy(src, dst)

for cid in train_set:
    copy_clip_files(cid, "train")
for cid in test_set:
    copy_clip_files(cid, "test")

manifest = {"train": train_set, "test": test_set}
with open(os.path.join(SAVE_DIR, "split_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

# Build master latent file from Video_latents
all_latents = []
for split in ["train", "test"]:
    lat_dir = os.path.join(SAVE_DIR, split, "Video_latents")
    lat_files = sorted(glob(os.path.join(lat_dir, "*.npy")))
    for lf in lat_files:
        arr = np.load(lf)
        all_latents.append(torch.from_numpy(arr))

if all_latents:
    master_tensor = torch.stack(all_latents)
    torch.save(master_tensor, os.path.join(SAVE_DIR, "all_latents.pt"))
    print(f"Master latent tensor saved with shape {master_tensor.shape}")

print(f"Split done. Train clips: {len(train_set)}, Test clips: {len(test_set)}")
print(f"Manifest saved to {os.path.join(SAVE_DIR, 'split_manifest.json')}")