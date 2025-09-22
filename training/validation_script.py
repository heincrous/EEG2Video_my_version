"""
VALIDATE PROCESSED DATASET
--------------------------
Checks shapes and counts for all modalities, both subject-independent
and subject-dependent, including master list files and subject-level
master arrays. Handles partial datasets gracefully.
"""

import os
import numpy as np
from tqdm import tqdm

base_dir = "/content/drive/MyDrive/EEG2Video_data/processed/"

# modalities
modalities_subindep = ["Video_mp4", "Video_gif", "Video_latents", "BLIP_text", "BLIP_embeddings"]
modalities_subdep = ["EEG_segments", "EEG_windows", "EEG_features"]

# expected trailing shapes for per-clip files
expected_trailing = {
    "EEG_segments": (400, 62),
    "EEG_windows": (7, 62, 100),
    "EEG_features": (310,),
    "Video_latents": (4, 36, 64),  # adjust if different
    "BLIP_embeddings": (512,),     # adjust if different
}

def check_npy(path, modality):
    """Check trailing shape of a numpy file for a given modality."""
    try:
        arr = np.load(path)
        exp = expected_trailing.get(modality, None)
        if exp and arr.shape[-len(exp):] != exp:
            print(f"[{modality}] Shape mismatch: {path}, got {arr.shape}, expected *{exp}")
    except Exception as e:
        print(f"[{modality}] Error loading {path}: {e}")

def verify_master_lists(modality):
    """Check train/test index master files and counts."""
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
    """Check subject-level .npy master arrays before splitting."""
    try:
        arr = np.load(path)
        if modality == "EEG_segments":
            assert arr.shape[-2:] == (62,400), f"Bad shape {arr.shape} in {path}"
        elif modality == "EEG_windows":
            assert arr.shape[-3:] == (62,100), f"Bad shape {arr.shape} in {path}"
        elif modality == "EEG_features":
            assert arr.shape[-1] == 310, f"Bad shape {arr.shape} in {path}"
        print(f"[{modality}] {os.path.basename(path)} shape {arr.shape}")
    except Exception as e:
        print(f"[{modality}] Error loading {path}: {e}")

def count_files(modality):
    """Count number of files in train/test for a modality."""
    for split in ["train", "test"]:
        split_dir = os.path.join(base_dir, modality, split)
        if not os.path.exists(split_dir):
            continue
        count = 0
        for root, _, files in os.walk(split_dir):
            for fname in files:
                if fname.endswith(".npy") or fname.endswith(".mp4") or fname.endswith(".gif") or fname.endswith(".txt"):
                    count += 1
                    # optionally check shape for npy
                    if fname.endswith(".npy"):
                        check_npy(os.path.join(root, fname), modality)
        print(f"[{modality}] {split} set has {count} files")

# check subject-independent modalities
for mod in modalities_subindep:
    mod_dir = os.path.join(base_dir, mod)
    print(f"\n=== Checking {mod} ===")
    count_files(mod)
    verify_master_lists(mod)

# check subject-dependent modalities
for mod in modalities_subdep:
    mod_dir = os.path.join(base_dir, mod)
    print(f"\n=== Checking {mod} ===")

    # subject-level master arrays
    for fname in os.listdir(mod_dir):
        if fname.endswith(".npy") and not os.path.isdir(os.path.join(mod_dir, fname)):
            check_subject_master(os.path.join(mod_dir, fname), mod)

    count_files(mod)
    verify_master_lists(mod)
