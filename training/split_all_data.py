"""
DATA SPLITTING INSTRUCTIONS (WITH MASTER FILES)
-----------------------------------------------
Rule:
  - For each Block (1–7)
  - For each class (00–39)
  - Clip01–Clip04 → train
  - Clip05 → test

Inputs:
  Subject-independent modalities:
    - Video_mp4
    - Video_gif
    - Video_latents
    - BLIP_text
    - BLIP_embeddings

  Subject-dependent modalities:
    - EEG_segments (5D subject-level arrays: [7,40,5,62,400])
    - EEG_windows
    - EEG_features

Outputs:
  Subject-independent:
    processed/{Modality}/train/BlockY/classYY_clipZZ.*
    processed/{Modality}/test/BlockY/classYY_clipZZ.*

  Subject-dependent:
    processed/{Modality}/train/subX/BlockY/classYY_clipZZ.*
    processed/{Modality}/test/subX/BlockY/classYY_clipZZ.*

Master Files:
  For faster loading, create train/test index files after splitting.
"""

import os
import numpy as np
import shutil
from tqdm import tqdm

base_dir = "/content/drive/MyDrive/EEG2Video_data/processed/"
modalities_subindep = ["Video_mp4", "Video_gif", "Video_latents", "BLIP_text", "BLIP_embeddings"]
modalities_subdep = ["EEG_segments", "EEG_windows", "EEG_features"]

def get_split(clip_id):
    return "train" if clip_id < 4 else "test"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def safe_link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)

# --- find blocks that exist in ALL subject-independent modalities ---
available_blocks = None
for mod in modalities_subindep:
    mod_dir = os.path.join(base_dir, mod)
    blocks = [b for b in os.listdir(mod_dir) if os.path.isdir(os.path.join(mod_dir, b))]
    available_blocks = set(blocks) if available_blocks is None else available_blocks & set(blocks)

print("Blocks available in all modalities:", sorted(list(available_blocks)))

# subject-independent split
for mod in modalities_subindep:
    in_dir = os.path.join(base_dir, mod)
    out_train = os.path.join(in_dir, "train")
    out_test = os.path.join(in_dir, "test")
    ensure_dir(out_train)
    ensure_dir(out_test)

    train_list, test_list = [], []

    for block in tqdm(sorted(available_blocks), desc=f"Splitting {mod}"):
        block_path = os.path.join(in_dir, block)
        for fname in os.listdir(block_path):
            if not any(fname.endswith(ext) for ext in [".mp4", ".gif", ".npy", ".txt"]):
                continue
            clip_id = int(fname.split("_")[-1].replace("clip","").split(".")[0]) - 1
            split = get_split(clip_id)
            out_dir = os.path.join(in_dir, split, block)
            ensure_dir(out_dir)
            src = os.path.join(block_path, fname)
            dst = os.path.join(out_dir, fname)
            safe_link(src, dst)
            (train_list if split=="train" else test_list).append(dst)

    with open(os.path.join(in_dir, "train_list.txt"), "w") as f:
        f.write("\n".join(train_list))
    with open(os.path.join(in_dir, "test_list.txt"), "w") as f:
        f.write("\n".join(test_list))

# subject-dependent split
for mod in modalities_subdep:
    in_dir = os.path.join(base_dir, mod)
    out_train = os.path.join(in_dir, "train")
    out_test = os.path.join(in_dir, "test")
    ensure_dir(out_train)
    ensure_dir(out_test)

    train_list, test_list = [], []

    for subj in tqdm(os.listdir(in_dir), desc=f"Splitting {mod}"):
        subj_path = os.path.join(in_dir, subj)
        if not os.path.isdir(subj_path):
            continue

        for block in sorted(available_blocks):
            block_path = os.path.join(subj_path, block)
            if not os.path.exists(block_path):
                continue
            for fname in os.listdir(block_path):
                if not fname.endswith(".npy"):
                    continue
                clip_id = int(fname.split("_")[-1].replace("clip","").split(".")[0]) - 1
                split = get_split(clip_id)
                out_dir = os.path.join(in_dir, split, subj, block)
                ensure_dir(out_dir)
                src = os.path.join(block_path, fname)
                dst = os.path.join(out_dir, fname)
                safe_link(src, dst)
                (train_list if split=="train" else test_list).append(dst)

    with open(os.path.join(in_dir, "train_list.txt"), "w") as f:
        f.write("\n".join(train_list))
    with open(os.path.join(in_dir, "test_list.txt"), "w") as f:
        f.write("\n".join(test_list))
