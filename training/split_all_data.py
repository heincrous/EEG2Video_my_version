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
    - Video_mp4  (all processed blocks)
    - Video_gif  (all processed blocks)
    - Video_latents  (all processed blocks)
    - BLIP_text  (must align with Video_latents → only include blocks where latents exist)
    - BLIP_embeddings  (all processed blocks, must align with EEG_features)

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

Special Notes:
  - BLIP_text aligns dynamically with Video_latents
  - BLIP_embeddings + EEG features/windows/segments: use all processed blocks
  - Videos, GIFs, latents: split everything present
"""

import os
import shutil
from tqdm import tqdm

base_dir = "/content/drive/MyDrive/EEG2Video_data/processed/"
modalities_videos = ["Video_mp4", "Video_gif", "Video_latents"]
modalities_text   = ["BLIP_text"]
modalities_embed  = ["BLIP_embeddings"]
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

# ---------------- Subject-independent: Videos ----------------
for mod in modalities_videos:
    in_dir = os.path.join(base_dir, mod)
    if not os.path.exists(in_dir):
        continue
    blocks = [b for b in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, b))]
    out_train = os.path.join(in_dir, "train")
    out_test = os.path.join(in_dir, "test")
    ensure_dir(out_train); ensure_dir(out_test)

    train_list, test_list = [], []
    for block in tqdm(sorted(blocks), desc=f"Splitting {mod}"):
        block_path = os.path.join(in_dir, block)
        for fname in os.listdir(block_path):
            if not any(fname.endswith(ext) for ext in [".mp4", ".gif", ".npy"]):
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

# ---------------- Subject-independent: BLIP text (align with Video_latents) ----------------
video_latents_dir = os.path.join(base_dir, "Video_latents")
video_blocks = [b for b in os.listdir(video_latents_dir) if os.path.isdir(os.path.join(video_latents_dir, b))]

for mod in modalities_text:
    in_dir = os.path.join(base_dir, mod)
    if not os.path.exists(in_dir):
        continue
    blocks = [b for b in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, b)) and b in video_blocks]
    out_train = os.path.join(in_dir, "train")
    out_test = os.path.join(in_dir, "test")
    ensure_dir(out_train); ensure_dir(out_test)

    train_list, test_list = [], []
    for block in tqdm(sorted(blocks), desc=f"Splitting {mod}"):
        block_path = os.path.join(in_dir, block)
        for fname in os.listdir(block_path):
            if not fname.endswith(".txt"):
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

# ---------------- Subject-independent: BLIP embeddings ----------------
for mod in modalities_embed:
    in_dir = os.path.join(base_dir, mod)
    if not os.path.exists(in_dir):
        continue
    blocks = [b for b in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, b))]
    out_train = os.path.join(in_dir, "train")
    out_test = os.path.join(in_dir, "test")
    ensure_dir(out_train); ensure_dir(out_test)

    train_list, test_list = [], []
    for block in tqdm(sorted(blocks), desc=f"Splitting {mod}"):
        block_path = os.path.join(in_dir, block)
        for fname in os.listdir(block_path):
            if not fname.endswith(".npy"):
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

# ---------------- Subject-dependent modalities ----------------
for mod in modalities_subdep:
    in_dir = os.path.join(base_dir, mod)
    if not os.path.exists(in_dir):
        continue
    out_train = os.path.join(in_dir, "train")
    out_test = os.path.join(in_dir, "test")
    ensure_dir(out_train); ensure_dir(out_test)

    train_list, test_list = [], []
    for subj in tqdm(os.listdir(in_dir), desc=f"Splitting {mod}"):
        subj_path = os.path.join(in_dir, subj)
        if not os.path.isdir(subj_path):
            continue
        blocks = [b for b in os.listdir(subj_path) if os.path.isdir(os.path.join(subj_path, b))]
        for block in sorted(blocks):
            block_path = os.path.join(subj_path, block)
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

# ---------------- Post-check summary ----------------
def count_lines(path):
    return sum(1 for _ in open(path)) if os.path.exists(path) else 0

for mod in modalities_videos + modalities_text + modalities_embed + modalities_subdep:
    mod_path = os.path.join(base_dir, mod)
    train_count = count_lines(os.path.join(mod_path, "train_list.txt"))
    test_count = count_lines(os.path.join(mod_path, "test_list.txt"))
    if train_count or test_count:
        print(f"{mod}: train={train_count}, test={test_count}, total={train_count+test_count}")
