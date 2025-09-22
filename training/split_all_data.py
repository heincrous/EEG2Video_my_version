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

  Example per-modality:
    processed/Video_latents/train_list.txt
    processed/Video_latents/test_list.txt

    processed/EEG_windows/train_list.txt
    processed/EEG_windows/test_list.txt

    processed/BLIP_embeddings/train_list.txt
    processed/BLIP_embeddings/test_list.txt

  Each .txt file contains absolute paths of all samples in that split.

  Additionally, save subject-level arrays before splitting:
    processed/EEG_segments/subX.npy
    Shape = [7,40,5,62,400]

    processed/EEG_features/subX.npy
    Shape = [7,40,5,310]

    processed/EEG_windows/subX.npy
    Shape = [7,40,5,7,62,100]

  During splitting, expand these subject-level files into per-clip
  train/test files according to the 4/1 rule above.
"""

import os
import numpy as np
from tqdm import tqdm

# paths
base_dir = "/content/drive/MyDrive/EEG2Video_data/processed/"
modalities_subindep = ["Video_mp4", "Video_gif", "Video_latents", "BLIP_text", "BLIP_embeddings"]
modalities_subdep = ["EEG_segments", "EEG_windows", "EEG_features"]

# train/test split rule
def get_split(clip_id):
    return "train" if clip_id < 4 else "test"

# ensure dirs exist
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# subject-independent split
for mod in modalities_subindep:
    in_dir = os.path.join(base_dir, mod)
    out_train = os.path.join(base_dir, mod, "train")
    out_test = os.path.join(base_dir, mod, "test")
    ensure_dir(out_train)
    ensure_dir(out_test)

    train_list, test_list = [], []

    for block in tqdm(os.listdir(in_dir), desc=f"Splitting {mod}"):
        block_path = os.path.join(in_dir, block)
        if not os.path.isdir(block_path):
            continue
        for fname in os.listdir(block_path):
            if not any(fname.endswith(ext) for ext in [".mp4", ".gif", ".npy", ".txt"]):
                continue
            parts = fname.split("_")
            clip_id = int(parts[-1].replace("clip","").replace(".mp4","").replace(".gif","").replace(".npy","").replace(".txt","")) - 1
            split = get_split(clip_id)
            out_dir = os.path.join(base_dir, mod, split, block)
            ensure_dir(out_dir)
            src = os.path.join(block_path, fname)
            dst = os.path.join(out_dir, fname)
            if not os.path.exists(dst):
                os.symlink(src, dst)  # save space with symlink
            if split == "train":
                train_list.append(dst)
            else:
                test_list.append(dst)

    # write master lists
    with open(os.path.join(base_dir, mod, "train_list.txt"), "w") as f:
        f.write("\n".join(train_list))
    with open(os.path.join(base_dir, mod, "test_list.txt"), "w") as f:
        f.write("\n".join(test_list))

# subject-dependent split
for mod in modalities_subdep:
    in_dir = os.path.join(base_dir, mod)
    out_train = os.path.join(base_dir, mod, "train")
    out_test = os.path.join(base_dir, mod, "test")
    ensure_dir(out_train)
    ensure_dir(out_test)

    train_list, test_list = [], []

    for subj in tqdm(os.listdir(in_dir), desc=f"Splitting {mod}"):
        subj_path = os.path.join(in_dir, subj)
        if not os.path.isdir(subj_path):
            continue

        # case 1: subject-level array before splitting
        subj_file = os.path.join(subj_path + ".npy")
        if os.path.exists(subj_file):
            arr = np.load(subj_file)  # EEG_segments: [7,40,5,62,400], EEG_features: [7,40,5,310], EEG_windows: [7,40,5,7,62,100]
            for block_id in range(7):
                for class_id in range(40):
                    for clip_id in range(5):
                        split = get_split(clip_id)
                        out_dir = os.path.join(base_dir, mod, split, subj, f"Block{block_id+1}")
                        ensure_dir(out_dir)
                        fname = f"class{class_id:02d}_clip{clip_id+1:02d}.npy"
                        save_path = os.path.join(out_dir, fname)
                        np.save(save_path, arr[block_id, class_id, clip_id])
                        if split == "train":
                            train_list.append(save_path)
                        else:
                            test_list.append(save_path)
        else:
            # case 2: already expanded into per-clip files
            for block in os.listdir(subj_path):
                block_path = os.path.join(subj_path, block)
                for fname in os.listdir(block_path):
                    if not fname.endswith(".npy"):
                        continue
                    parts = fname.split("_")
                    clip_id = int(parts[-1].replace("clip","").replace(".npy","")) - 1
                    split = get_split(clip_id)
                    out_dir = os.path.join(base_dir, mod, split, subj, block)
                    ensure_dir(out_dir)
                    src = os.path.join(block_path, fname)
                    dst = os.path.join(out_dir, fname)
                    if not os.path.exists(dst):
                        os.symlink(src, dst)
                    if split == "train":
                        train_list.append(dst)
                    else:
                        test_list.append(dst)

    # write master lists
    with open(os.path.join(base_dir, mod, "train_list.txt"), "w") as f:
        f.write("\n".join(train_list))
    with open(os.path.join(base_dir, mod, "test_list.txt"), "w") as f:
        f.write("\n".join(test_list))