import os
import random
import shutil
import json
from glob import glob

# CONFIG
BASE_DIR = "/content/drive/MyDrive/EEG2Video_data/processed"
MODALITIES = ["EEG_segments", "Video_latents", "BLIP_embeddings"]
SAVE_DIR = os.path.join(BASE_DIR, "Split_4train1test")
SEED = 42
random.seed(SEED)

# make split dirs
for split in ["train", "test"]:
    for modality in MODALITIES:
        os.makedirs(os.path.join(SAVE_DIR, split, modality), exist_ok=True)

# --- Step 1: build mapping of (Block, class) -> list of clip IDs ---
clips_by_class = {}

# always scan video latents (shared across subjects) to build the class/clip keys
video_files = glob(os.path.join(BASE_DIR, "Video_latents", "Block*/*.npy"))
for f in video_files:
    rel = os.path.relpath(f, os.path.join(BASE_DIR, "Video_latents"))
    block = rel.split("/")[0]  # Block1
    base = os.path.splitext(os.path.basename(rel))[0]  # class23_clip04
    class_id = base.split("_")[0]  # class23
    key = (block, class_id)
    clips_by_class.setdefault(key, []).append(os.path.splitext(rel)[0])  # Block1/class23_clip04

# --- Step 2: split 4/1 per class ---
train_set, test_set = [], []
for key, clips in clips_by_class.items():
    clips = sorted(clips)  # clip01 … clip05
    if len(clips) != 5:
        print(f"Warning: {key} has {len(clips)} clips (expected 5)")
    random.shuffle(clips)
    train_set.extend(clips[:4])
    test_set.extend(clips[4:])

# --- Step 3: copy function ---
def copy_clip_files(clip_id, split):
    block = clip_id.split("/")[0]  # BlockX
    base = os.path.basename(clip_id)  # classYY_clipZZ

    for modality in MODALITIES:
        modality_dir = os.path.join(BASE_DIR, modality)

        if modality == "EEG_segments":
            # EEG has subject folders
            eeg_subjects = sorted(os.listdir(modality_dir))
            for subj in eeg_subjects:
                src = os.path.join(modality_dir, subj, block, base + ".npy")
                if os.path.exists(src):
                    dest_dir = os.path.join(SAVE_DIR, split, modality, subj, block)
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy(src, os.path.join(dest_dir, os.path.basename(src)))
        else:
            # Video_latents / BLIP_embeddings
            src = os.path.join(modality_dir, clip_id + ".npy")
            if os.path.exists(src):
                dest_dir = os.path.join(SAVE_DIR, split, modality, block)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy(src, os.path.join(dest_dir, os.path.basename(src)))

# --- Step 4: execute copy ---
for cid in train_set:
    copy_clip_files(cid, "train")
for cid in test_set:
    copy_clip_files(cid, "test")

# --- Step 5: save manifest ---
manifest = {"train": train_set, "test": test_set}
with open(os.path.join(SAVE_DIR, "split_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Split done. Train clips: {len(train_set)}, Test clips: {len(test_set)}")
print(f"Manifest saved to {os.path.join(SAVE_DIR, 'split_manifest.json')}")