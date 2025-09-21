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

# collect all clip IDs grouped by (BlockX, classYY)
clips_by_class = {}
for modality in MODALITIES:
    modality_dir = os.path.join(BASE_DIR, modality)
    files = glob(os.path.join(modality_dir, "Block*/*.npy")) + glob(os.path.join(modality_dir, "Block*/*.npz"))
    for f in files:
        rel = os.path.relpath(f, modality_dir)       # e.g. Block1/class23_clip04.npy
        block = rel.split("/")[0]                    # Block1
        base = os.path.splitext(os.path.basename(rel))[0]  # class23_clip04
        class_id = base.split("_")[0]                # class23
        key = (block, class_id)
        clips_by_class.setdefault(key, set()).add(os.path.splitext(rel)[0])  # BlockX/classYY_clipZZ

train_set, test_set = [], []

# enforce 4 train / 1 test
for key, clips in clips_by_class.items():
    clips = sorted(list(clips))   # clip01 … clip05
    if len(clips) != 5:
        print(f"Warning: {key} has {len(clips)} clips (expected 5)")
    random.shuffle(clips)
    train_set.extend(clips[:4])
    test_set.extend(clips[4:])

def copy_clip_files(clip_id, split):
    block = clip_id.split("/")[0]
    for modality in MODALITIES:
        modality_dir = os.path.join(BASE_DIR, modality)
        dest_dir = os.path.join(SAVE_DIR, split, modality, block)
        os.makedirs(dest_dir, exist_ok=True)
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

print(f"Split done. Train clips: {len(train_set)}, Test clips: {len(test_set)}")
print(f"Manifest saved to {os.path.join(SAVE_DIR, 'split_manifest.json')}")
