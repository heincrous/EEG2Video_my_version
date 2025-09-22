"""
DATA SPLITTING RULES (MASTER FILES)
-----------------------------------

General Rule:
  - For each Block (1–7)
  - For each class (00–39)
  - Clip01–Clip04 → train
  - Clip05        → test

Outputs:
  Each modality has:
    processed/{Modality}/train_list.txt   (MASTER FILE)
    processed/{Modality}/test_list.txt    (MASTER FILE)

Subject-independent modalities (no subX in path):
  - Video_mp4: split independently, full lists
  - Video_gif: split independently, full lists
  - Video_latents: split independently, full lists → ANCHOR for EEG_windows and BLIP_text
  - BLIP_text: split, but KEEP ONLY entries that also exist in Video_latents
  - BLIP_embeddings: split, but KEEP ONLY entries that also exist in EEG_features

Subject-dependent modalities (with subX/ in path):
  - EEG_segments: split independently per subject, full lists
  - EEG_windows: split per subject, but KEEP ONLY entries that also exist in Video_latents
  - EEG_features: split per subject, full lists → ANCHOR for BLIP_embeddings

Master File Principle:
  - The train/test lists written for each modality ARE the master files.
  - Alignment rules are enforced when writing these lists:
      * EEG_windows ↔ Video_latents
      * BLIP_text ↔ Video_latents
      * BLIP_embeddings ↔ EEG_features
  - All other modalities keep full independent splits.
"""

import os
import shutil
from tqdm import tqdm

base_dir = "/content/drive/MyDrive/EEG2Video_data/processed/"

# ---------------- Helpers ----------------
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

def collect_files(in_dir, exts=(".npy", ".mp4", ".gif", ".txt")):
    files = []
    for root, _, fnames in os.walk(in_dir):
        for f in fnames:
            if f.endswith(exts):
                rel = os.path.relpath(os.path.join(root, f), in_dir)
                files.append(rel)
    return sorted(files)

def parse_clip_id(fname):
    return int(fname.split("_")[-1].replace("clip", "").split(".")[0]) - 1

# ---------------- 1. Video_latents (anchor) ----------------
video_latents_dir = os.path.join(base_dir, "Video_latents")
video_latents_files = collect_files(video_latents_dir, exts=(".npy",))

latents_train, latents_test = [], []
for rel in video_latents_files:
    clip_id = parse_clip_id(rel)
    (latents_train if get_split(clip_id)=="train" else latents_test).append(rel)

with open(os.path.join(video_latents_dir, "train_list.txt"), "w") as f:
    f.write("\n".join(latents_train))
with open(os.path.join(video_latents_dir, "test_list.txt"), "w") as f:
    f.write("\n".join(latents_test))

print(f"[Video_latents] train={len(latents_train)} test={len(latents_test)}")

# ---------------- 2. EEG_features (anchor for embeddings) ----------------
eeg_features_dir = os.path.join(base_dir, "EEG_features")
eeg_features_files = collect_files(eeg_features_dir, exts=(".npy",))

features_train, features_test = [], []
for rel in eeg_features_files:
    clip_id = parse_clip_id(rel)
    (features_train if get_split(clip_id)=="train" else features_test).append(rel)

with open(os.path.join(eeg_features_dir, "train_list.txt"), "w") as f:
    f.write("\n".join(features_train))
with open(os.path.join(eeg_features_dir, "test_list.txt"), "w") as f:
    f.write("\n".join(features_test))

print(f"[EEG_features] train={len(features_train)} test={len(features_test)}")

# ---------------- 3. EEG_windows (aligned with Video_latents) ----------------
eeg_windows_dir = os.path.join(base_dir, "EEG_windows")
eeg_windows_files = collect_files(eeg_windows_dir, exts=(".npy",))

windows_train, windows_test = [], []
for rel in eeg_windows_files:
    if rel.replace("sub1"+os.sep, "") in latents_train:
        windows_train.append(rel)
    elif rel.replace("sub1"+os.sep, "") in latents_test:
        windows_test.append(rel)

with open(os.path.join(eeg_windows_dir, "train_list.txt"), "w") as f:
    f.write("\n".join(windows_train))
with open(os.path.join(eeg_windows_dir, "test_list.txt"), "w") as f:
    f.write("\n".join(windows_test))

print(f"[EEG_windows] train={len(windows_train)} test={len(windows_test)}")

# ---------------- 4. BLIP_text (aligned with Video_latents) ----------------
blip_text_dir = os.path.join(base_dir, "BLIP_text")
blip_files = collect_files(blip_text_dir, exts=(".txt",))

text_train, text_test = [], []
for rel in blip_files:
    if rel.replace(".txt", ".npy") in latents_train:
        text_train.append(rel)
    elif rel.replace(".txt", ".npy") in latents_test:
        text_test.append(rel)

with open(os.path.join(blip_text_dir, "train_list.txt"), "w") as f:
    f.write("\n".join(text_train))
with open(os.path.join(blip_text_dir, "test_list.txt"), "w") as f:
    f.write("\n".join(text_test))

print(f"[BLIP_text] train={len(text_train)} test={len(text_test)}")

# ---------------- 5. BLIP_embeddings (aligned with EEG_features) ----------------
blip_embed_dir = os.path.join(base_dir, "BLIP_embeddings")
embed_files = collect_files(blip_embed_dir, exts=(".npy",))

embed_train, embed_test = [], []
for rel in embed_files:
    if rel in [r.replace("sub1"+os.sep, "") for r in features_train]:
        embed_train.append(rel)
    elif rel in [r.replace("sub1"+os.sep, "") for r in features_test]:
        embed_test.append(rel)

with open(os.path.join(blip_embed_dir, "train_list.txt"), "w") as f:
    f.write("\n".join(embed_train))
with open(os.path.join(blip_embed_dir, "test_list.txt"), "w") as f:
    f.write("\n".join(embed_test))

print(f"[BLIP_embeddings] train={len(embed_train)} test={len(embed_test)}")

# ---------------- 6. EEG_segments (independent) ----------------
eeg_segments_dir = os.path.join(base_dir, "EEG_segments")
seg_files = collect_files(eeg_segments_dir, exts=(".npy",))

seg_train, seg_test = [], []
for rel in seg_files:
    clip_id = parse_clip_id(rel)
    (seg_train if get_split(clip_id)=="train" else seg_test).append(rel)

with open(os.path.join(eeg_segments_dir, "train_list.txt"), "w") as f:
    f.write("\n".join(seg_train))
with open(os.path.join(eeg_segments_dir, "test_list.txt"), "w") as f:
    f.write("\n".join(seg_test))

print(f"[EEG_segments] train={len(seg_train)} test={len(seg_test)}")

# ---------------- 7. Video_mp4 & Video_gif (independent) ----------------
for mod, ext in [("Video_mp4", ".mp4"), ("Video_gif", ".gif")]:
    in_dir = os.path.join(base_dir, mod)
    files = collect_files(in_dir, exts=(ext,))
    train, test = [], []
    for rel in files:
        clip_id = parse_clip_id(rel)
        (train if get_split(clip_id)=="train" else test).append(rel)
    with open(os.path.join(in_dir, "train_list.txt"), "w") as f:
        f.write("\n".join(train))
    with open(os.path.join(in_dir, "test_list.txt"), "w") as f:
        f.write("\n".join(test))
    print(f"[{mod}] train={len(train)} test={len(test)}")
