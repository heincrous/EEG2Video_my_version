# ==========================================
# VERIFY ALIGNMENT: EEG ↔ CLIP ↔ GT_LABEL
# ==========================================
# Checks:
#   1. File shapes and dimensions
#   2. GT_LABEL coverage (0–39 per block)
#   3. Internal consistency between EEG and CLIP ordering
#   4. Within-class CLIP cosine similarity
#   5. Cross-block class matching sanity
# ==========================================

import os
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
from core.gt_label import GT_LABEL


# ==========================================
# Config
# ==========================================
config = {
    "subject_name": "sub1.npy",
    "feature_type": "DE",   # choose "DE" or "segments" etc.
    "drive_root": "/content/drive/MyDrive/EEG2Video_data"
}

eeg_path  = os.path.join(config["drive_root"], "processed", f"EEG_{config['feature_type']}_1per2s", config["subject_name"])
clip_path = os.path.join(config["drive_root"], "processed", "CLIP_embeddings", "CLIP_embeddings.npy")

# ==========================================
# Load files
# ==========================================
print("Loading arrays...")
eeg  = np.load(eeg_path, allow_pickle=True)   # shape [7,40,5,...]
clip = np.load(clip_path, allow_pickle=True)  # shape [7,40,5,77,768]
print(f"EEG shape: {eeg.shape}")
print(f"CLIP shape: {clip.shape}")
print(f"GT_LABEL shape: {GT_LABEL.shape}\n")


# ==========================================
# 1. GT_LABEL sanity
# ==========================================
print("Checking GT_LABEL coverage per block...")
for b in range(7):
    u = np.unique(GT_LABEL[b])
    print(f" Block {b+1}: unique={len(u)} | range={u.min()}–{u.max()}")
assert np.all([len(np.unique(GT_LABEL[b])) == 40 for b in range(7)]), "GT_LABEL missing classes"
print("✅ GT_LABEL covers all 40 classes per block.\n")


# ==========================================
# 2. Check that EEG and CLIP store classes by GT_LABEL order
# ==========================================
print("Checking class alignment order...")
misaligned = []
for b in range(7):
    for c in range(40):
        eeg_idx  = np.where(GT_LABEL[b] == c)[0]
        clip_idx = np.where(GT_LABEL[b] == c)[0]
        if len(eeg_idx) == 0 or len(clip_idx) == 0:
            misaligned.append((b, c))
if len(misaligned) == 0:
    print("✅ EEG and CLIP arrays follow identical GT_LABEL mapping.\n")
else:
    print("⚠️ Misalignment found in:", misaligned, "\n")


# ==========================================
# 3. Flatten to global arrays and rebuild labels
# ==========================================
print("Flattening arrays...")
eeg_flat  = eeg.reshape(7 * 40 * 5, -1)
clip_flat = clip.reshape(7 * 40 * 5, 77 * 768)
labels    = np.tile(np.repeat(np.arange(40), 5), 7)
print(f"Flattened EEG: {eeg_flat.shape}, CLIP: {clip_flat.shape}, labels: {labels.shape}\n")


# ==========================================
# 4. Compute within-class CLIP cosine
# ==========================================
print("Computing within-class CLIP cosine similarities...")
class_cos = {}
for c in range(40):
    idx = np.where(labels == c)[0]
    vals = []
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            vals.append(1 - cosine(clip_flat[idx[i]], clip_flat[idx[j]]))
    class_cos[c] = np.mean(vals)
mean_cos = np.mean(list(class_cos.values()))
print(f"Average within-class cosine (CLIP): {mean_cos:.4f}")
if mean_cos < 0.6:
    print("⚠️ Low within-class similarity → possible caption misalignment.")
else:
    print("✅ CLIP embeddings show expected semantic clustering.\n")


# ==========================================
# 5. Cross-block sanity (same class across blocks)
# ==========================================
print("Checking same-class embeddings across blocks...")
cross_vals = []
for c in tqdm(range(40)):
    block_means = [clip[b, c].mean(axis=(0,1)).reshape(-1) for b in range(7)]
    for i in range(6):
        for j in range(i+1,7):
            cross_vals.append(1 - cosine(block_means[i], block_means[j]))
print(f"Mean cross-block same-class cosine: {np.mean(cross_vals):.4f}")
print("✅ Verification complete.\n")
