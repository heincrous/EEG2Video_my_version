# ==========================================
# VERIFY ALIGNMENT: EEG ↔ CLIP ↔ GT_LABEL
# WITH WITHIN vs BETWEEN CLASS COSINE
# ==========================================

import os, numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
from core.gt_label import GT_LABEL


# ==========================================
# Config
# ==========================================
cfg = {
    "subject_name": "sub1.npy",
    "feature_type": "DE",
    "drive_root": "/content/drive/MyDrive/EEG2Video_data"
}

eeg_path  = os.path.join(cfg["drive_root"], "processed", f"EEG_{cfg['feature_type']}_1per2s", cfg["subject_name"])
clip_path = os.path.join(cfg["drive_root"], "processed", "CLIP_embeddings", "CLIP_embeddings.npy")


# ==========================================
# Load
# ==========================================
clip = np.load(clip_path, allow_pickle=True)  # [7,40,5,77,768]
print("Loaded CLIP embeddings:", clip.shape)

# Flatten
clip_flat = clip.reshape(7 * 40 * 5, 77 * 768)
labels = np.tile(np.repeat(np.arange(40), 5), 7)
print("Flattened CLIP:", clip_flat.shape, "Labels:", labels.shape)


# ==========================================
# Within-class cosine
# ==========================================
print("\nComputing within-class cosine...")
within_vals = []
for c in tqdm(range(40)):
    idx = np.where(labels == c)[0]
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            within_vals.append(1 - cosine(clip_flat[idx[i]], clip_flat[idx[j]]))
within_mean = np.mean(within_vals)
print(f"Within-class mean cosine: {within_mean:.4f}")


# ==========================================
# Between-class cosine
# ==========================================
print("Computing between-class cosine (sampled)...")
np.random.seed(0)
n_pairs = 2000
between_vals = []
for _ in tqdm(range(n_pairs)):
    i, j = np.random.choice(len(labels), 2, replace=False)
    if labels[i] != labels[j]:
        between_vals.append(1 - cosine(clip_flat[i], clip_flat[j]))
between_mean = np.mean(between_vals)
print(f"Between-class mean cosine: {between_mean:.4f}")


# ==========================================
# Report
# ==========================================
print("\n===== COSINE SUMMARY =====")
print(f"Within-class  : {within_mean:.4f}")
print(f"Between-class : {between_mean:.4f}")
print(f"Separation Δ  : {within_mean - between_mean:.4f}")
if within_mean > between_mean:
    print("✅ Classes are separable — alignment likely correct.")
else:
    print("⚠️ No separation — possible misalignment or bad captions.")
