"""
EXTRACT DE/PSD FEATURES FROM EEG SEGMENTS
------------------------------------------
Input:
  processed/EEG_segments/subX/BlockY/classYY_clipZZ.npy
    Shape = [400,62]

Process:
  - Compute Power Spectral Density (Welch/FFT).
  - Integrate band power across:
      δ (1–4 Hz), θ (4–8 Hz), α (8–14 Hz),
      β (14–31 Hz), γ (31–50 Hz).
  - Convert each band to Differential Entropy (DE).
  - 62 channels × 5 bands = 310 features.

Output:
  processed/EEG_features/subX/BlockY/classYY_clipZZ.npy
    Shape = [310]
"""

# Pseudocode steps:
# 1. Load [400,62] EEG segment
# 2. For each channel, compute PSD
# 3. Integrate into 5 bands
# 4. Convert band variance → DE
# 5. Save 310-dim vector as .npy

import os
import numpy as np
from scipy.signal import welch
from tqdm import tqdm

# parameters
fs = 200
channels = 62
batch_size = 64  # adjust based on RAM

# frequency bands (Hz)
bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 14),
    "beta":  (14, 31),
    "gamma": (31, 50)
}

# paths
in_dir = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments/"
out_dir = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_features/"
os.makedirs(out_dir, exist_ok=True)

def compute_de_psd(eeg_segment, fs=200):
    """Compute DE features for one EEG clip [400,62] → [310]."""
    features = []
    for ch in range(eeg_segment.shape[1]):
        f, Pxx = welch(eeg_segment[:, ch], fs=fs, nperseg=fs*2)
        for band in bands.values():
            idx = np.logical_and(f >= band[0], f < band[1])
            band_power = np.sum(Pxx[idx])
            de = np.log(band_power + 1e-8)
            features.append(de)
    return np.array(features, dtype=np.float32)

# collect subjects
subjects = [s for s in sorted(os.listdir(in_dir)) if os.path.isdir(os.path.join(in_dir, s))]

print("\nAvailable subjects:")
for idx, subj in enumerate(subjects):
    print(f"{idx}: {subj}")

choices = input("\nEnter subject indices to process (comma separated): ")
choices = [int(c.strip()) for c in choices.split(",") if c.strip().isdigit()]

# process selected subjects
for idx in choices:
    subj = subjects[idx]
    subj_path = os.path.join(in_dir, subj)

    for block in sorted(os.listdir(subj_path)):
        block_path = os.path.join(subj_path, block)
        if not os.path.isdir(block_path):
            continue

        out_block = os.path.join(out_dir, subj, block)
        os.makedirs(out_block, exist_ok=True)

        fnames = [f for f in os.listdir(block_path) if f.endswith(".npy")]
        for i in tqdm(range(0, len(fnames), batch_size), desc=f"{subj} {block}"):
            batch = fnames[i:i+batch_size]

            # load batch into memory
            segs = [np.load(os.path.join(block_path, f)) for f in batch]

            # compute features for batch
            feats_batch = [compute_de_psd(seg, fs=fs) for seg in segs]

            # save batch
            for fname, feats in zip(batch, feats_batch):
                save_path = os.path.join(out_block, fname)
                np.save(save_path, feats)

print("\nProcessing complete.")
