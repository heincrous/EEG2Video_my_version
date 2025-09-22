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
fre = 200
segment_len = 400
channels = 62

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
    """
    Compute DE features for one EEG clip.
    Input:  [400,62]
    Output: [310] (62 channels × 5 bands)
    """
    features = []
    for ch in range(eeg_segment.shape[1]):  # loop over channels
        f, Pxx = welch(eeg_segment[:, ch], fs=fs, nperseg=fs*2)
        for band in bands.values():
            idx = np.logical_and(f >= band[0], f < band[1])
            band_power = np.sum(Pxx[idx])
            de = np.log(band_power + 1e-8)   # Differential Entropy
            features.append(de)
    return np.array(features, dtype=np.float32)  # [310]

# walk through subjects
for subj in os.listdir(in_dir):
    subj_path = os.path.join(in_dir, subj)
    if not os.path.isdir(subj_path):
        continue

    for block in os.listdir(subj_path):
        block_path = os.path.join(subj_path, block)
        out_block = os.path.join(out_dir, subj, block)
        os.makedirs(out_block, exist_ok=True)

        for fname in tqdm(os.listdir(block_path), desc=f"{subj} {block}"):
            if not fname.endswith(".npy"):
                continue

            seg = np.load(os.path.join(block_path, fname))  # [400,62]
            feats = compute_de_psd(seg, fs=fre)             # [310]
            assert feats.shape == (310,)

            save_path = os.path.join(out_block, fname)
            np.save(save_path, feats)
