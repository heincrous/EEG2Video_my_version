import os
import numpy as np
from DE_PSD import DE_PSD
from tqdm import tqdm

# Parameters
fre = 200
in_dir = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments/"
out_de_dir = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_DE/"
out_psd_dir = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_PSD/"

# Ensure base output dirs exist
os.makedirs(out_de_dir, exist_ok=True)
os.makedirs(out_psd_dir, exist_ok=True)

# Subjects
subjects = [s for s in sorted(os.listdir(in_dir)) if os.path.isdir(os.path.join(in_dir, s))]

print("\nAvailable subjects:")
for idx, subj in enumerate(subjects):
    print(f"{idx}: {subj}")

choices = input("\nEnter subject indices to process (comma separated): ")
choices = [int(c.strip()) for c in choices.split(",") if c.strip().isdigit()]

for idx in choices:
    subj = subjects[idx]
    subj_path = os.path.join(in_dir, subj)

    print(f"\nProcessing subject {subj}...")

    # Make subject-level output folders
    subj_de_dir = os.path.join(out_de_dir, subj)
    subj_psd_dir = os.path.join(out_psd_dir, subj)
    os.makedirs(subj_de_dir, exist_ok=True)
    os.makedirs(subj_psd_dir, exist_ok=True)

    for block in sorted(os.listdir(subj_path)):
        block_path = os.path.join(subj_path, block)
        if not os.path.isdir(block_path):
            continue
        print("block:", block)

        # Make block-level output folders
        block_de_dir = os.path.join(subj_de_dir, block)
        block_psd_dir = os.path.join(subj_psd_dir, block)
        os.makedirs(block_de_dir, exist_ok=True)
        os.makedirs(block_psd_dir, exist_ok=True)

        # Process all class/clip files in this block
        files = sorted([f for f in os.listdir(block_path) if f.endswith(".npy")])
        for f in tqdm(files, desc=f"{subj}/{block}"):
            seg = np.load(os.path.join(block_path, f))  # shape [400, 62]
            seg = seg.T  # → (62, 400)
            de, psd = DE_PSD(seg, fre, 2)

            # Save per-clip outputs
            np.save(os.path.join(block_de_dir, f), de)   # shape (62, 5)
            np.save(os.path.join(block_psd_dir, f), psd) # shape (62, 5)

print("\nProcessing complete. Hierarchical DE/PSD files saved.")
