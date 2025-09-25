"""
EXTRACT DE/PSD FEATURES (SUBJECT-LEVEL ARRAYS)
----------------------------------------------
Input:
  processed/EEG_segments/subX.npy
    Shape = [7,40,5,400,62]

Process:
  - For each clip [400,62], compute DE and PSD features using DE_PSD().
  - Each result = [62,5] (channels × frequency bands).
  - Preserve subject-level hierarchy.

Output:
  processed/EEG_DE/subX.npy   shape [7,40,5,62,5]
  processed/EEG_PSD/subX.npy  shape [7,40,5,62,5]
"""

import os
import numpy as np
from DE_PSD import DE_PSD
from tqdm import tqdm

# parameters
fre = 200
in_dir = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments/"
out_de_dir = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_DE/"
out_psd_dir = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_PSD/"

os.makedirs(out_de_dir, exist_ok=True)
os.makedirs(out_psd_dir, exist_ok=True)

# subjects (now stored as subject-level .npy arrays)
subjects = [f for f in sorted(os.listdir(in_dir)) if f.endswith(".npy")]

print("\nAvailable subjects:")
for idx, subj in enumerate(subjects):
    print(f"{idx}: {subj}")

choices = input("\nEnter subject indices to process (comma separated): ")
choices = [int(c.strip()) for c in choices.split(",") if c.strip().isdigit()]

for idx in choices:
    subj_file = subjects[idx]
    subj_name = subj_file.replace(".npy", "")
    subj_path = os.path.join(in_dir, subj_file)

    print(f"\nProcessing {subj_name}...")

    # load subject EEG segments
    data = np.load(subj_path)  # [7,40,5,400,62]

    # allocate outputs
    de_array  = np.zeros((7,40,5,62,5), dtype=np.float32)
    psd_array = np.zeros((7,40,5,62,5), dtype=np.float32)

    for b in range(7):
        for c in tqdm(range(40), desc=f"{subj_name} Block{b+1}"):
            for k in range(5):
                seg = data[b,c,k]        # [400,62]
                seg = seg.T              # [62,400]
                de, psd = DE_PSD(seg, fre, 2)
                de_array[b,c,k]  = de    # [62,5]
                psd_array[b,c,k] = psd   # [62,5]

    # save subject-level arrays
    np.save(os.path.join(out_de_dir,  f"{subj_name}.npy"), de_array)
    np.save(os.path.join(out_psd_dir, f"{subj_name}.npy"), psd_array)

    print(f"Saved DE → {os.path.join(out_de_dir,  subj_name+'.npy')} {de_array.shape}")
    print(f"Saved PSD → {os.path.join(out_psd_dir, subj_name+'.npy')} {psd_array.shape}")

print("\nProcessing complete. Subject-level DE/PSD files saved.")
