"""
EXTRACT DE/PSD FEATURES (SUBJECT-LEVEL ARRAYS)
----------------------------------------------
Input:
  processed/EEG_segments/subX.npy
    Shape = [7,40,5,62,400]

Process:
  - For each clip [62,400], split into two 1s windows ([62,200] each).
  - For each 1s window, compute DE and PSD features using DE_PSD().
  - Each window result = [62,5] (channels × frequency bands).
  - Stack into [2,62,5] per clip.
  - Preserve subject-level hierarchy.

Output:
  processed/EEG_DE_1per1s/subX.npy   shape [7,40,5,2,62,5]
  processed/EEG_PSD_1per1s/subX.npy  shape [7,40,5,2,62,5]
"""

import os
import numpy as np
from DE_PSD import DE_PSD
from tqdm import tqdm

# parameters
fre = 200
in_dir = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments/"
out_de_dir = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_DE_1per1s/"
out_psd_dir = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_PSD_1per1s/"

os.makedirs(out_de_dir, exist_ok=True)
os.makedirs(out_psd_dir, exist_ok=True)

# subjects (now stored as subject-level .npy arrays)
subjects = [f for f in sorted(os.listdir(in_dir)) if f.endswith(".npy")]

print("\nAvailable subjects:")
for idx, subj in enumerate(subjects):
    print(f"{idx}: {subj}")

choices = input("\nEnter subject indices to process (comma separated, 'all' for all): ").strip()
if choices.lower() == "all":
    selected_idxs = list(range(len(subjects)))
else:
    selected_idxs = [int(c.strip()) for c in choices.split(",") if c.strip().isdigit()]

for idx in selected_idxs:
    subj_file = subjects[idx]
    subj_name = subj_file.replace(".npy", "")
    subj_path = os.path.join(in_dir, subj_file)

    print(f"\nProcessing {subj_name}...")

    # load subject EEG segments
    data = np.load(subj_path)  # [7,40,5,62,400]

    # allocate outputs
    de_array  = np.zeros((7,40,5,2,62,5), dtype=np.float32)
    psd_array = np.zeros((7,40,5,2,62,5), dtype=np.float32)

    for b in range(7):
        for c in tqdm(range(40), desc=f"{subj_name} Block{b+1}"):
            for k in range(5):
                seg = data[b,c,k]  # [62,400]

                # split into two 1-second windows
                seg1 = seg[:, :200]
                seg2 = seg[:, 200:]

                de1, psd1 = DE_PSD(seg1, fre, 1)  # [62,5]
                de2, psd2 = DE_PSD(seg2, fre, 1)  # [62,5]

                de_array[b,c,k,0] = de1
                de_array[b,c,k,1] = de2
                psd_array[b,c,k,0] = psd1
                psd_array[b,c,k,1] = psd2

    # save subject-level arrays
    out_de_path  = os.path.join(out_de_dir,  f"{subj_name}.npy")
    out_psd_path = os.path.join(out_psd_dir, f"{subj_name}.npy")

    np.save(out_de_path, de_array)
    np.save(out_psd_path, psd_array)

    print(f"Saved DE  → {out_de_path} {de_array.shape}")
    print(f"Saved PSD → {out_psd_path} {psd_array.shape}")

print("\nProcessing complete.")
