# ==========================================
# EXTRACT DE/PSD FEATURES (1 per 2s segment)
# ==========================================
# Input:
#   EEG2Video_data/processed/EEG_segments/subX.npy
#   Shape = [7, 40, 5, 62, 400]
#
# Process:
#   - For each 2 s clip [62, 400], compute DE and PSD features using DE_PSD().
#   - Each result = [62, 5] (channels × frequency bands).
#   - Stack into [7, 40, 5, 62, 5].
#
# Output:
#   EEG2Video_data/processed/EEG_DE_1per2s/subX.npy   [7, 40, 5, 62, 5]
#   EEG2Video_data/processed/EEG_PSD_1per2s/subX.npy  [7, 40, 5, 62, 5]
# ==========================================

import os
import numpy as np
from tqdm import tqdm
from .DE_PSD import DE_PSD

# ==========================================
# CONFIG
# ==========================================
config = {
    "sampling_rate": 200,  # Hz
    "segment_len":   400,  # samples (2 s @ 200 Hz)
    "drive_root":    "/content/drive/MyDrive/EEG2Video_data"
}

# paths
in_dir      = os.path.join(config["drive_root"], "processed", "EEG_segments")
out_de_dir  = os.path.join(config["drive_root"], "processed", "EEG_DE_1per2s")
out_psd_dir = os.path.join(config["drive_root"], "processed", "EEG_PSD_1per2s")

os.makedirs(out_de_dir, exist_ok=True)
os.makedirs(out_psd_dir, exist_ok=True)

def list_subject_files(directory):
    return sorted([f for f in os.listdir(directory) if f.endswith(".npy")])

# ==========================================
# Main loop
# ==========================================
subjects = list_subject_files(in_dir)
print("\nAvailable subject files:")
for i, subj in enumerate(subjects):
    print(f"[{i}] {subj}")

chosen = input("\nEnter indices of subjects to process (comma separated, 'all' for all): ").strip()
if chosen.lower() == "all":
    selected_idxs = list(range(len(subjects)))
else:
    selected_idxs = [int(c.strip()) for c in chosen.split(",") if c.strip().isdigit()]

for idx in selected_idxs:
    subj_file = subjects[idx]
    subj_name = subj_file.replace(".npy", "")
    subj_path = os.path.join(in_dir, subj_file)

    print(f"\nProcessing {subj_name}...")
    data = np.load(subj_path)  # [7, 40, 5, 62, 400]

    de_array  = np.zeros((7, 40, 5, 62, 5), dtype=np.float32)
    psd_array = np.zeros((7, 40, 5, 62, 5), dtype=np.float32)

    for b in range(7):
        for c in tqdm(range(40), desc=f"{subj_name} Block {b+1}"):
            for k in range(5):
                seg = data[b, c, k]  # [62, 400]
                de, psd = DE_PSD(seg, config["sampling_rate"], 2)  # 2-second window
                de_array[b, c, k]  = de
                psd_array[b, c, k] = psd

    out_de_path  = os.path.join(out_de_dir,  f"{subj_name}.npy")
    out_psd_path = os.path.join(out_psd_dir, f"{subj_name}.npy")
    np.save(out_de_path, de_array)
    np.save(out_psd_path, psd_array)

    print(f"Saved DE  → {out_de_path}, shape {de_array.shape}")
    print(f"Saved PSD → {out_psd_path}, shape {psd_array.shape}")

print("\nProcessing complete.")
