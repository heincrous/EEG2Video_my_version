# ==========================================
# SEGMENT EEG INTO CLIP-ALIGNED SEGMENTS
# ==========================================
# Input:
#   EEG2Video_data/raw/EEG/subX.npy
#   Each subject file shape = [7, 62, 104000]
#   Sampling rate = 200 Hz
#   Channels = 62
#   Block length = 40 × (3 s hint + 5 × 2 s clips) = 520 s
#
# Process:
#   - Use GT_LABEL (7 × 40) to align EEG with video class order
#   - For each class slot: skip 3 s hint (600 samples)
#   - Extract 5 × 2 s clips (5 × 400 samples)
#   - Collect into subject array: [7, 40, 5, 62, 400]
#
# Output:
#   EEG2Video_data/processed/EEG_segments/subX.npy
# ==========================================

# === Standard libraries ===
import os

# === Third-party libraries ===
import numpy as np
from tqdm import tqdm

# === Repo imports ===
from core_files.gt_label import GT_LABEL   # shape (7,40), values 0–39


# ==========================================
# CONFIGURATION (EDITABLE PARAMETERS)
# ==========================================
config = {
    "sampling_rate":    200,   # Hz
    "segment_seconds":  2,     # seconds per EEG segment
    "hint_seconds":     3,     # seconds of hint before each class
    "channels":         62,
    "clips_per_class":  5,

    "drive_root":       "/content/drive/MyDrive/EEG2Video_data"
}

# Derived values
config["segment_len"] = config["segment_seconds"] * config["sampling_rate"]
config["hint_len"]    = config["hint_seconds"] * config["sampling_rate"]

raw_dir = os.path.join(config["drive_root"], "raw", "EEG")
out_dir = os.path.join(config["drive_root"], "processed", "EEG_segments")
os.makedirs(out_dir, exist_ok=True)


# ==========================================
# Helper: list subject files
# ==========================================
def list_subject_files(directory):
    return sorted([f for f in os.listdir(directory) if f.endswith(".npy")])


# ==========================================
# Main processing loop
# ==========================================
all_files = list_subject_files(raw_dir)
print("Available subject files:")
for i, f in enumerate(all_files):
    print(f"{i}: {f}")

chosen = input("Enter indices of subjects to process (comma separated, 'all' for all): ").strip()
if chosen.lower() == "all":
    selected_files = all_files
else:
    idxs = [int(x) for x in chosen.split(",")]
    selected_files = [all_files[i] for i in idxs]

for subj_file in selected_files:
    subj_name = subj_file.replace(".npy", "")
    eeg_data = np.load(os.path.join(raw_dir, subj_file))  # shape [7, 62, 104000]

    subj_array = np.zeros(
        (7, 40, config["clips_per_class"], config["channels"], config["segment_len"]),
        dtype=np.float32,
    )

    for block_id in range(7):
        now_data = eeg_data[block_id]  # [62, T_block]
        l = 0

        for order_idx in tqdm(range(40), desc=f"{subj_name} Block {block_id+1}"):
            true_class = GT_LABEL[block_id, order_idx]

            # skip hint
            l += config["hint_len"]

            for clip_id in range(config["clips_per_class"]):
                start_idx = l
                end_idx   = l + config["segment_len"]

                if end_idx > now_data.shape[1]:
                    raise ValueError(
                        f"{subj_name} Block {block_id+1}: insufficient samples "
                        f"for class {true_class}, clip {clip_id}. "
                        f"Available {now_data.shape[1]}, needed {end_idx}"
                    )

                eeg_slice = now_data[:, start_idx:end_idx]  # [62, segment_len]
                subj_array[block_id, true_class, clip_id] = eeg_slice
                l = end_idx

        expected_len = (config["hint_len"] + config["clips_per_class"] * config["segment_len"]) * 40
        if l != expected_len:
            print(f"Warning: {subj_name} Block {block_id+1} ended at {l}, expected {expected_len}.")

    out_path = os.path.join(out_dir, f"{subj_name}.npy")
    np.save(out_path, subj_array)
    print(f"Saved {subj_name} → {out_path}, shape {subj_array.shape}")

print("\nProcessing complete.")
