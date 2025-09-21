import os
import re
import numpy as np

# CONFIG
RAW_EEG_DIR = "/content/drive/MyDrive/EEG2Video_data/raw/EEG/"
BLIP_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_embeddings/"
SAVE_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments/"
LOG_FILE = "/content/drive/MyDrive/EEG2Video_data/processed/processed_log.txt"
PROCESS_TAG = "[EEG]"

os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------------------------------------------------
# Utility: load/save processed log
# --------------------------------------------------------------------
def load_processed_log():
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, "r") as f:
        return set(line.strip() for line in f.readlines())

def update_processed_log(entry):
    if entry not in load_processed_log():
        with open(LOG_FILE, "a") as f:
            f.write(entry + "\n")

# --------------------------------------------------------------------
# Utility: natural sort (so class01, class02 ... not class1, class10)
# --------------------------------------------------------------------
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

# --------------------------------------------------------------------
# EEG loading
# Each subject file shape: (7, 62, 104000) = blocks × channels × timepoints
# One block = 200 clips × 2s per clip = 400s = 400*200Hz = 80,000 samples
# Preprocessed here into (62, 400) per clip
# --------------------------------------------------------------------
def load_subject_eeg(subject_file):
    return np.load(os.path.join(RAW_EEG_DIR, subject_file))  # (7, 62, 104000)

# --------------------------------------------------------------------
# Main preprocessing
# --------------------------------------------------------------------
def preprocess_eeg_from_blip():
    processed = load_processed_log()
    processed_count, skipped_count = 0, 0

    subjects = natural_sort([f for f in os.listdir(RAW_EEG_DIR) if f.endswith(".npy")])
    if not subjects:
        raise RuntimeError(f"No subject EEG files found in {RAW_EEG_DIR}")
    print("Using subject EEG file:", subjects[0])
    eeg_data = load_subject_eeg(subjects[0])  # just sub1 for now
    print("EEG data shape:", eeg_data.shape)

    # loop over BLIP embeddings as ground-truth naming
    for block_name in natural_sort(os.listdir(BLIP_DIR)):
        block_path = os.path.join(BLIP_DIR, block_name)
        if not os.path.isdir(block_path):
            continue

        save_block_dir = os.path.join(SAVE_DIR, block_name)
        os.makedirs(save_block_dir, exist_ok=True)

        # get all BLIP npys (e.g., class01_clip01.npy)
        blip_files = natural_sort([f for f in os.listdir(block_path) if f.endswith(".npy")])

        # which block index in EEG data (Block1 → 0, Block2 → 1, etc.)
        block_idx = int(re.findall(r'\d+', block_name)[0]) - 1
        block_eeg = eeg_data[block_idx]  # (62, 104000)

        clip_len = 2 * 200  # 2 seconds * 200Hz = 400 samples
        for i, blip_file in enumerate(blip_files):
            entry_name = f"{PROCESS_TAG} {block_name}/{blip_file.replace('.npy','')}"
            if entry_name in processed:
                skipped_count += 1
                continue

            start = i * clip_len
            end = start + clip_len
            if end > block_eeg.shape[1]:
                print(f"Warning: clip {blip_file} exceeds EEG length, skipping")
                continue

            clip_eeg = block_eeg[:, start:end]  # (62, 400)
            save_path = os.path.join(save_block_dir, blip_file.replace(".npy", ".npy"))
            np.save(save_path, clip_eeg)

            update_processed_log(entry_name)
            processed_count += 1

    print(f"\nSummary: {processed_count} EEG clips processed, {skipped_count} skipped")

# --------------------------------------------------------------------
if __name__ == "__main__":
    preprocess_eeg_from_blip()