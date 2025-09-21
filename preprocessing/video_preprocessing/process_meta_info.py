import os
import numpy as np

# CONFIG
RAW_META_DIR = "/content/drive/MyDrive/EEG2Video_data/raw/meta-info/"
SAVE_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/Meta_info/"
LOG_FILE = "/content/drive/MyDrive/EEG2Video_data/processed/processed_log.txt"
PROCESS_TAG = "[META]"

os.makedirs(SAVE_DIR, exist_ok=True)

META_FILES = [
    "All_video_color.npy",
    "All_video_face_apperance.npy",   # filenames are 'apperance' in SEED-DV
    "All_video_human_apperance.npy",
    "All_video_label.npy",
    "All_video_obj_number.npy",
    "All_video_optical_flow_score.npy"
]

def load_processed_log():
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, "r") as f:
        return set(line.strip() for line in f.readlines())

def update_processed_log(entry):
    if entry not in load_processed_log():
        with open(LOG_FILE, "a") as f:
            f.write(entry + "\n")

# Load all meta arrays, flatten if stored as 7 blocks
meta_data = {}
for fname in META_FILES:
    path = os.path.join(RAW_META_DIR, fname)
    arr = np.load(path, allow_pickle=True)
    # If it's a list of 7 arrays (one per block), flatten to (280,)
    if arr.shape[0] == 7:
        arr = np.concatenate(arr, axis=0)
    meta_data[fname.split(".")[0]] = arr

# Expect shape (280,) = 7 blocks × 40 classes
total_classes = meta_data["All_video_label"].shape[0]
if total_classes != 280:
    print(f"Warning: expected 280 class entries, found {total_classes}")

processed = load_processed_log()
processed_count, skipped_count = 0, 0

# Iterate over 280 class entries and expand each to 5 clips
for idx in range(total_classes):
    block_id = idx // 40 + 1
    class_id = (idx % 40) + 1

    for clip_id in range(1, 6):
        block_name = f"Block{block_id}"
        clip_name = f"class{class_id:02d}_clip{clip_id:02d}"
        entry = f"{PROCESS_TAG} {block_name}/{clip_name}"

        if entry in processed:
            skipped_count += 1
            continue

        save_block_dir = os.path.join(SAVE_DIR, block_name)
        os.makedirs(save_block_dir, exist_ok=True)

        # Same meta-info for all 5 clips of this class
        clip_info = {k: v[idx] for k, v in meta_data.items()}

        save_path = os.path.join(save_block_dir, f"{clip_name}.npz")
        np.savez(save_path, **clip_info)

        update_processed_log(entry)
        processed_count += 1

print(f"\nSummary: {processed_count} meta-info clips processed, {skipped_count} skipped")