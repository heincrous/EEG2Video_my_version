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

# Load all meta arrays
meta_data = {}
for fname in META_FILES:
    path = os.path.join(RAW_META_DIR, fname)
    meta_data[fname.split(".")[0]] = np.load(path, allow_pickle=True)

# Expect shape (1400,)
total_clips = meta_data["All_video_label"].shape[0]
if total_clips != 1400:
    print(f"Warning: expected 1400 clips, found {total_clips}")

processed = load_processed_log()
processed_count, skipped_count = 0, 0

# Iterate over 1400 clips
for idx in range(total_clips):
    block_id = idx // 200 + 1
    class_id = (idx % 200) // 5 + 1
    clip_id = (idx % 5) + 1

    block_name = f"Block{block_id}"
    clip_name = f"class{class_id:02d}_clip{clip_id:02d}"
    entry = f"{PROCESS_TAG} {block_name}/{clip_name}"

    if entry in processed:
        skipped_count += 1
        continue

    save_block_dir = os.path.join(SAVE_DIR, block_name)
    os.makedirs(save_block_dir, exist_ok=True)

    clip_info = {k: v[idx] for k, v in meta_data.items()}

    save_path = os.path.join(save_block_dir, f"{clip_name}.npz")
    np.savez(save_path, **clip_info)

    update_processed_log(entry)
    processed_count += 1

print(f"\nSummary: {processed_count} meta-info clips processed, {skipped_count} skipped")