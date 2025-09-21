import os
import numpy as np

# CONFIG
RAW_META_DIR = "/content/drive/MyDrive/EEG2Video_data/raw/meta-info/"
SAVE_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/Meta_info/"

os.makedirs(SAVE_DIR, exist_ok=True)

META_FILES = [
    "All_video_color.npy",
    "All_video_face_apperance.npy",   # filenames are 'apperance' in SEED-DV
    "All_video_human_apperance.npy",
    "All_video_label.npy",
    "All_video_obj_number.npy",
    "All_video_optical_flow_score.npy"
]

# -----------------------------
# Load all meta arrays
# -----------------------------
meta_data = {}
for fname in META_FILES:
    path = os.path.join(RAW_META_DIR, fname)
    arr = np.load(path, allow_pickle=True)
    # If it’s a list of 7 arrays (one per block), flatten
    if arr.shape[0] == 7:
        arr = np.concatenate(arr, axis=0)
    meta_data[fname.split(".")[0]] = arr

# Expect 280 entries = 7 blocks × 40 classes
total_classes = meta_data["All_video_label"].shape[0]
if total_classes != 280:
    print(f"Warning: expected 280 class entries, found {total_classes}")

# -----------------------------
# Ask user which blocks to process
# -----------------------------
available_blocks = [f"Block{i}" for i in range(1, 8)]
print("Available blocks:", available_blocks)

user_input = input("Enter blocks to process (comma separated, e.g. Block1,Block2): ")
block_list = [b.strip() for b in user_input.split(",") if b.strip() in available_blocks]

if not block_list:
    raise ValueError("No valid blocks selected!")

processed_count = 0

# -----------------------------
# Save meta-info per class/clip
# -----------------------------
for block_name in block_list:
    block_id = int(block_name.replace("Block", ""))
    print(f"\nProcessing {block_name} ...")

    save_block_dir = os.path.join(SAVE_DIR, block_name)
    os.makedirs(save_block_dir, exist_ok=True)

    # Each block has 40 classes
    for class_id in range(1, 41):
        idx = (block_id - 1) * 40 + (class_id - 1)

        for clip_id in range(1, 6):
            clip_name = f"class{class_id:02d}_clip{clip_id:02d}"

            # Meta-info is identical for all 5 clips of this class
            clip_info = {k: v[idx] for k, v in meta_data.items()}

            save_path = os.path.join(save_block_dir, f"{clip_name}.npz")
            np.savez(save_path, **clip_info)

            processed_count += 1

    print(f"Finished {block_name}, saved into {save_block_dir}")

print(f"\nSummary: {processed_count} meta-info clips processed")