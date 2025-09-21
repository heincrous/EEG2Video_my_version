import os
import re
import torch
import numpy as np
import clip
from core_files.gt_label import GT_LABEL   # <-- import the ground-truth order

# CONFIG
RAW_BLIP_DIR = "/content/drive/MyDrive/EEG2Video_data/raw/BLIP-caption/"
EMB_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_embeddings/"
CAP_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_captions/"

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(CAP_DIR, exist_ok=True)

# Load CLIP model (ViT-B/32)
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

def get_blip_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".txt")]
    return sorted(files, key=lambda x: int(re.findall(r'\d+', x)[0]))

# Map raw video filenames to Block names
BLOCK_MAP = {
    "1st_10min": "Block1",
    "2nd_10min": "Block2",
    "3rd_10min": "Block3",
    "4th_10min": "Block4",
    "5th_10min": "Block5",
    "6th_10min": "Block6",
    "7th_10min": "Block7",
}

# -----------------------------
# Ask user which BLIP caption files to process
# -----------------------------
all_blip_files = get_blip_files(RAW_BLIP_DIR)
print("Available BLIP caption files:", all_blip_files)

user_input = input("Enter BLIP caption files to process (comma separated, e.g. 1st_10min.txt,2nd_10min.txt): ")
blip_list = [f.strip() for f in user_input.split(",") if f.strip() in all_blip_files]

if not blip_list:
    raise ValueError("No valid BLIP caption files selected!")

processed_count = 0

# -----------------------------
# Process selected caption files
# -----------------------------
for blip_file in blip_list:
    block_key = os.path.splitext(blip_file)[0]  # e.g. "1st_10min"
    block_name = BLOCK_MAP.get(block_key, block_key)
    block_idx = int(block_name.replace("Block", "")) - 1

    block_emb_dir = os.path.join(EMB_DIR, block_name)
    block_cap_dir = os.path.join(CAP_DIR, block_name)
    os.makedirs(block_emb_dir, exist_ok=True)
    os.makedirs(block_cap_dir, exist_ok=True)

    print(f"\nProcessing {blip_file} -> {block_name}")

    with open(os.path.join(RAW_BLIP_DIR, blip_file), "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) != 200:
        print(f"Warning: {blip_file} has {len(lines)} captions (expected 200)")

    for class_pos in range(40):
        true_class = GT_LABEL[block_idx, class_pos]
        for clip_id in range(5):
            idx = class_pos * 5 + clip_id
            if idx >= len(lines):
                continue
            caption = lines[idx]

            with torch.no_grad():
                text = clip.tokenize([caption]).to(device)
                embedding = model.encode_text(text)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            embedding = embedding.cpu().numpy()

            base_name = f"class{true_class:02d}_clip{clip_id+1:02d}"

            # save embedding as .npy (for training)
            np.save(os.path.join(block_emb_dir, base_name + ".npy"), embedding)

            # save caption as .txt (for inspection/debugging)
            with open(os.path.join(block_cap_dir, base_name + ".txt"), "w") as ftxt:
                ftxt.write(caption)

            processed_count += 1

    print(f"Finished {blip_file}, saved {processed_count} embeddings so far")

print(f"\nSummary: {processed_count} captions embedded")