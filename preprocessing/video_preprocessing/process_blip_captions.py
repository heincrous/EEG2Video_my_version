import os
import re
import torch
import numpy as np
import clip

# CONFIG
RAW_BLIP_DIR = "/content/drive/MyDrive/EEG2Video_data/raw/BLIP-caption/"
SAVE_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_embeddings/"

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load CLIP model (ViT-B/32)
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

def get_blip_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".txt")]
    return sorted(files, key=lambda x: int(re.findall(r'\d+', x)[0]))

# -----------------------------
# Ask user which BLIP caption files to process
# -----------------------------
all_blip_files = get_blip_files(RAW_BLIP_DIR)
print("Available BLIP caption files:", all_blip_files)

user_input = input("Enter BLIP caption files to process (comma separated, e.g. block1.txt,block2.txt): ")
blip_list = [f.strip() for f in user_input.split(",") if f.strip() in all_blip_files]

if not blip_list:
    raise ValueError("No valid BLIP caption files selected!")

processed_count = 0

# -----------------------------
# Process selected caption files
# -----------------------------
for block_idx, blip_file in enumerate(blip_list, start=1):
    block_name = f"Block{all_blip_files.index(blip_file) + 1}"
    block_save_dir = os.path.join(SAVE_DIR, block_name)
    os.makedirs(block_save_dir, exist_ok=True)

    print(f"\nProcessing {blip_file} -> {block_name}")

    with open(os.path.join(RAW_BLIP_DIR, blip_file), "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) != 200:
        print(f"Warning: {blip_file} has {len(lines)} captions (expected 200)")

    for class_id in range(40):
        for clip_id in range(5):
            idx = class_id * 5 + clip_id
            if idx >= len(lines):
                continue
            caption = lines[idx]

            with torch.no_grad():
                text = clip.tokenize([caption]).to(device)
                embedding = model.encode_text(text)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            embedding = embedding.cpu().numpy()

            save_path = os.path.join(block_save_dir, f"class{class_id+1:02d}_clip{clip_id+1:02d}.npy")
            np.save(save_path, embedding)

            processed_count += 1

    print(f"Finished {blip_file}, saved {processed_count} embeddings so far")

print(f"\nSummary: {processed_count} captions embedded")