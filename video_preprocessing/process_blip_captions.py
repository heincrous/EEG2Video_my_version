import os
import re
import torch
import numpy as np
import clip

# CONFIG
RAW_BLIP_DIR = "/content/drive/MyDrive/EEG2Video_data/raw/BLIP-caption/"
SAVE_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_embeddings/"
LOG_FILE = "/content/drive/MyDrive/EEG2Video_data/processed/processed_log.txt"
PROCESS_TAG = "[BLIP]"

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load CLIP model (ViT-B/32)
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

def get_files_in_directory(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".txt")]
    return sorted(files, key=lambda x: int(re.findall(r'\d+', x)[0]))

def load_processed_log():
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, "r") as f:
        return set(line.strip() for line in f.readlines())

def update_processed_log(entry):
    if entry not in load_processed_log():
        with open(LOG_FILE, "a") as f:
            f.write(entry + "\n")

# Main loop
blip_files = get_files_in_directory(RAW_BLIP_DIR)
processed = load_processed_log()
processed_count, skipped_count = 0, 0

for block_idx, blip_file in enumerate(blip_files, start=1):
    block_name = f"Block{block_idx}"
    block_save_dir = os.path.join(SAVE_DIR, block_name)
    os.makedirs(block_save_dir, exist_ok=True)

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
            entry = f"{PROCESS_TAG} {block_name}/class{class_id+1:02d}_clip{clip_id+1:02d}"
            if entry in processed:
                skipped_count += 1
                continue

            with torch.no_grad():
                text = clip.tokenize([caption]).to(device)
                embedding = model.encode_text(text)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            embedding = embedding.cpu().numpy()

            save_path = os.path.join(block_save_dir, f"class{class_id+1:02d}_clip{clip_id+1:02d}.npy")
            np.save(save_path, embedding)

            update_processed_log(entry)
            processed_count += 1

print(f"\nSummary: {processed_count} captions embedded, {skipped_count} skipped")