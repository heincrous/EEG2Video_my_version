# ==========================================
# GENERATE CLIP EMBEDDINGS FROM BLIP CAPTIONS
# ==========================================
# Input:
#   EEG2Video_data/raw/BLIP-caption/Xth_10min.txt
#   Each file has 200 captions per block.
#
# Process:
#   - Read captions directly (no GT ordering)
#   - Encode with Stable Diffusion v1-4 CLIP text encoder
#   - Store block-wise arrays
#
# Output:
#   EEG2Video_data/processed/BLIP_text/BLIP_text.npy
#       shape [7, 200]  (captions)
#   EEG2Video_data/processed/CLIP_embeddings/CLIP_embeddings.npy
#       shape [7, 200, 77, 768]  (embeddings)
# ==========================================

import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel


# ==========================================
# CONFIG
# ==========================================
config = {
    "drive_root": "/content/drive/MyDrive/EEG2Video_data",
    "model_path": "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4",
    "batch_size": 128,
}

caption_dir   = os.path.join(config["drive_root"], "raw", "BLIP-caption")
out_text_dir  = os.path.join(config["drive_root"], "processed", "BLIP_text_authors")
out_embed_dir = os.path.join(config["drive_root"], "processed", "CLIP_embeddings_authors")

os.makedirs(out_text_dir, exist_ok=True)
os.makedirs(out_embed_dir, exist_ok=True)


# ==========================================
# LOAD CLIP TOKENIZER + TEXT ENCODER
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = CLIPTokenizer.from_pretrained(config["model_path"], subfolder="tokenizer")
encoder   = CLIPTextModel.from_pretrained(config["model_path"], subfolder="text_encoder").to(device)
encoder.eval()

ordinals = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th"]


# ==========================================
# ALLOCATE OUTPUT ARRAYS
# ==========================================
all_texts = np.empty((7, 200), dtype=object)
all_embs  = np.zeros((7, 200, 77, 768), dtype=np.float32)


# ==========================================
# MAIN LOOP
# ==========================================
for block_id in range(7):
    fname = f"{ordinals[block_id]}_10min.txt"
    fpath = os.path.join(caption_dir, fname)
    if not os.path.exists(fpath):
        print(f"Missing caption file: {fname}, skipping block {block_id+1}")
        continue

    with open(fpath, "r") as f:
        captions = [c.strip() for c in f.readlines() if c.strip()]

    print(f"Processing Block {block_id+1}: {len(captions)} captions")

    for start in tqdm(range(0, len(captions), config["batch_size"]), desc=f"Block {block_id+1}"):
        batch_caps = captions[start:start + config["batch_size"]]

        tokens = tokenizer(
            batch_caps,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            emb = encoder(tokens["input_ids"])[0].cpu().numpy()

        for j, caption in enumerate(batch_caps):
            idx = start + j
            if idx < 200:
                all_texts[block_id, idx] = caption
                all_embs[block_id, idx]  = emb[j]


# ==========================================
# SAVE OUTPUT FILES
# ==========================================
np.save(os.path.join(out_text_dir,  "BLIP_text_authors.npy"),  all_texts)
np.save(os.path.join(out_embed_dir, "CLIP_embeddings_authors.npy"), all_embs)

print("\nDone.")
print(f"Saved BLIP_text_authors.npy shape: {all_texts.shape}")
print(f"Saved CLIP_embeddings_authors.npy shape: {all_embs.shape}")
