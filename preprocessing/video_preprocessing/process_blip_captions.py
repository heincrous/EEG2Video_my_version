# ==========================================
# ALIGN BLIP CAPTIONS TO CLIPS
# ==========================================
# Input:
#   EEG2Video_data/raw/BLIP-caption/Xth_10min.txt
#   Each file contains 200 captions per block
#
# Process:
#   - Use BLIP-provided captions (strings).
#   - Generate embeddings for each caption using Stable Diffusion v1-4
#     tokenizer + text_encoder (CLIP).
#   - Each embedding has shape [77, 768].
#   - Align embeddings with GT_LABEL class order.
#
# Output:
#   EEG2Video_data/processed/BLIP_text/BLIP_text.npy
#       Shape [7, 40, 5] (captions as strings)
#   EEG2Video_data/processed/CLIP_embeddings/CLIP_embeddings.npy
#       Shape [7, 40, 5, 77, 768]
# ==========================================

import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel

from core.gt_label import GT_LABEL


# ==========================================
# CONFIG
# ==========================================
config = {
    "drive_root": "/content/drive/MyDrive/EEG2Video_data",
    "model_path": "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4",
    "batch_size": 128,
}

caption_dir   = os.path.join(config["drive_root"], "raw", "BLIP-caption")
out_text_dir  = os.path.join(config["drive_root"], "processed", "BLIP_text")
out_embed_dir = os.path.join(config["drive_root"], "processed", "CLIP_embeddings")

os.makedirs(out_text_dir, exist_ok=True)
os.makedirs(out_embed_dir, exist_ok=True)


# ==========================================
# Load CLIP tokenizer and text encoder
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = CLIPTokenizer.from_pretrained(config["model_path"], subfolder="tokenizer")
encoder   = CLIPTextModel.from_pretrained(config["model_path"], subfolder="text_encoder").to(device)
encoder.eval()

ordinals = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th"]


# ==========================================
# Allocate block-level arrays
# ==========================================
all_texts = np.empty((7, 40, 5), dtype=object)
all_embs  = np.zeros((7, 40, 5, 77, 768), dtype=np.float32)


# ==========================================
# Main processing loop
# ==========================================
for block_id in range(7):
    fname = f"{ordinals[block_id]}_10min.txt"
    fpath = os.path.join(caption_dir, fname)
    if not os.path.exists(fpath):
        print(f"Missing caption file: {fname}, skipping Block {block_id+1}")
        continue

    with open(fpath, "r") as f:
        captions = [c.strip() for c in f.readlines()]

    if len(captions) != 200:
        print(f"Warning: Block {block_id+1} has {len(captions)} captions (expected 200)")

    for start in tqdm(range(0, len(captions), config["batch_size"]), desc=f"Block {block_id+1}"):
        batch_caps = captions[start:start+config["batch_size"]]

        tokens = tokenizer(
            batch_caps,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            emb = encoder(**tokens)[0].cpu().numpy()  # [B,77,768]

        for j, caption in enumerate(batch_caps):
            flat_idx   = start + j
            order_idx  = flat_idx // 5
            clip_id    = flat_idx % 5
            true_class = GT_LABEL[block_id, order_idx]

            all_texts[block_id, true_class, clip_id] = caption
            all_embs[block_id, true_class, clip_id]  = emb[j]


# ==========================================
# Save block-level arrays
# ==========================================
out_text_path  = os.path.join(out_text_dir,  "BLIP_text.npy")
out_embed_path = os.path.join(out_embed_dir, "CLIP_embeddings.npy")

np.save(out_text_path, all_texts)
np.save(out_embed_path, all_embs)

print(f"\nFinished all blocks →")
print(f"Saved BLIP_text.npy shape {all_texts.shape}")
print(f"Saved CLIP_embeddings.npy shape {all_embs.shape}")
print("\nProcessing complete.")