"""
ALIGN BLIP CAPTIONS TO CLIPS (Stable Diffusion v1-4 CLIP embeddings only)
----------------------------------------------------------------------
Generates embeddings for each caption using the Stable Diffusion v1-4
tokenizer + text_encoder. These are the exact embeddings used during
diffusion training. Each embedding has shape [77,768].
"""

import os, sys
import numpy as np
from tqdm import tqdm
import torch
from transformers import CLIPTokenizer, CLIPTextModel

# === Paths ===
caption_dir   = "/content/drive/MyDrive/EEG2Video_data/raw/BLIP-caption/"
out_text_dir  = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/"
out_embed_dir = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_embeddings/"

pretrained_model_path = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"

# === Import GT_LABEL ===
repo_root = "/content/EEG2Video_my_version"
sys.path.append(os.path.join(repo_root, "core_files"))
from gt_label import GT_LABEL

os.makedirs(out_text_dir, exist_ok=True)
os.makedirs(out_embed_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Stable Diffusion CLIP
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
encoder   = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder").to(device)
encoder.eval()

ordinals = ["1st","2nd","3rd","4th","5th","6th","7th"]
batch_size = 256

for block_id in range(7):
    fname = f"{ordinals[block_id]}_10min.txt"
    fpath = os.path.join(caption_dir, fname)
    if not os.path.exists(fpath):
        print(f"Missing caption file: {fname}, skipping Block {block_id+1}")
        continue

    with open(fpath, "r") as f:
        captions = [c.strip() for c in f.readlines()]

    text_block_dir  = os.path.join(out_text_dir, f"Block{block_id+1}")
    embed_block_dir = os.path.join(out_embed_dir, f"Block{block_id+1}")
    os.makedirs(text_block_dir, exist_ok=True)
    os.makedirs(embed_block_dir, exist_ok=True)

    for start in tqdm(range(0,len(captions),batch_size), desc=f"Block {block_id+1}"):
        batch_caps = captions[start:start+batch_size]

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
            flat_idx = start+j
            order_idx = flat_idx // 5
            clip_id   = flat_idx % 5
            true_class = GT_LABEL[block_id, order_idx]

            # save caption
            text_path = os.path.join(
                text_block_dir,
                f"class{true_class:02d}_clip{clip_id+1:02d}.txt"
            )
            with open(text_path,"w") as f:
                f.write(caption)

            # save embedding
            embed_path = os.path.join(
                embed_block_dir,
                f"class{true_class:02d}_clip{clip_id+1:02d}.npy"
            )
            np.save(embed_path, emb[j])

    print(f"Finished Block {block_id+1}")
