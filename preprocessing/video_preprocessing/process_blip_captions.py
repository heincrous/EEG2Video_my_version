"""
ALIGN BLIP CAPTIONS TO CLIPS (Dual Embeddings)
----------------------------------------------------------------------
Generates two sets of embeddings for each caption:
  1. Stable Diffusion v1-4 CLIP embeddings (collapsed, used by diffusion)
     Saved in: BLIP_embeddings
  2. OpenAI CLIP ViT-L/14 embeddings (semantic-rich, for EEG predictor)
     Saved in: BLIP_embeddings_semantic
Both embeddings are shaped [77,768] (last_hidden_state).
"""

import os
import sys
import numpy as np
from tqdm import tqdm
import torch
from transformers import CLIPTokenizer, CLIPTextModel

# === Paths ===
caption_dir    = "/content/drive/MyDrive/EEG2Video_data/raw/BLIP-caption/"
out_text_dir   = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/"
out_embed_sd   = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_embeddings/"
out_embed_sem  = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_embeddings_semantic/"

# Path to pretrained SD v1-4 checkpoint
pretrained_model_path = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"

# === Import GT_LABEL ===
repo_root = "/content/EEG2Video_my_version"
sys.path.append(os.path.join(repo_root, "core_files"))
from gt_label import GT_LABEL

os.makedirs(out_text_dir, exist_ok=True)
os.makedirs(out_embed_sd, exist_ok=True)
os.makedirs(out_embed_sem, exist_ok=True)

# === Load encoders ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# Stable Diffusion CLIP
sd_tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
sd_encoder   = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder").to(device)
sd_encoder.eval()
print("Loaded SD CLIP text encoder (hidden size:", sd_encoder.config.hidden_size, ")")

# Semantic CLIP (OpenAI ViT-L/14)
sem_model_id = "openai/clip-vit-large-patch14"
sem_tokenizer = CLIPTokenizer.from_pretrained(sem_model_id)
sem_encoder   = CLIPTextModel.from_pretrained(sem_model_id).to(device)
sem_encoder.eval()
print("Loaded Semantic CLIP ViT-L/14 (hidden size:", sem_encoder.config.hidden_size, ")")

ordinals = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th"]
batch_size = 256

for block_id in range(7):
    fname = f"{ordinals[block_id]}_10min.txt"
    fpath = os.path.join(caption_dir, fname)
    if not os.path.exists(fpath):
        print(f"Missing caption file: {fname}, skipping Block {block_id+1}")
        continue

    with open(fpath, "r") as f:
        captions = [c.strip() for c in f.readlines()]

    if len(captions) != 200:
        print(f"[WARNING] Block {block_id+1} file has {len(captions)} captions, expected 200.")

    text_block_dir = os.path.join(out_text_dir, f"Block{block_id+1}")
    embed_sd_block = os.path.join(out_embed_sd, f"Block{block_id+1}")
    embed_sem_block = os.path.join(out_embed_sem, f"Block{block_id+1}")
    os.makedirs(text_block_dir, exist_ok=True)
    os.makedirs(embed_sd_block, exist_ok=True)
    os.makedirs(embed_sem_block, exist_ok=True)

    for start in tqdm(range(0, len(captions), batch_size), desc=f"Block {block_id+1}"):
        batch_caps = captions[start:start+batch_size]

        # Tokenize for SD
        sd_tokens = sd_tokenizer(
            batch_caps,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(device)

        # Tokenize for Semantic CLIP
        sem_tokens = sem_tokenizer(
            batch_caps,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            # SD embeddings
            emb_sd = sd_encoder(**sd_tokens)[0].cpu().numpy()      # [B,77,768]
            # Semantic embeddings
            emb_sem = sem_encoder(**sem_tokens)[0].cpu().numpy()   # [B,77,768]

        for j, caption in enumerate(batch_caps):
            flat_idx = start + j
            order_idx = flat_idx // 5
            clip_id = flat_idx % 5
            true_class = GT_LABEL[block_id, order_idx]

            # Save caption
            text_path = os.path.join(
                text_block_dir,
                f"class{true_class:02d}_clip{clip_id+1:02d}.txt"
            )
            with open(text_path, "w") as f:
                f.write(caption)

            # Save SD embedding
            sd_path = os.path.join(
                embed_sd_block,
                f"class{true_class:02d}_clip{clip_id+1:02d}.npy"
            )
            np.save(sd_path, emb_sd[j])

            # Save Semantic embedding
            sem_path = os.path.join(
                embed_sem_block,
                f"class{true_class:02d}_clip{clip_id+1:02d}.npy"
            )
            np.save(sem_path, emb_sem[j])

    print(f"Finished Block {block_id+1}: saved {len(os.listdir(text_block_dir))} captions")
