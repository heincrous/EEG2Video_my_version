"""
ALIGN BLIP CAPTIONS TO CLIPS (Stable Diffusion v1-4 CLIP embeddings)
----------------------------------------------------------------------
Uses the tokenizer + text_encoder from stable-diffusion-v1-4 so that
embeddings exactly match those used during diffusion training.
Each embedding is [77,768] from outputs.last_hidden_state.
"""

import os
import sys
import numpy as np
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
import torch

# === Paths ===
caption_dir   = "/content/drive/MyDrive/EEG2Video_data/raw/BLIP-caption/"
out_text_dir  = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/"
out_embed_dir = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_embeddings/"

# Path to the pretrained SD v1-4 checkpoint
pretrained_model_path = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"

# === Import GT_LABEL ===
repo_root = "/content/EEG2Video_my_version"
sys.path.append(os.path.join(repo_root, "core_files"))
from gt_label import GT_LABEL

os.makedirs(out_text_dir, exist_ok=True)
os.makedirs(out_embed_dir, exist_ok=True)

# === Load Stable Diffusion tokenizer + text encoder ===
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder").to(device)
text_encoder.eval()

hidden_size = text_encoder.config.hidden_size
print("Loaded Stable Diffusion CLIP text encoder hidden size:", hidden_size)
if hidden_size != 768:
    sys.exit(f"❌ Wrong model loaded (got hidden_size={hidden_size}). Aborting.")

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
    embed_block_dir = os.path.join(out_embed_dir, f"Block{block_id+1}")
    os.makedirs(text_block_dir, exist_ok=True)
    os.makedirs(embed_block_dir, exist_ok=True)

    for start in tqdm(range(0, len(captions), batch_size), desc=f"Block {block_id+1}"):
        batch_caps = captions[start:start+batch_size]

        tokens = tokenizer(
            batch_caps,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            # EXACTLY what diffusion training does:
            # text_encoder(prompt_ids)[0] = last_hidden_state
            seq_emb = text_encoder(**tokens)[0]  # [B,77,768]

        emb = seq_emb.cpu().numpy()

        for j, caption in enumerate(batch_caps):
            flat_idx = start + j
            order_idx = flat_idx // 5
            clip_id = flat_idx % 5
            true_class = GT_LABEL[block_id, order_idx]

            text_path = os.path.join(
                text_block_dir,
                f"class{true_class:02d}_clip{clip_id+1:02d}.txt"
            )
            with open(text_path, "w") as f:
                f.write(caption)

            if emb[j].shape != (77, 768):
                sys.exit(f"❌ Embedding shape mismatch: got {emb[j].shape}, expected (77,768).")

            embed_path = os.path.join(
                embed_block_dir,
                f"class{true_class:02d}_clip{clip_id+1:02d}.npy"
            )
            np.save(embed_path, emb[j])

    print(f"Finished Block {block_id+1}: saved {len(os.listdir(text_block_dir))} captions")
