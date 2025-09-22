"""
ALIGN BLIP CAPTIONS TO CLIPS
------------------------------
Input:
  raw/BLIP-caption/1st_10min.txt ... 7th_10min.txt

Process:
  - Each line = caption for one 2s clip (200 per block)
  - Captions are in randomized presentation order
  - Use GT_LABEL (7x40) to map order_idx → true class index
  - Save plain text
  - Tokenize + encode with CLIP
  - Save embeddings [77,768]

Output:
  processed/BLIP_text/BlockY/classYY_clipZZ.txt
  processed/BLIP_embeddings/BlockY/classYY_clipZZ.npy
"""

import os
import numpy as np
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
import torch
import sys

# paths
caption_dir = "/content/drive/MyDrive/EEG2Video_data/raw/BLIP-caption/"
out_text_dir = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/"
out_embed_dir = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_embeddings/"

# import GT_LABEL directly from core_files/gt_label.py
repo_root = "/content/EEG2Video_my_version"
sys.path.append(os.path.join(repo_root, "core_files"))
from gt_label import GT_LABEL   # GT_LABEL shape (7,40), values 0–39

os.makedirs(out_text_dir, exist_ok=True)
os.makedirs(out_embed_dir, exist_ok=True)

# load CLIP model + tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
text_encoder.eval()

# correct ordinal filenames for 1–7
ordinals = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th"]

# process block caption files
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

    # create block folders
    text_block_dir = os.path.join(out_text_dir, f"Block{block_id+1}")
    embed_block_dir = os.path.join(out_embed_dir, f"Block{block_id+1}")
    os.makedirs(text_block_dir, exist_ok=True)
    os.makedirs(embed_block_dir, exist_ok=True)

    # loop over 40 presented classes in this block
    for order_idx in tqdm(range(40), desc=f"Block {block_id+1}"):
        true_class = GT_LABEL[block_id, order_idx]  # map presentation → true class

        for clip_id in range(5):
            # caption index in presentation order
            flat_idx = order_idx * 5 + clip_id
            if flat_idx >= len(captions):
                continue
            caption = captions[flat_idx]

            # save plain text
            text_path = os.path.join(
                text_block_dir,
                f"class{true_class:02d}_clip{clip_id+1:02d}.txt"
            )
            with open(text_path, "w") as f:
                f.write(caption)

            # tokenize + encode
            tokens = tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                emb = text_encoder(**tokens).last_hidden_state  # [1,77,768]

            # save embedding
            emb = emb.squeeze(0).cpu().numpy()  # [77,768]
            embed_path = os.path.join(
                embed_block_dir,
                f"class{true_class:02d}_clip{clip_id+1:02d}.npy"
            )
            np.save(embed_path, emb)

    print(f"Finished Block {block_id+1}: saved {len(os.listdir(text_block_dir))} captions")
