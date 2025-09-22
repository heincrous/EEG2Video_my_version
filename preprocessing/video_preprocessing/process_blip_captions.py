"""
ALIGN BLIP CAPTIONS TO CLIPS
------------------------------
Input:
  raw/BLIP-caption/1st_10min.txt ... 7th_10min.txt

Process:
  - Each line = caption for one 2s clip
  - Use GT_LABEL to remap randomized order → class indices
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

# paths
caption_dir = "/content/drive/MyDrive/EEG2Video_data/raw/BLIP-caption/"
out_text_dir = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/"
out_embed_dir = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_embeddings/"

# resolve repo root and core files
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
gt_label_path = os.path.join(repo_root, "core_files", "gt_label.npy")

os.makedirs(out_text_dir, exist_ok=True)
os.makedirs(out_embed_dir, exist_ok=True)

# load GT_LABEL [7,40,5]
GT_LABEL = np.load(gt_label_path)

# load CLIP model + tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
text_encoder.eval()

# process block caption files
for block_id in range(7):
    fname = f"{block_id+1}th_10min.txt"
    fpath = os.path.join(caption_dir, fname)
    if not os.path.exists(fpath):
        continue

    with open(fpath, "r") as f:
        captions = f.readlines()
    captions = [c.strip() for c in captions]

    # create block folders
    text_block_dir = os.path.join(out_text_dir, f"Block{block_id+1}")
    embed_block_dir = os.path.join(out_embed_dir, f"Block{block_id+1}")
    os.makedirs(text_block_dir, exist_ok=True)
    os.makedirs(embed_block_dir, exist_ok=True)

    # loop classes/clips
    for class_id in tqdm(range(40), desc=f"Block {block_id+1}"):
        for clip_id in range(5):
            # get index from GT_LABEL
            idx = GT_LABEL[block_id, class_id, clip_id]
            caption = captions[idx]

            # save text
            text_path = os.path.join(text_block_dir, f"class{class_id:02d}_clip{clip_id+1:02d}.txt")
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
            embed_path = os.path.join(embed_block_dir, f"class{class_id:02d}_clip{clip_id+1:02d}.npy")
            np.save(embed_path, emb)
