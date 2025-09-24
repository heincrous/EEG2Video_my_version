"""
ALIGN BLIP CAPTIONS TO CLIPS (Batched, Strict CLIP ViT-L/14 with pooled embeddings)
------------------------------------------------------------
Ensures embeddings are meaningful and shaped [77,768].
Captions are kept as-is, only embeddings are regenerated.
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

# === Import GT_LABEL ===
repo_root = "/content/EEG2Video_my_version"
sys.path.append(os.path.join(repo_root, "core_files"))
from gt_label import GT_LABEL

os.makedirs(out_text_dir, exist_ok=True)
os.makedirs(out_embed_dir, exist_ok=True)

# === Load CLIP text encoder ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-large-patch14"

tokenizer = CLIPTokenizer.from_pretrained(model_id)
text_encoder = CLIPTextModel.from_pretrained(model_id).to(device)
text_encoder.eval()

# strict sanity check
hidden_size = text_encoder.config.hidden_size
print("Loaded CLIP text encoder hidden size:", hidden_size)
if hidden_size != 768:
    sys.exit(f"❌ Wrong model loaded (got hidden_size={hidden_size}). Aborting.")

# === Block file names ===
ordinals = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th"]

# batching setup
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

    # create block folders
    text_block_dir = os.path.join(out_text_dir, f"Block{block_id+1}")
    embed_block_dir = os.path.join(out_embed_dir, f"Block{block_id+1}")
    os.makedirs(text_block_dir, exist_ok=True)
    os.makedirs(embed_block_dir, exist_ok=True)

    # process in batches
    for start in tqdm(range(0, len(captions), batch_size), desc=f"Block {block_id+1}"):
        batch_caps = captions[start:start+batch_size]

        # tokenize
        tokens = tokenizer(
            batch_caps,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = text_encoder(**tokens)
            pooled = outputs.pooler_output  # [B,768]

        # expand pooled embeddings to [77,768] for pipeline compatibility
        emb = np.tile(pooled[:, None, :].cpu().numpy(), (1,77,1))  # [B,77,768]

        # save results
        for j, caption in enumerate(batch_caps):
            flat_idx = start + j
            order_idx = flat_idx // 5
            clip_id = flat_idx % 5
            true_class = GT_LABEL[block_id, order_idx]

            # save caption
            text_path = os.path.join(
                text_block_dir,
                f"class{true_class:02d}_clip{clip_id+1:02d}.txt"
            )
            with open(text_path, "w") as f:
                f.write(caption)

            # sanity check
            if emb[j].shape != (77, 768):
                sys.exit(f"❌ Embedding shape mismatch: got {emb[j].shape}, expected (77,768).")

            # save embedding
            embed_path = os.path.join(
                embed_block_dir,
                f"class{true_class:02d}_clip{clip_id+1:02d}.npy"
            )
            np.save(embed_path, emb[j])

    print(f"Finished Block {block_id+1}: saved {len(os.listdir(text_block_dir))} captions")
