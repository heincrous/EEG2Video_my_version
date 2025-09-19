# training/generate_text_embeddings.py

import os
import numpy as np
import torch
import clip
from tqdm import tqdm

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading CLIP...")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Use the actual filenames in your Drive
    caption_files = [
        "1st_10min.txt",
        "2nd_10min.txt",
        "3rd_10min.txt",
        "4th_10min.txt",
        "5th_10min.txt",
        "6th_10min.txt",
        "7th_10min.txt",
    ]

    base_dir = "/content/drive/MyDrive/Data/Raw/BLIP-caption"
    all_embeddings = []

    for fname in caption_files:
        caption_file = os.path.join(base_dir, fname)
        if not os.path.exists(caption_file):
            print(f"Warning: {caption_file} not found, skipping")
            continue

        # Load captions (one per line)
        with open(caption_file, "r") as f:
            captions = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(captions)} captions from {caption_file}")

        # Encode in batches
        batch_size = 32
        embeddings = []
        for i in tqdm(range(0, len(captions), batch_size)):
            batch = captions[i:i+batch_size]
            tokens = clip.tokenize(batch, truncate=True).to(device)
            with torch.no_grad():
                emb = model.encode_text(tokens)
            embeddings.append(emb.cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        all_embeddings.append(embeddings)

    if not all_embeddings:
        raise RuntimeError("No caption files found. Check BLIP-caption folder.")

    # Concatenate across all blocks
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print("Final embeddings shape:", all_embeddings.shape)

    # Save
    save_path = "/content/drive/MyDrive/Data/Raw/text_embedding.npy"
    np.save(save_path, all_embeddings)
    print(f"Saved embeddings to {save_path}")

if __name__ == "__main__":
    main()
