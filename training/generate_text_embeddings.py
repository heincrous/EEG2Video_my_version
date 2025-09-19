import os
import numpy as np
import torch
import clip
from tqdm import tqdm

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading CLIP...")
    model, preprocess = clip.load("ViT-B/32", device=device)

    all_embeddings = []

    # Loop over 1st_10min.txt ... 7th_10min.txt
    for i in range(1, 8):
        caption_file = f"/content/drive/MyDrive/Data/Raw/BLIP-caption/{i}st_10min.txt" if i == 1 else \
                       f"/content/drive/MyDrive/Data/Raw/BLIP-caption/{i}th_10min.txt"

        if not os.path.exists(caption_file):
            print(f"Warning: {caption_file} not found, skipping")
            continue

        # Load lines of captions
        with open(caption_file, "r") as f:
            captions = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(captions)} captions from {caption_file}")

        # Encode with CLIP
        text = clip.tokenize(captions, truncate=True).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text).cpu().numpy()

        all_embeddings.append(text_features)

    if len(all_embeddings) == 0:
        raise RuntimeError("No caption files found. Check BLIP-caption folder.")

    # Concatenate across all blocks
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print("Final embeddings shape:", all_embeddings.shape)

    # Save inside training folder
    save_path = "/content/EEG2Video_my_version/training/text_embedding.npy"
    np.save(save_path, all_embeddings)
    print(f"Saved embeddings to {save_path}")

if __name__ == "__main__":
    main()
