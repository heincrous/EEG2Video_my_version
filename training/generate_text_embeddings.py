# training/generate_text_embeddings.py

import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading CLIP text model (ViT-L/14)...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

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
            tokens = tokenizer(
                batch,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = text_model(**tokens)
                emb = outputs.last_hidden_state  # [B, 77, 768]

            embeddings.append(emb.cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)  # [num_captions, 77, 768]
        all_embeddings.append(embeddings)

    if not all_embeddings:
        raise RuntimeError("No caption files found. Check BLIP-caption folder.")

    # Concatenate across all blocks
    all_embeddings = np.concatenate(all_embeddings, axis=0)  # [N, 77, 768]
    print("Final embeddings shape:", all_embeddings.shape)

    # Save
    save_path = "/content/drive/MyDrive/Data/Raw/text_embeddings_full.npy"
    np.save(save_path, all_embeddings)
    print(f"Saved embeddings to {save_path}")

if __name__ == "__main__":
    main()
