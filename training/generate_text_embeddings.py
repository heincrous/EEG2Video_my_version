import os
import numpy as np
import torch
import clip

def main():
    # Paths
    blip_folder = "/content/drive/MyDrive/Data/Raw/BLIP-caption"
    save_path = "/content/EEG2Video_my_version/training_original/text_embedding.npy"

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    all_embeddings = []

    # Loop over block caption files
    for block_id in range(1, 8):  # block01.txt ... block07.txt
        caption_file = os.path.join(blip_folder, f"block{block_id:02d}.txt")
        if not os.path.exists(caption_file):
            print(f"Warning: {caption_file} not found, skipping")
            continue

        with open(caption_file, "r") as f:
            captions = [line.strip() for line in f.readlines() if line.strip()]

        print(f"Encoding {len(captions)} captions from {caption_file}")

        # Encode with CLIP
        with torch.no_grad():
            text_tokens = clip.tokenize(captions).to(device)
            text_embeddings = model.encode_text(text_tokens).float()
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)  # normalize

        all_embeddings.append(text_embeddings.cpu().numpy())

    # Concatenate into one array
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print("Final embedding shape:", all_embeddings.shape)

    # Save as .npy file inside training folder
    np.save(save_path, all_embeddings)
    print(f"Saved embeddings to {save_path}")

if __name__ == "__main__":
    main()
