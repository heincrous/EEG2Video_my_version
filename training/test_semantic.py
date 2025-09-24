# ==========================================
# Semantic Predictor Multi-Sample Evaluation
# ==========================================

# === Standard libraries ===
import os
import pickle
import random

# === Third-party libraries ===
import numpy as np
import torch
import torch.nn.functional as F

# === Repo imports ===
from train_semantic import SemanticPredictor


# ==========================================
# Cosine similarity helper
# ==========================================
def cosine_similarity(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum().item()


# ==========================================
# Main evaluation
# ==========================================
if __name__ == "__main__":
    # === Paths ===
    eeg_root       = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_DE"
    blip_root      = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_embeddings"
    test_list_path = os.path.join(eeg_root, "test_list.txt")
    ckpt_root      = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_predictor"

    # === Checkpoint selection ===
    ckpts = [f for f in os.listdir(ckpt_root) if f.startswith("semantic_predictor_") and f.endswith(".pt")]
    print("\nAvailable checkpoints:")
    for i, ck in enumerate(ckpts):
        print(f"[{i}] {ck}")

    choice      = int(input("\nEnter checkpoint index: ").strip())
    ckpt_file   = ckpts[choice]
    tag         = ckpt_file.replace("semantic_predictor_", "").replace(".pt", "")
    ckpt_path   = os.path.join(ckpt_root, ckpt_file)
    scaler_path = os.path.join(ckpt_root, f"scaler_{tag}.pkl")

    print(f"\nLoading checkpoint: {ckpt_file}")
    print(f"Loading scaler: scaler_{tag}.pkl")

    # === Load scaler ===
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # === Load model ===
    model = SemanticPredictor().cuda()
    checkpoint = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # === Load test list ===
    with open(test_list_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # === Evaluate 10 random samples ===
    num_samples = 10
    chosen = random.sample(lines, num_samples)

    total_mse, total_cos = 0.0, 0.0
    print("\n=== Multi-Sample Evaluation ===")
    print(f"Checkpoint used: {ckpt_file}\n")

    for rel_path in chosen:
        eeg_path  = os.path.join(eeg_root, rel_path)
        blip_rel  = "/".join(rel_path.split("/")[1:])
        blip_path = os.path.join(blip_root, blip_rel)

        eeg  = np.load(eeg_path).reshape(-1)   # (310,)
        blip = np.load(blip_path).reshape(-1)  # (77*768,)

        eeg_scaled = scaler.transform([eeg])[0]
        eeg_tensor = torch.tensor(eeg_scaled, dtype=torch.float32).unsqueeze(0).cuda()
        blip_tensor = torch.tensor(blip, dtype=torch.float32).cuda()

        with torch.no_grad():
            pred = model(eeg_tensor).squeeze(0)

        mse = F.mse_loss(pred, blip_tensor).item()
        cos = cosine_similarity(pred, blip_tensor)

        total_mse += mse
        total_cos += cos

        print(f"{rel_path} | MSE={mse:.6f}, Cosine={cos:.6f}")

    avg_mse = total_mse / num_samples
    avg_cos = total_cos / num_samples

    print("\n=== Averages across 10 random samples ===")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Average Cosine similarity: {avg_cos:.6f}")
