# import os
# import torch
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# import pickle

# # -------------------------------------------------------------------------
# # Semantic Predictor (same as training definition)
# # -------------------------------------------------------------------------
# class SemanticPredictor(nn.Module):
#     def __init__(self):
#         super(SemanticPredictor, self).__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(310, 10000),
#             nn.ReLU(),
#             nn.Linear(10000, 10000),
#             nn.ReLU(),
#             nn.Linear(10000, 10000),
#             nn.ReLU(),
#             nn.Linear(10000, 10000),
#             nn.ReLU(),
#             nn.Linear(10000, 77 * 768)
#         )
#     def forward(self, eeg):
#         return self.mlp(eeg)


# # -------------------------------------------------------------------------
# # Dataset wrapper (loads DE + BLIP embeddings from subject npz)
# # -------------------------------------------------------------------------
# class NPZSemanticDataset(Dataset):
#     def __init__(self, npz_files, scaler, max_samples=None):
#         self.samples = []
#         self.scaler = scaler
#         for f in npz_files:
#             data = np.load(f, allow_pickle=True)
#             eeg = data["EEG_DE"]                 # (N, 62, 5)
#             txt = data["BLIP_embeddings"]        # (N, 77, 768)
#             for i in range(len(eeg)):
#                 eeg_flat = eeg[i].reshape(-1)    # 310
#                 txt_flat = txt[i].reshape(-1)    # 77*768
#                 self.samples.append((eeg_flat, txt_flat))
#         if max_samples:
#             self.samples = self.samples[:max_samples]

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         eeg, txt = self.samples[idx]
#         eeg = self.scaler.transform([eeg])[0]
#         return torch.tensor(eeg, dtype=torch.float32), torch.tensor(txt, dtype=torch.float32)


# # -------------------------------------------------------------------------
# # Cosine similarity helper
# # -------------------------------------------------------------------------
# def cosine_similarity(a, b):
#     a = F.normalize(a, dim=-1)
#     b = F.normalize(b, dim=-1)
#     return (a * b).sum().item()


# # -------------------------------------------------------------------------
# # Main evaluation
# # -------------------------------------------------------------------------
# if __name__ == "__main__":
#     bundle_root = "/content/drive/MyDrive/EEG2Video_data/processed/SubjectBundles"
#     ckpt_root   = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_predictor"

#     # List available checkpoints
#     ckpts = [f for f in os.listdir(ckpt_root) if f.startswith("semantic_predictor_") and f.endswith(".pt")]
#     print("\nAvailable checkpoints:")
#     for i, ck in enumerate(ckpts):
#         print(f"{i}: {ck}")

#     choice = input("\nSelect checkpoint index: ").strip()
#     ckpt_file = ckpts[int(choice)]
#     tag = ckpt_file.replace("semantic_predictor_", "").replace(".pt", "")
#     ckpt_path = os.path.join(ckpt_root, ckpt_file)
#     scaler_path = os.path.join(ckpt_root, f"scaler_{tag}.pkl")

#     print(f"\nLoading checkpoint: {ckpt_file}")
#     print(f"Loading scaler: scaler_{tag}.pkl")

#     # Pick relevant test npz files
#     all_bundles = sorted([f for f in os.listdir(bundle_root) if f.endswith("_test.npz")])
#     if tag == "all":
#         test_npz = [os.path.join(bundle_root, f) for f in all_bundles]
#     else:
#         selected_subjects = tag.split("_")
#         test_npz = [os.path.join(bundle_root, f"{s}_test.npz") for s in selected_subjects if f"{s}_test.npz" in all_bundles]

#     print(f"Using {len(test_npz)} test .npz files for evaluation")

#     # Load scaler
#     with open(scaler_path, "rb") as f:
#         scaler = pickle.load(f)

#     # Build dataset
#     dataset = NPZSemanticDataset(test_npz, scaler, max_samples=200)  # set None for full eval
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

#     # Load model
#     model = SemanticPredictor().cuda()
#     checkpoint = torch.load(ckpt_path, map_location="cuda")
#     model.load_state_dict(checkpoint["state_dict"])
#     model.eval()

#     # Evaluation loop
#     total_mse, total_cos, count = 0.0, 0.0, 0
#     with torch.no_grad():
#         for eeg, txt in tqdm(dataloader, desc="Evaluating"):
#             eeg, txt = eeg.cuda(non_blocking=True), txt.cuda(non_blocking=True)
#             pred = model(eeg)
#             mse = F.mse_loss(pred, txt, reduction="none").mean(dim=1)
#             cos = F.cosine_similarity(pred, txt, dim=1)
#             total_mse += mse.sum().item()
#             total_cos += cos.sum().item()
#             count += eeg.size(0)

#     avg_mse = total_mse / count
#     avg_cos = total_cos / count
#     print("\n=== Semantic Predictor Evaluation ===")
#     print(f"Checkpoint used: {ckpt_file}")
#     print(f"Samples evaluated: {count}")
#     print(f"Average MSE: {avg_mse:.6f}")
#     print(f"Average Cosine similarity: {avg_cos:.6f}")

#     # Inspect first few examples
#     for idx in range(min(3, len(dataset))):
#         eeg, txt = dataset[idx]
#         with torch.no_grad():
#             pred = model(eeg.unsqueeze(0).cuda()).squeeze(0)
#         mse = F.mse_loss(pred, txt.cuda()).item()
#         cos = cosine_similarity(pred, txt.cuda())
#         print(f"Sample {idx}: MSE={mse:.6f}, Cosine={cos:.6f}")

# ==========================================
# Semantic Predictor Evaluation
# ==========================================

# === Standard libraries ===
import os
import pickle

# === Third-party libraries ===
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# === Repo imports ===
from train_semantic import SemanticPredictor


# ==========================================
# Dataset wrapper (EEG_DE + BLIP_embeddings via test list)
# ==========================================
class ListSemanticDataset(Dataset):
    """
    Loads EEG_DE (input) and BLIP_embeddings (target) based on EEG_DE/test_list.txt.

    Each line in test_list.txt looks like:
      BlockX/classYY_clipZZ.npy

    Input  → EEG_DE/BlockX/classYY_clipZZ.npy, flattened to (310,)
    Target → BLIP_embeddings/BlockX/classYY_clipZZ.npy, flattened to (77*768,)
    """
    def __init__(self, test_list_path, eeg_root, blip_root, scaler, max_samples=None):
        self.samples = []
        self.scaler  = scaler

        with open(test_list_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        for rel_path in lines:
            eeg_path  = os.path.join(eeg_root, rel_path)
            blip_path = os.path.join(blip_root, rel_path)

            eeg  = np.load(eeg_path)   # (62,5)
            blip = np.load(blip_path)  # (77,768)

            eeg_flat  = eeg.reshape(-1)   # (310,)
            blip_flat = blip.reshape(-1)  # (77*768,)

            self.samples.append((eeg_flat, blip_flat))

        if max_samples:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        eeg, txt = self.samples[idx]
        eeg = self.scaler.transform([eeg])[0]
        return torch.tensor(eeg, dtype=torch.float32), torch.tensor(txt, dtype=torch.float32)


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
        print(f"  [{i}] {ck}")

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

    # === Build dataset from EEG_DE test list ===
    dataset    = ListSemanticDataset(test_list_path, eeg_root, blip_root, scaler, max_samples=200)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # === Load model ===
    model = SemanticPredictor().cuda()
    checkpoint = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # === Evaluation ===
    total_mse, total_cos, count = 0.0, 0.0, 0
    with torch.no_grad():
        for eeg, txt in tqdm(dataloader, desc="Evaluating"):
            eeg, txt = eeg.cuda(non_blocking=True), txt.cuda(non_blocking=True)
            pred = model(eeg)
            mse = F.mse_loss(pred, txt, reduction="none").mean(dim=1)
            cos = F.cosine_similarity(pred, txt, dim=1)
            total_mse += mse.sum().item()
            total_cos += cos.sum().item()
            count += eeg.size(0)

    avg_mse = total_mse / count
    avg_cos = total_cos / count
    print("\n=== Semantic Predictor Evaluation ===")
    print(f"Checkpoint used: {ckpt_file}")
    print(f"Samples evaluated: {count}")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Average Cosine similarity: {avg_cos:.6f}")

    # === Inspect first few examples ===
    for idx in range(min(3, len(dataset))):
        eeg, txt = dataset[idx]
        with torch.no_grad():
            pred = model(eeg.unsqueeze(0).cuda()).squeeze(0)
        mse = F.mse_loss(pred, txt.cuda()).item()
        cos = cosine_similarity(pred, txt.cuda())
        print(f"Sample {idx}: MSE={mse:.6f}, Cosine={cos:.6f}")
