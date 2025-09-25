# ==========================================
# train_semantic_interactive.py (patched to avoid BLIP duplication)
# ==========================================

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# === Repo imports ===
repo_root = "/content/EEG2Video_my_version"
sys.path.append(repo_root)
from core_files.models import shallownet, deepnet, eegnet, tsconv, conformer

# ==========================================
# Semantic Predictor (MLP)
# ==========================================
class SemanticPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[1024, 2048, 1024], out_dim=77*768):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ==========================================
# EEG-BLIP Dataset (lazy loading, no preloading)
# ==========================================
class EEGTextDataset(Dataset):
    def __init__(self, bundle_file, feature_type="windows", train=True):
        bundle = np.load(bundle_file, allow_pickle=True)
        self.eeg_all = bundle["EEG_data"].item()
        self.blip_emb = bundle["BLIP_embeddings"]
        self.feature_type = feature_type
        self.blocks = range(0,6) if train else [6]

        # Build index list of (subj, block, concept, clip, window)
        self.index = []
        for subj, subj_data in self.eeg_all.items():
            if feature_type not in ["windows", "segments"]:
                raise ValueError("Invalid feature type")

            for b in self.blocks:
                for c in range(40):
                    for k in range(5):
                        if feature_type == "windows":
                            for w in range(7):   # 7 sliding windows
                                self.index.append((subj, b, c, k, w))
                        else:  # segments
                            self.index.append((subj, b, c, k, None))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        subj, b, c, k, w = self.index[idx]
        subj_data = self.eeg_all[subj]

        if self.feature_type == "windows":
            eeg_clip = subj_data["EEG_windows"][b, c, k][w]   # (62,100)
            eeg_clip = torch.tensor(eeg_clip, dtype=torch.float32).unsqueeze(0)  # (1,62,100)
        else:  # segments
            eeg_clip = subj_data["EEG_segments"][b, c, k]     # (62,400)
            eeg_clip = torch.tensor(eeg_clip, dtype=torch.float32).unsqueeze(0)  # (1,62,400)

        blip_clip = self.blip_emb[b, c, k].flatten()
        blip_clip = torch.tensor(blip_clip, dtype=torch.float32)

        return eeg_clip, blip_clip

# ==========================================
# Main
# ==========================================
def main():
    drive_root = "/content/drive/MyDrive/EEG2Video_data/processed"
    bundle_file = f"{drive_root}/BLIP_EEG_bundle.npz"

    mode_choice = input("Select mode (train/dry): ").strip()
    feature_type = input("Select feature type (windows/segments): ").strip()

    encoders_all = {
        "shallownet": shallownet,
        "deepnet": deepnet,
        "eegnet": eegnet,
        "tsconv": tsconv,
        "conformer": conformer
    }
    if feature_type == "windows":
        valid_encoders = {k:v for k,v in encoders_all.items() if k != "deepnet"}  # deepnet fails at T=100
        C, T = 62, 100
    elif feature_type == "segments":
        valid_encoders = encoders_all
        C, T = 62, 400
    else:
        raise ValueError("Invalid feature type")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mode_choice == "dry":
        eeg = torch.randn(1, 1, C, T).to(device)
        for name, EncoderClass in valid_encoders.items():
            encoder = EncoderClass(out_dim=512, C=C, T=T).to(device)
            semantic = SemanticPredictor(input_dim=512).to(device)
            out = semantic(encoder(eeg))
            print(f"[{feature_type}] {name}: EEG {eeg.shape} -> Output {out.shape}")
        print("Dry run completed for all valid encoder-feature combos")
        return

    if mode_choice == "train":
        print(f"Available encoders for {feature_type}: {list(valid_encoders.keys())}")
        encoder_choice = input("Select encoder: ").strip()
        epochs_choice  = int(input("Enter number of epochs: "))

        if encoder_choice not in valid_encoders:
            raise ValueError("Invalid encoder choice")

        EncoderClass = valid_encoders[encoder_choice]
        encoder = EncoderClass(out_dim=512, C=C, T=T).to(device)
        semantic = SemanticPredictor(input_dim=512).to(device)

        train_ds = EEGTextDataset(bundle_file, feature_type=feature_type, train=True)
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(list(encoder.parameters())+list(semantic.parameters()), lr=5e-4)

        ckpt_dir = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)

        for epoch in range(epochs_choice):
            total = 0
            for eeg, text in train_loader:
                eeg, text = eeg.to(device), text.to(device)
                optimizer.zero_grad()
                pred = semantic(encoder(eeg))
                loss = criterion(pred, text)
                loss.backward()
                optimizer.step()
                total += loss.item()
            avg_loss = total/len(train_loader)
            print(f"Epoch {epoch+1}/{epochs_choice} Loss: {avg_loss:.4f}")

        ckpt_path = os.path.join(ckpt_dir, f"semantic_{feature_type}_{encoder_choice}.pt")
        torch.save({
            'epoch': epochs_choice,
            'encoder_state_dict': encoder.state_dict(),
            'semantic_state_dict': semantic.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'final_loss': avg_loss
        }, ckpt_path)
        print(f"Final checkpoint saved to {ckpt_path}")

if __name__ == "__main__":
    main()
