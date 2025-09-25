# ==========================================
# train_semantic_interactive.py
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
# EEG-BLIP Dataset
# ==========================================
class EEGTextDataset(Dataset):
    def __init__(self, eeg_file, blip_file, train=True):
        eeg_all = np.load(eeg_file, allow_pickle=True).item()
        blip_all = np.load(blip_file, allow_pickle=True)

        X, Y = [], []
        for subj, subj_data in eeg_all.items():
            eeg_windows = subj_data["EEG_windows"]   # [7,40,5,7,62,100]
            blip_emb = blip_all["BLIP_embeddings"]   # [7,40,5,77,768]

            blocks = range(0,6) if train else [6]
            for b in blocks:
                for c in range(40):
                    for k in range(5):
                        eeg_clip = eeg_windows[b,c,k]
                        blip_clip = blip_emb[b,c,k]
                        for w in range(7):
                            X.append(eeg_clip[w])         # (62,100)
                            Y.append(blip_clip.flatten()) # (77*768,)

        self.X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(1)
        self.Y = torch.tensor(np.array(Y), dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

# ==========================================
# Main
# ==========================================
def main():
    drive_root = "/content/drive/MyDrive/EEG2Video_data/processed"
    eeg_file   = f"{drive_root}/BLIP_EEG_bundle.npz"
    blip_file  = f"{drive_root}/BLIP_Video_bundle.npz"

    # === Interactive user input ===
    encoder_choice = input("Select encoder (shallownet, deepnet, eegnet, tsconv, conformer): ").strip()
    mode_choice    = input("Select mode (train/dry): ").strip()
    epochs_choice  = int(input("Enter number of epochs: "))

    encoders = {
        "shallownet": shallownet,
        "deepnet": deepnet,
        "eegnet": eegnet,
        "tsconv": tsconv,
        "conformer": conformer
    }
    if encoder_choice not in encoders:
        raise ValueError("Invalid encoder choice")

    EncoderClass = encoders[encoder_choice]
    encoder = EncoderClass(out_dim=512, C=62, T=100)
    semantic = SemanticPredictor(input_dim=512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, semantic = encoder.to(device), semantic.to(device)

    if mode_choice == "dry":
        ds = EEGTextDataset(eeg_file, blip_file, train=True)
        eeg, text = ds[0]
        eeg, text = eeg.unsqueeze(0).to(device), text.unsqueeze(0).to(device)
        out = semantic(encoder(eeg))
        print("Dry run OK")
        print("EEG input:", eeg.shape)
        print("Predicted output:", out.shape)
        return

    if mode_choice == "train":
        train_ds = EEGTextDataset(eeg_file, blip_file, train=True)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(list(encoder.parameters())+list(semantic.parameters()), lr=1e-3)

        # === Checkpoint directory ===
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

        # === Save only the final checkpoint ===
        ckpt_path = os.path.join(ckpt_dir, "semantic_final.pt")
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
