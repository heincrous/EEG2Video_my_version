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
# EEG-BLIP Dataset (patched to avoid BLIP duplication)
# ==========================================
class EEGTextDataset(Dataset):
    def __init__(self, bundle_file, feature_type="windows", train=True):
        bundle = np.load(bundle_file, allow_pickle=True)
        eeg_all = bundle["EEG_data"].item()
        blip_emb = bundle["BLIP_embeddings"]   # shape [7,40,5,77,768]

        X, Y = [], []
        blocks = range(0,6) if train else [6]

        for subj, subj_data in eeg_all.items():
            if feature_type == "windows":
                eeg_feature = subj_data["EEG_windows"]   # [7,40,5,7,62,100]
            elif feature_type == "segments":
                eeg_feature = subj_data["EEG_segments"]  # [7,40,5,62,400]
            else:
                raise ValueError("Invalid feature type")

            for b in blocks:
                for c in range(40):
                    for k in range(5):
                        eeg_clip = eeg_feature[b, c, k]
                        # only fetch BLIP once, not per subject
                        blip_clip = blip_emb[b, c, k].flatten()

                        if feature_type == "windows":
                            for w in range(7):
                                X.append(eeg_clip[w])  # (62,100)
                                Y.append(blip_clip)    # same caption for all 7 windows
                        else:  # segments
                            X.append(eeg_clip)        # (62,400)
                            Y.append(blip_clip)

        X = np.array(X)
        if feature_type == "windows":
            self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,62,100)
        else:
            self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,62,400)
        self.Y = torch.tensor(np.array(Y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

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
