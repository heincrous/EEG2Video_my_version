# ==========================================
# train_semantic_interactive.py
# ==========================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# === Imports for encoders ===
from braindecode.models import ShallowFBCSPNet, Deep4Net, EEGNetv4

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
    def __init__(self, bundle_file, train=True):
        bundle = np.load(bundle_file, allow_pickle=True)
        eeg_all = bundle["EEG_data"].item()          # dict of subjects
        blip_emb = bundle["BLIP_embeddings"]         # shared across subjects

        X, Y = [], []
        for subj, subj_data in eeg_all.items():
            eeg_windows = subj_data["EEG_windows"]   # [7,40,5,7,62,100]

            blocks = range(0,6) if train else [6]
            for b in blocks:
                for c in range(40):
                    for k in range(5):
                        eeg_clip = eeg_windows[b, c, k]
                        blip_clip = blip_emb[b, c, k]
                        for w in range(7):
                            X.append(eeg_clip[w])         # (62,100)
                            Y.append(blip_clip.flatten()) # (77*768,)

        # final tensor: (N,62,100)
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
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

    encoders = {
        "shallownet": ShallowFBCSPNet,
        "deepnet": Deep4Net,
        "eegnet": EEGNetv4,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mode_choice == "dry":
        eeg = torch.randn(1, 62, 100).to(device)  # dummy input (B,C,T)

        for name, EncoderClass in encoders.items():
            encoder = EncoderClass(
                n_chans=62,
                n_outputs=512,
                input_window_samples=100,
                final_conv_length="auto"
            ).to(device)
            semantic = SemanticPredictor(input_dim=512).to(device)

            enc_out = encoder(eeg)
            out = semantic(enc_out)

            print(f"[{name}] EEG input: {eeg.shape} -> Encoder out: {enc_out.shape} -> Predicted out: {out.shape}")

        print("Dry run completed for ShallowNet, DeepNet, EEGNet")
        return

    if mode_choice == "train":
        encoder_choice = input("Select encoder (shallownet, deepnet, eegnet): ").strip()
        epochs_choice  = int(input("Enter number of epochs: "))

        if encoder_choice not in encoders:
            raise ValueError("Invalid encoder choice")

        EncoderClass = encoders[encoder_choice]
        encoder = EncoderClass(
            n_chans=62,
            n_outputs=512,
            n_times=100,
            final_conv_length="auto"
        ).to(device)

        semantic = SemanticPredictor(input_dim=512).to(device)

        train_ds = EEGTextDataset(bundle_file, train=True)
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
                enc_out = encoder(eeg)
                pred = semantic(enc_out)
                loss = criterion(pred, text)
                loss.backward()
                optimizer.step()
                total += loss.item()
            avg_loss = total/len(train_loader)
            print(f"Epoch {epoch+1}/{epochs_choice} Loss: {avg_loss:.4f}")

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
