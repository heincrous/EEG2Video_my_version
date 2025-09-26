# ==========================================
# train_semantic_from_DE.py
# Author-style semantic predictor (EEG → BLIP)
# ==========================================
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# -------------------------------------------------
# Semantic Predictor MLP (authors' CLIP-like)
# -------------------------------------------------
class SemanticPredictor(nn.Module):
    def __init__(self, in_dim=310, out_shape=(77,768)):
        super().__init__()
        out_dim = out_shape[0] * out_shape[1]
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, out_dim),
        )
        self.out_shape = out_shape

    def forward(self, x):
        out = self.mlp(x)
        return out.view(-1, *self.out_shape)

# -------------------------------------------------
# Dataset wrapper
# -------------------------------------------------
class EEG2BLIPDataset(Dataset):
    def __init__(self, eeg_feats, blip_feats):
        scaler = StandardScaler().fit(eeg_feats)
        eeg_feats = scaler.transform(eeg_feats)
        self.X = eeg_feats.astype(np.float32)
        self.Y = blip_feats.astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])

# -------------------------------------------------
# Train loop
# -------------------------------------------------
def train(model, train_loader, val_loader, device, epochs=50, lr=5e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs*len(train_loader))
    for ep in range(epochs):
        model.train(); total_loss = 0
        for eeg, blip in train_loader:
            eeg, blip = eeg.to(device), blip.to(device)
            pred = model(eeg)
            loss = F.mse_loss(pred, blip)
            opt.zero_grad(); loss.backward(); opt.step(); scheduler.step()
            total_loss += loss.item()
        # validation
        model.eval(); val_loss=0; preds=[]
        with torch.no_grad():
            for eeg, blip in val_loader:
                eeg, blip = eeg.to(device), blip.to(device)
                pred = model(eeg)
                val_loss += F.mse_loss(pred, blip).item()
                preds.append(pred.view(pred.size(0), -1).cpu())
        preds = torch.cat(preds, dim=0)
        print(f"Epoch {ep+1}, TrainLoss {total_loss/len(train_loader):.4f}, "
              f"ValLoss {val_loss/len(val_loader):.4f}, "
              f"Var(dim) {preds.var(dim=0).mean().item():.6f}, "
              f"Var(samples) {preds.var(dim=1).mean().item():.6f}")

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subj_name = "sub1"
    drive_root = "/content/drive/MyDrive/EEG2Video_data/processed"

    # load DE features (shape [7,40,5,62,5]) and flatten to (N,310)
    de = np.load(os.path.join(drive_root, "EEG_DE", f"{subj_name}.npy"))  # (7,40,5,62,5)
    N = de.shape[0]*de.shape[1]*de.shape[2]
    de_flat = de.reshape(N, 62*5)

    # load BLIP embeddings (shape [7,40,5,77,768]) and flatten to (N,77,768)
    blip = np.load(os.path.join(drive_root, "BLIP_embeddings", "BLIP_embeddings.npy"))
    blip_flat = blip.reshape(N, 77, 768)

    # dataset split
    ds = EEG2BLIPDataset(de_flat, blip_flat)
    n = len(ds); split=int(0.8*n)
    train_loader = DataLoader(torch.utils.data.Subset(ds, range(split)), batch_size=256, shuffle=True)
    val_loader   = DataLoader(torch.utils.data.Subset(ds, range(split,n)), batch_size=256)

    # model
    model = SemanticPredictor(in_dim=310, out_shape=(77,768)).to(device)
    train(model, train_loader, val_loader, device, epochs=400)

    # save
    torch.save({'state_dict': model.state_dict()}, 
               f"/content/drive/MyDrive/EEG2Video_checkpoints/semantic_predictor_{subj_name}.pt")

if __name__ == "__main__":
    main()
