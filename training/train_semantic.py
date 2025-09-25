# ==========================================
# train_semantic_kfold_collapsecheck.py
# ==========================================
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# === Repo imports ===
repo_root = "/content/EEG2Video_my_version"
sys.path.append(repo_root)
from core_files.models import eegnet, conformer, glfnet_mlp

# -------------------------------------------------
# Fusion model (fixed 4 encoders)
# -------------------------------------------------
class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleDict({
            "de": glfnet_mlp(out_dim=128, emb_dim=64, input_dim=62*5),
            "psd": glfnet_mlp(out_dim=128, emb_dim=64, input_dim=62*5),
            "windows": eegnet(out_dim=128, C=62, T=100),
            "segments": conformer(out_dim=128, C=62, T=400),
        })
        self.total_dim = 128 * 4
        self.classifier = nn.Linear(self.total_dim, 40)
    def forward(self, inputs, return_feats=False):
        feats = [enc(inputs[name]) for name, enc in self.encoders.items()]
        fused = torch.cat(feats, dim=-1)
        if return_feats:
            return fused
        return self.classifier(fused)

# -------------------------------------------------
# Semantic predictor
# -------------------------------------------------
class SemanticPredictor(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=[1024, 2048], out_dim=77*768):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# -------------------------------------------------
# Dataset
# -------------------------------------------------
class EEG2BLIPDataset(Dataset):
    def __init__(self, Xs, blip_embs):
        self.Xs, self.blip_embs = Xs, blip_embs
        self.keys = list(Xs.keys())
    def __len__(self):
        return len(self.blip_embs)
    def __getitem__(self, idx):
        out = {}
        for k in self.keys:
            x = torch.tensor(self.Xs[k][idx], dtype=torch.float32)
            if k in ["windows", "segments"]:
                x = x.unsqueeze(0)  # CNN input format: (1, 62, T)
            out[k] = x
        target = torch.tensor(self.blip_embs[idx], dtype=torch.float32)
        return out, target

# -------------------------------------------------
# Train 1 fold (with collapse check)
# -------------------------------------------------
def train_one_fold(fusion, predictor, train_loader, val_loader, device, num_epochs=30, lr=1e-4):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(predictor.parameters(), lr=lr)
    best_val = 1e9

    for epoch in range(num_epochs):
        fusion.eval(); predictor.train()
        for Xs, blip in train_loader:
            Xs = {k: v.to(device) for k,v in Xs.items()}
            blip = blip.to(device)
            with torch.no_grad():
                feats = fusion(Xs, return_feats=True)
            pred = predictor(feats)
            loss = criterion(pred, blip)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        predictor.eval(); val_loss = 0; count = 0
        all_preds = []
        with torch.no_grad():
            for Xs, blip in val_loader:
                Xs = {k: v.to(device) for k,v in Xs.items()}
                blip = blip.to(device)
                feats = fusion(Xs, return_feats=True)
                pred = predictor(feats)
                val_loss += criterion(pred, blip).item(); count += 1
                all_preds.append(pred.cpu())
        val_loss /= max(1, count)

        preds = torch.cat(all_preds, dim=0)  # (N, 59136)
        var_per_dim = preds.var(dim=0).mean().item()
        var_across_samples = preds.var(dim=1).mean().item()

        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss {val_loss:.4f}, "
              f"PredVar(dim) {var_per_dim:.6f}, PredVar(samples) {var_across_samples:.6f}")

        if val_loss < best_val:
            best_val = val_loss

    return best_val

# -------------------------------------------------
# K-Fold CV
# -------------------------------------------------
def run_cv(subj_name, drive_root, device):
    # === Load EEG ===
    de       = np.load(os.path.join(drive_root, "EEG_DE", f"{subj_name}.npy"))         # (7,40,5,62,5)
    psd      = np.load(os.path.join(drive_root, "EEG_PSD", f"{subj_name}.npy"))        # (7,40,5,62,5)
    windows  = np.load(os.path.join(drive_root, "EEG_windows", f"{subj_name}.npy"))    # (7,40,5,7,62,100)
    segments = np.load(os.path.join(drive_root, "EEG_segments", f"{subj_name}.npy"))  # (7,40,5,62,400)

    blip = np.load(os.path.join(drive_root, "BLIP_embeddings", "BLIP_embeddings.npy"))  # (7,40,5,77,768)
    blip = blip.reshape(7, 40, 5, -1)  # → (7,40,5,59136)

    assert blip.shape == (7, 40, 5, 77 * 768), f"BLIP reshape failed, got {blip.shape}"

    all_val_losses, all_test_losses = [], []

    for test_block in range(7):
        val_block = (test_block - 1) % 7
        train_blocks = [i for i in range(7) if i not in [test_block, val_block]]

        def collect(blocks):
            Xs = {"de": [], "psd": [], "windows": [], "segments": []}
            Ys = []
            for b in blocks:
                for c in range(40):
                    for k in range(5):
                        Xs["de"].append(de[b,c,k])
                        Xs["psd"].append(psd[b,c,k])
                        Xs["windows"].append(windows[b,c,k].mean(0))  # (7,62,100) → (62,100)
                        Xs["segments"].append(segments[b,c,k])
                        Ys.append(blip[b,c,k])
            Xs = {k: np.array(v) for k,v in Xs.items()}
            Ys = np.array(Ys)
            return Xs, Ys

        X_train, Y_train = collect(train_blocks)
        X_val,   Y_val   = collect([val_block])
        X_test,  Y_test  = collect([test_block])

        train_loader = DataLoader(EEG2BLIPDataset(X_train,Y_train), batch_size=32, shuffle=True)
        val_loader   = DataLoader(EEG2BLIPDataset(X_val,Y_val), batch_size=32)
        test_loader  = DataLoader(EEG2BLIPDataset(X_test,Y_test), batch_size=32)

        # === Load frozen fusion encoder ===
        fusion = FusionModel().to(device)
        ckpt_dir = "/content/drive/MyDrive/EEG2Video_checkpoints/fusion_checkpoints"
        ckpt_path = os.path.join(ckpt_dir, f"fusion_checkpoint_{subj_name}.pt")
        fusion.load_state_dict(torch.load(ckpt_path, map_location=device))
        for p in fusion.parameters(): p.requires_grad = False

        # === Train predictor ===
        predictor = SemanticPredictor(input_dim=fusion.total_dim).to(device)
        val_loss = train_one_fold(fusion, predictor, train_loader, val_loader, device)

        # === Test eval ===
        predictor.eval(); test_loss = 0; count = 0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for Xs, blip in test_loader:
                Xs = {k: v.to(device) for k,v in Xs.items()}
                blip = blip.to(device)
                feats = fusion(Xs, return_feats=True)
                pred = predictor(feats)
                test_loss += criterion(pred, blip).item(); count += 1
        test_loss /= max(1, count)

        all_val_losses.append(val_loss)
        all_test_losses.append(test_loss)
        print(f"Fold {test_block+1}/7: Val {val_loss:.4f}, Test {test_loss:.4f}")

    print("\n=== Final Results ===")
    print(f"Avg Val Loss:  {np.mean(all_val_losses):.4f}")
    print(f"Avg Test Loss: {np.mean(all_test_losses):.4f}")

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    drive_root = "/content/drive/MyDrive/EEG2Video_data/processed"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subj_name = input("Enter subject name: ").strip()
    run_cv(subj_name, drive_root, device)

if __name__ == "__main__":
    main()
