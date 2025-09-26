# ==========================================
# train_semantic_kfold_collapsecheck.py (structured predictor + norm targets in dataset)
# ==========================================
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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
# Semantic predictor (structured output: [77,768])
# -------------------------------------------------
class SemanticPredictor(nn.Module):
    def __init__(self, input_dim=512, out_dim=(77,768)):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, out_dim[0]*out_dim[1])
        self.out_dim = out_dim
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, *self.out_dim)  # [B,77,768]

# -------------------------------------------------
# Dataset (normalise BLIP targets once here)
# -------------------------------------------------
class EEG2BLIPDataset(Dataset):
    def __init__(self, Xs, Ys):
        self.Xs = Xs
        Ys = Ys / (np.linalg.norm(Ys, axis=-1, keepdims=True) + 1e-8)
        self.Ys = Ys.astype(np.float32)
        self.keys = list(Xs.keys())
    def __len__(self):
        return len(self.Ys)
    def __getitem__(self, idx):
        out = {}
        for k in self.keys:
            x = torch.tensor(self.Xs[k][idx], dtype=torch.float32)
            if k in ["windows", "segments"]:
                x = x.unsqueeze(0)
            out[k] = x
        target = torch.tensor(self.Ys[idx], dtype=torch.float32)  # [77,768], already normed
        return out, target

# -------------------------------------------------
# Loss (MSE + cosine + variance reg)
# -------------------------------------------------
def semantic_loss(pred, target, alpha=0.5, beta=0.1):
    pred_norm = F.normalize(pred, dim=-1)  # normalise predictions
    mse = F.mse_loss(pred_norm, target)
    cos = 1 - F.cosine_similarity(
        pred_norm.flatten(1), target.flatten(1), dim=-1
    ).mean()
    var_loss = -pred_norm.var(dim=0).mean()
    return mse + alpha * cos + beta * var_loss

# -------------------------------------------------
# Train 1 fold
# -------------------------------------------------
def train_one_fold(fusion, predictor, train_loader, val_loader, device, subj_name,
                   num_epochs=30, lr=1e-3):
    optimizer = optim.AdamW(predictor.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(train_loader)
    )
    best_val = 1e9
    best_state_dict = None

    for epoch in range(num_epochs):
        fusion.eval(); predictor.train()
        for Xs, blip in train_loader:
            Xs = {k: v.to(device) for k,v in Xs.items()}
            blip = blip.to(device)
            with torch.no_grad():
                feats = fusion(Xs, return_feats=True)
            pred = predictor(feats)
            loss = semantic_loss(pred, blip)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            scheduler.step()

        predictor.eval(); val_loss = 0; count = 0
        all_preds = []
        with torch.no_grad():
            for Xs, blip in val_loader:
                Xs = {k: v.to(device) for k,v in Xs.items()}
                blip = blip.to(device)
                feats = fusion(Xs, return_feats=True)
                pred = predictor(feats)
                val_loss += semantic_loss(pred, blip).item(); count += 1
                all_preds.append(pred.cpu())
        val_loss /= max(1, count)

        preds = torch.cat(all_preds, dim=0)
        var_per_dim = preds.var(dim=0).mean().item()
        var_across_samples = preds.var(dim=1).mean().item()

        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss {val_loss:.4f}, "
              f"PredVar(dim) {var_per_dim:.6f}, PredVar(samples) {var_across_samples:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state_dict = predictor.state_dict()

    save_dir = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"semantic_checkpoint_{subj_name}.pt")
    torch.save(best_state_dict, save_path)
    print(f"Best semantic predictor saved to: {save_path}")
    return best_val

# -------------------------------------------------
# K-Fold CV
# -------------------------------------------------
def run_cv(subj_name, drive_root, device):
    de       = np.load(os.path.join(drive_root, "EEG_DE", f"{subj_name}.npy"))
    psd      = np.load(os.path.join(drive_root, "EEG_PSD", f"{subj_name}.npy"))
    windows  = np.load(os.path.join(drive_root, "EEG_windows", f"{subj_name}.npy"))
    segments = np.load(os.path.join(drive_root, "EEG_segments", f"{subj_name}.npy"))

    blip_raw = np.load(os.path.join(drive_root, "BLIP_embeddings", "BLIP_embeddings.npy"))
    assert blip_raw.shape == (7, 40, 5, 77, 768)

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
                        Xs["windows"].append(windows[b,c,k].mean(0))
                        Xs["segments"].append(segments[b,c,k])
                        Ys.append(blip_raw[b,c,k])  # [77,768]
            return {k: np.array(v) for k,v in Xs.items()}, np.array(Ys)

        print(f"\n--- Fold {test_block+1}/7 ---")
        X_train, Y_train = collect(train_blocks)
        X_val,   Y_val   = collect([val_block])
        X_test,  Y_test  = collect([test_block])

        train_loader = DataLoader(EEG2BLIPDataset(X_train, Y_train), batch_size=32, shuffle=True)
        val_loader   = DataLoader(EEG2BLIPDataset(X_val, Y_val), batch_size=32)
        test_loader  = DataLoader(EEG2BLIPDataset(X_test, Y_test), batch_size=32)

        fusion = FusionModel().to(device)
        fusion_ckpt = f"/content/drive/MyDrive/EEG2Video_checkpoints/fusion_checkpoints/fusion_checkpoint_{subj_name}.pt"
        fusion.load_state_dict(torch.load(fusion_ckpt, map_location=device))
        for p in fusion.parameters(): p.requires_grad = False

        predictor = SemanticPredictor(input_dim=fusion.total_dim).to(device)
        val_loss = train_one_fold(fusion, predictor, train_loader, val_loader, device, subj_name)

        # Test
        predictor.eval(); test_loss = 0; count = 0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for Xs, blip in test_loader:
                Xs = {k: v.to(device) for k,v in Xs.items()}; blip = blip.to(device)
                feats = fusion(Xs, return_feats=True)
                pred = predictor(feats)
                pred_norm = F.normalize(pred, dim=-1)
                test_loss += criterion(pred_norm, blip).item(); count += 1
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
