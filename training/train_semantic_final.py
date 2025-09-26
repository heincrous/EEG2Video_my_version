# ==========================================
# Semantic predictor (EEG → BLIP)
# Only best checkpoint saved
# ==========================================
import os, itertools, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------
# Semantic Predictor MLP
# -------------------------------------------------
class SemanticPredictor(nn.Module):
    def __init__(self, in_dim, out_shape=(77,768), use_dropout=False):
        super().__init__()
        out_dim = out_shape[0] * out_shape[1]
        layers = [nn.Linear(in_dim, 10000), nn.ReLU()]
        for _ in range(3):
            layers.append(nn.Linear(10000, 10000))
            layers.append(nn.ReLU())
            if use_dropout:
                layers.append(nn.Dropout(0.3))
        layers.append(nn.Linear(10000, out_dim))
        self.mlp = nn.Sequential(*layers)
        self.out_shape = out_shape
    def forward(self, x):
        out = self.mlp(x)
        return out.view(-1, *self.out_shape)

# -------------------------------------------------
# Dataset wrapper
# -------------------------------------------------
class EEG2BLIPDataset(Dataset):
    def __init__(self, feat_dict, blip_feats):
        self.scalers = {}
        proc_feats = []
        for k,v in feat_dict.items():
            scaler = StandardScaler().fit(v)
            v_scaled = scaler.transform(v)
            self.scalers[k] = scaler
            proc_feats.append(v_scaled)
        X = np.concatenate(proc_feats, axis=1)
        self.X = X.astype(np.float32)
        self.Y = blip_feats.astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])

# -------------------------------------------------
# Loss functions
# -------------------------------------------------
def cosine_loss(pred, target):
    pred = F.normalize(pred.view(pred.size(0), -1), dim=-1)
    target = F.normalize(target.view(target.size(0), -1), dim=-1)
    return 1 - (pred * target).sum(-1).mean()

def contrastive_loss(pred, target, temperature=0.07):
    pred = F.normalize(pred.view(pred.size(0), -1), dim=-1)
    target = F.normalize(target.view(target.size(0), -1), dim=-1)
    logits = pred @ target.t() / temperature
    labels = torch.arange(pred.size(0), device=pred.device)
    return F.cross_entropy(logits, labels)

# -------------------------------------------------
# Train loop with early stopping
# -------------------------------------------------
def train(model, train_loader, val_loader, device, cfg, feat_name, subj_name):
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["epochs"] * len(train_loader)
    )
    best_val = float("inf")
    patience, wait = 20, 0
    ckpt_dir = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"semantic_predictor_{subj_name}_{feat_name}_best.pt")

    for ep in range(cfg["epochs"]):
        # Training
        model.train(); total_loss = 0
        for eeg, blip in train_loader:
            eeg, blip = eeg.to(device), blip.to(device)
            pred = model(eeg)
            if cfg["loss_type"] == "mse":
                loss = F.mse_loss(pred, blip)
            elif cfg["loss_type"] == "cosine":
                loss = cosine_loss(pred, blip)
            elif cfg["loss_type"] == "contrastive":
                loss = contrastive_loss(pred, blip)
            if cfg["use_var_reg"]:
                var_loss = 1.0 / (pred.view(pred.size(0), -1).var(dim=0).mean() + 1e-6)
                loss += 0.01 * var_loss
            opt.zero_grad(); loss.backward(); opt.step(); scheduler.step()
            total_loss += loss.item()

        # Validation
        model.eval(); val_loss=0; preds=[]
        with torch.no_grad():
            for eeg, blip in val_loader:
                eeg, blip = eeg.to(device), blip.to(device)
                pred = model(eeg)
                if cfg["loss_type"] == "mse":
                    val_loss += F.mse_loss(pred, blip).item()
                elif cfg["loss_type"] == "cosine":
                    val_loss += cosine_loss(pred, blip).item()
                else:
                    val_loss += contrastive_loss(pred, blip).item()
                preds.append(pred.view(pred.size(0), -1).cpu())
        val_loss /= len(val_loader)

        print(f"[{feat_name}] Epoch {ep+1}, "
              f"TrainLoss {total_loss/len(train_loader):.4f}, "
              f"ValLoss {val_loss:.4f}")

        # Checkpointing
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save({'state_dict': model.state_dict()}, ckpt_path)
            print(f"New best checkpoint saved at epoch {ep+1}")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered")
                break

    return ckpt_path

# -------------------------------------------------
# Evaluate on held-out test set
# -------------------------------------------------
def evaluate(model, test_loader, device, cfg, feat_name):
    model.eval(); test_loss=0; preds=[]
    with torch.no_grad():
        for eeg, blip in test_loader:
            eeg, blip = eeg.to(device), blip.to(device)
            pred = model(eeg)
            if cfg["loss_type"] == "mse":
                test_loss += F.mse_loss(pred, blip).item()
            elif cfg["loss_type"] == "cosine":
                test_loss += cosine_loss(pred, blip).item()
            else:
                test_loss += contrastive_loss(pred, blip).item()
            preds.append(pred.view(pred.size(0), -1).cpu())
    preds = torch.cat(preds, dim=0)
    stats = {
        "test_loss": test_loss/len(test_loader),
        "pred_var_dim": preds.var(dim=0).mean().item(),
        "pred_var_samp": preds.var(dim=1).mean().item()
    }
    print(f"[{feat_name}] TestLoss {stats['test_loss']:.4f}, "
          f"Var(dim) {stats['pred_var_dim']:.6f}, "
          f"Var(samples) {stats['pred_var_samp']:.6f}")
    return stats

# -------------------------------------------------
# Utility to load features
# -------------------------------------------------
def load_features(choice_list, subj_name, drive_root):
    feat_dict = {}
    if "de" in choice_list:
        de = np.load(os.path.join(drive_root,"EEG_DE",f"{subj_name}.npy"))
        feat_dict["de"] = de.reshape(-1, 62*5)
    if "psd" in choice_list:
        psd = np.load(os.path.join(drive_root,"EEG_PSD",f"{subj_name}.npy"))
        feat_dict["psd"] = psd.reshape(-1, 62*5)
    if "windows" in choice_list:
        win = np.load(os.path.join(drive_root,"EEG_windows",f"{subj_name}.npy"))
        feat_dict["windows"] = win.mean(3).reshape(-1, 62*100)
    if "segments" in choice_list:
        seg = np.load(os.path.join(drive_root,"EEG_segments",f"{subj_name}.npy"))
        feat_dict["segments"] = seg.reshape(-1, 62*400)
    return feat_dict

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subj_name = "sub1"
    drive_root = "/content/drive/MyDrive/EEG2Video_data/processed"

    mode = input("Run mode (one/max): ").strip().lower()
    if mode == "one":
        choices = input("Select features (comma separated from: de, psd, windows, segments, all): ")
        choices = [c.strip() for c in choices.split(",")]
        if "all" in choices: choices = ["de","psd","windows","segments"]
        combos = [choices]
    else:
        base = ["de","psd","windows","segments"]
        combos = [list(combo) for r in range(1, len(base)+1) for combo in itertools.combinations(base, r)]

    blip = np.load(os.path.join(drive_root,"BLIP_embeddings","BLIP_embeddings.npy"))
    blip_flat = blip.reshape(-1, 77, 768)

    for combo in combos:
        feat_dict = load_features(combo, subj_name, drive_root)
        ds = EEG2BLIPDataset(feat_dict, blip_flat)

        block_size = 200
        train_blocks, val_blocks, test_blocks = [0,1,2,3,4], [5], [6]
        def block_indices(block_list):
            return [i for b in block_list for i in range(b*block_size,(b+1)*block_size)]
        train_idx, val_idx, test_idx = block_indices(train_blocks), block_indices(val_blocks), block_indices(test_blocks)

        train_loader = DataLoader(torch.utils.data.Subset(ds, train_idx), batch_size=CFG["batch_size"], shuffle=True)
        val_loader   = DataLoader(torch.utils.data.Subset(ds, val_idx), batch_size=CFG["batch_size"])
        test_loader  = DataLoader(torch.utils.data.Subset(ds, test_idx), batch_size=CFG["batch_size"])

        in_dim = ds.X.shape[1]
        model = SemanticPredictor(in_dim=in_dim, out_shape=(77,768), use_dropout=CFG["use_dropout"]).to(device)
        feat_name = "_".join(combo)

        ckpt_path = train(model, train_loader, val_loader, device, CFG, feat_name, subj_name)
        model.load_state_dict(torch.load(ckpt_path)['state_dict'])
        evaluate(model, test_loader, device, CFG, feat_name)

# -------------------------------------------------
# Config
# -------------------------------------------------
CFG = {
    "loss_type": "mse",
    "use_var_reg": False,
    "use_dropout": False,
    "lr": 5e-4,
    "batch_size": 128,
    "epochs": 200,
}

if __name__ == "__main__":
    main()

# DE & PSD together is redundant (both frequency domain)
# contrastive makes variance blow-up
