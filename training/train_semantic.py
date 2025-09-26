# ==========================================
# train_semantic_experiments.py
# Minimal baseline + toggles + multi-task
# ==========================================
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

repo_root = "/content/EEG2Video_my_version"
sys.path.append(repo_root)
from core_files.models import eegnet, conformer, glfnet_mlp

# -------------------------------------------------
# Fusion model (intact, with classifier)
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
    def __init__(self, input_dim=512, out_dim=(77,768)):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, out_dim[0]*out_dim[1])
        self.out_dim = out_dim
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, *self.out_dim)

# -------------------------------------------------
# Dataset
# -------------------------------------------------
class EEG2BLIPDataset(Dataset):
    def __init__(self, Xs, Ys, normalise_targets=False):
        self.Xs = Xs
        if normalise_targets:
            Ys = Ys / (np.linalg.norm(Ys, axis=-1, keepdims=True) + 1e-8)
        self.Ys = Ys.astype(np.float32)
        self.keys = list(Xs.keys())
    def __len__(self): return len(self.Ys)
    def __getitem__(self, idx):
        out = {}
        for k in self.keys:
            x = torch.tensor(self.Xs[k][idx], dtype=torch.float32)
            if k in ["windows","segments"]:
                x = x.unsqueeze(0)
            out[k] = x
        target = torch.tensor(self.Ys[idx], dtype=torch.float32)
        return out, target

# -------------------------------------------------
# Loss builder
# -------------------------------------------------
def build_loss(use_mse=True, use_cosine=False, use_var=False, use_infonce=False,
               alpha=0.5, beta=0.01, tau=0.07, normalise_preds=False,
               use_multitask=False, lambda_cls=0.1):
    ce_loss = nn.CrossEntropyLoss()

    def loss_fn(pred, target, feats=None, fusion=None, Xs=None):
        # pred: semantic predictor output [B,77,768]
        if normalise_preds:
            pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)

        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        loss = 0.0
        if use_mse:
            loss += F.mse_loss(pred, target)
        if use_cosine:
            cos = 1 - F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()
            loss += alpha * cos
        if use_var:
            var_loss = -pred_flat.var(dim=0).mean()
            loss += beta * var_loss
        if use_infonce:
            logits = pred_flat @ target_flat.T / tau
            labels = torch.arange(pred_flat.size(0), device=pred.device)
            infonce = F.cross_entropy(logits, labels)
            loss += infonce

        if use_multitask and fusion is not None and feats is not None:
            # classifier logits from fusion
            cls_logits = fusion.classifier(feats)
            labels = torch.arange(cls_logits.size(0), device=cls_logits.device) % 40
            cls_loss = ce_loss(cls_logits, labels)
            loss += lambda_cls * cls_loss

        return loss
    return loss_fn

# -------------------------------------------------
# Train loop
# -------------------------------------------------
def train_one_fold(fusion, predictor, train_loader, val_loader, device,
                   loss_fn, epochs=10, lr=1e-3,
                   use_multitask=False):
    opt = optim.AdamW(predictor.parameters(), lr=lr)
    for ep in range(epochs):
        predictor.train()
        for Xs, Ys in train_loader:
            Xs = {k:v.to(device) for k,v in Xs.items()}
            Ys = Ys.to(device)
            with torch.no_grad(): feats = fusion(Xs, return_feats=True)
            pred = predictor(feats)
            loss = loss_fn(pred, Ys, feats=feats, fusion=fusion, Xs=Xs)
            opt.zero_grad(); loss.backward(); opt.step()
        # validation
        predictor.eval(); all_preds=[]; val_loss,n=0,0
        with torch.no_grad():
            for Xs, Ys in val_loader:
                Xs = {k:v.to(device) for k,v in Xs.items()}
                Ys = Ys.to(device)
                feats = fusion(Xs, return_feats=True)
                pred = predictor(feats)
                val_loss += loss_fn(pred, Ys, feats=feats, fusion=fusion, Xs=Xs).item(); n+=1
                all_preds.append(F.normalize(pred,dim=-1).view(pred.size(0),-1).cpu())
        preds = torch.cat(all_preds,dim=0)
        print(f"Epoch {ep+1}, ValLoss {val_loss/n:.4f}, "
              f"Var(dim) {preds.var(dim=0).mean().item():.6f}, "
              f"Var(samples) {preds.var(dim=1).mean().item():.6f}")

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subj_name = input("Enter subject name: ").strip()
    drive_root = "/content/drive/MyDrive/EEG2Video_data/processed"

    # load data
    de       = np.load(os.path.join(drive_root,"EEG_DE",f"{subj_name}.npy"))
    psd      = np.load(os.path.join(drive_root,"EEG_PSD",f"{subj_name}.npy"))
    windows  = np.load(os.path.join(drive_root,"EEG_windows",f"{subj_name}.npy"))
    segments = np.load(os.path.join(drive_root,"EEG_segments",f"{subj_name}.npy"))
    blip_raw = np.load(os.path.join(drive_root,"BLIP_embeddings","BLIP_embeddings.npy"))

    Xs={"de":[],"psd":[],"windows":[],"segments":[]}; Ys=[]
    for b in range(7):
        for c in range(40):
            for k in range(5):
                Xs["de"].append(de[b,c,k])
                Xs["psd"].append(psd[b,c,k])
                Xs["windows"].append(windows[b,c,k].mean(0))
                Xs["segments"].append(segments[b,c,k])
                Ys.append(blip_raw[b,c,k])
    Xs={k:np.array(v) for k,v in Xs.items()}; Ys=np.array(Ys)

    ds = EEG2BLIPDataset(Xs,Ys,normalise_targets=True)
    n = len(ds); split=int(0.8*n)
    train_loader=DataLoader(torch.utils.data.Subset(ds,range(split)),batch_size=32,shuffle=True)
    val_loader=DataLoader(torch.utils.data.Subset(ds,range(split,n)),batch_size=32)

    fusion = FusionModel().to(device)
    fusion_ckpt = f"/content/drive/MyDrive/EEG2Video_checkpoints/fusion_checkpoints/fusion_checkpoint_{subj_name}.pt"
    fusion.load_state_dict(torch.load(fusion_ckpt,map_location=device))
    for p in fusion.parameters(): p.requires_grad=False

    predictor = SemanticPredictor(input_dim=fusion.total_dim).to(device)

    # choose options
    loss_fn = build_loss(
        use_mse=True,
        use_cosine=True,
        use_var=False,
        use_infonce=False,
        normalise_preds=False,
        use_multitask=True,   # enable multitask
        lambda_cls=1        # weight for classification loss
    )

    train_one_fold(fusion,predictor,train_loader,val_loader,
                   device,loss_fn,epochs=10, use_multitask=True)

if __name__=="__main__":
    main()
