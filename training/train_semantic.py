# ==========================================
# Semantic Predictor Training (Bundle-Based, Blocks 1–6 train, Block 7 test)
# ==========================================

import os, sys, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Add repo root
repo_root = "/content/EEG2Video_my_version"
sys.path.append(repo_root)

from core_files.models import (
    eegnet, shallownet, deepnet, tsconv, conformer, mlpnet,
    glfnet_mlp, glmnet
)

# ==========================================
# Dataset wrapper
# ==========================================
class EEGTextDataset(Dataset):
    def __init__(self, eeg_array, text_array, scaler=None, fit_scaler=False):
        self.eeg = eeg_array
        self.text = text_array
        flat = self.eeg.reshape(self.eeg.shape[0], -1)
        if fit_scaler:
            self.scaler = StandardScaler().fit(flat)
        else:
            self.scaler = scaler
        flat = self.scaler.transform(flat)
        self.eeg = flat.reshape(self.eeg.shape)
    def __len__(self): return self.eeg.shape[0]
    def __getitem__(self, idx):
        return torch.tensor(self.eeg[idx], dtype=torch.float32), torch.tensor(self.text[idx], dtype=torch.float32)

# ==========================================
# Wrappers
# ==========================================
class WindowEncoderWrapper(nn.Module):
    def __init__(self, base_encoder, out_dim, reduce="mean"):
        super().__init__()
        self.base = base_encoder
        self.reduce = reduce
    def forward(self, x):
        B, W, C, T = x.shape
        x = x.view(B*W, 1, C, T)
        feats = self.base(x)
        feats = feats.view(B, W, -1)
        if self.reduce == "mean":
            return feats.mean(1)
        elif self.reduce == "none":
            return feats.view(B, -1)
        else:
            raise ValueError("reduce must be 'mean' or 'none'")

class ReshapeWrapper(nn.Module):
    def __init__(self, base_model, n_tokens=77):
        super().__init__()
        self.base = base_model
        self.n_tokens = n_tokens
    def forward(self, x):
        out = self.base(x)
        return out.view(out.size(0), self.n_tokens, 768)

# ==========================================
# Loss functions
# ==========================================
def cosine_loss(pred, target):
    pred = F.normalize(pred.view(pred.size(0), -1), dim=-1)
    target = F.normalize(target.view(target.size(0), -1), dim=-1)
    return 1 - (pred * target).sum(dim=-1).mean()
def contrastive_loss(pred, target, temperature=0.07):
    B = pred.size(0)
    pred = F.normalize(pred.view(B, -1), dim=-1)
    target = F.normalize(target.view(B, -1), dim=-1)
    logits = pred @ target.t() / temperature
    labels = torch.arange(B, device=pred.device)
    return F.cross_entropy(logits, labels)
def mse_cosine_loss(pred, target): return F.mse_loss(pred, target) + cosine_loss(pred, target)
def mse_contrastive_loss(pred, target): return F.mse_loss(pred, target) + contrastive_loss(pred, target)
def cosine_contrastive_loss(pred, target): return cosine_loss(pred, target) + contrastive_loss(pred, target)
def mse_cosine_contrastive_loss(pred, target): return F.mse_loss(pred, target) + cosine_loss(pred, target) + contrastive_loss(pred, target)

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    mode = input("\nMode (train / dry): ").strip()

    # Dummy shapes for dry run
    dummy_windows = torch.randn(2, 7, 62, 100).cuda()
    dummy_de = torch.randn(2, 62, 5).cuda()
    dummy_psd = torch.randn(2, 62, 5).cuda()
    dummy_txt = torch.randn(2, 77, 768).cuda()

    output_dim = 77*768
    input_dim_windows = 7*62*100
    input_dim_depsd = 62*5

    if mode == "dry":
        print("\n--- DRY RUN ---")

        # Windows features
        print("\n[Windows features]")
        win_encoders = {
            "eegnet": eegnet(out_dim=output_dim, C=62, T=100),
            "shallownet": shallownet(out_dim=output_dim, C=62, T=100),
            "deepnet": deepnet(out_dim=output_dim, C=62, T=100),
            "tsconv": tsconv(out_dim=output_dim, C=62, T=100),
            "glmnet": glmnet(out_dim=output_dim, emb_dim=256, C=62, T=100),
            "conformer": conformer(out_dim=output_dim),
            "mlp": mlpnet(out_dim=output_dim, input_dim=input_dim_windows)
        }
        for name, base in win_encoders.items():
            for reduce in ["mean", "none"]:
                if name in ["conformer", "mlp"]:
                    model = ReshapeWrapper(base.cuda())
                else:
                    model = ReshapeWrapper(WindowEncoderWrapper(base.cuda(), out_dim=output_dim, reduce=reduce))
                out = model(dummy_windows)
                print(f"{name} ({reduce}): {out.shape}")

        # DE features
        print("\n[DE features]")
        de_encoders = {
            "mlp": mlpnet(out_dim=output_dim, input_dim=input_dim_depsd),
            "glfnet_mlp": glfnet_mlp(out_dim=output_dim, emb_dim=256, input_dim=input_dim_depsd)
        }
        for name, base in de_encoders.items():
            model = ReshapeWrapper(base.cuda())
            out = model(dummy_de)
            print(f"{name}: {out.shape}")

        # PSD features
        print("\n[PSD features]")
        psd_encoders = {
            "mlp": mlpnet(out_dim=output_dim, input_dim=input_dim_depsd),
            "glfnet_mlp": glfnet_mlp(out_dim=output_dim, emb_dim=256, input_dim=input_dim_depsd)
        }
        for name, base in psd_encoders.items():
            model = ReshapeWrapper(base.cuda())
            out = model(dummy_psd)
            print(f"{name}: {out.shape}")

        print("\nDry run complete.")
        sys.exit(0)

    # ---- Training mode ----
    feature_type = input("\nEnter feature type (windows / DE / PSD): ").strip()
    bundle_path = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_EEG_bundle.npz"
    save_root   = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"
    os.makedirs(save_root, exist_ok=True)

    if feature_type == "windows":
        print("\nEncoders: mlp / eegnet / shallownet / deepnet / tsconv / conformer / glmnet")
        encoder_type = input("Enter encoder type: ").strip()
        window_reduce = input("Window reduction (mean / none): ").strip() or "mean"
    elif feature_type in ["DE", "PSD"]:
        print("\nEncoders: mlp / glfnet_mlp")
        encoder_type = input("Enter encoder type: ").strip()
        window_reduce = "mean"
    else:
        raise ValueError("feature_type must be windows / DE / PSD")

    print("\nLoss options: mse / cosine / contrastive / mse+cosine / mse+contrastive / cosine+contrastive / mse+cosine+contrastive")
    loss_type = input("Enter loss type: ").strip()
    num_epochs = int(input("\nEnter number of epochs (default 50): ") or 50)

    # Load bundle
    data = np.load(bundle_path, allow_pickle=True)
    blip_emb = data["BLIP_embeddings"]
    eeg_dict = data["EEG_data"].item()

    eeg_list, txt_list = [], []
    for subj, feats in eeg_dict.items():
        eeg = feats[f"EEG_{feature_type}"]
        eeg_list.append(eeg)
        txt_list.append(blip_emb)
    eeg_all = np.stack(eeg_list)
    txt_all = np.stack(txt_list)

    train_eeg = eeg_all[:, :6].reshape(-1, *eeg_all.shape[3:])
    test_eeg  = eeg_all[:, 6:].reshape(-1, *eeg_all.shape[3:])
    train_txt = txt_all[:, :6].reshape(-1, 77, 768)
    test_txt  = txt_all[:, 6:].reshape(-1, 77, 768)

    train_set = EEGTextDataset(train_eeg, train_txt, fit_scaler=True)
    test_set  = EEGTextDataset(test_eeg, test_txt, scaler=train_set.scaler, fit_scaler=False)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

    output_dim = 77*768
    input_dim  = train_set.eeg.shape[1] * np.prod(train_set.eeg.shape[2:])

    if feature_type == "windows":
        if encoder_type == "glmnet":
            base = glmnet(out_dim=output_dim, emb_dim=256, C=62, T=100).cuda()
            model = WindowEncoderWrapper(base, out_dim=output_dim, reduce=window_reduce).cuda()
        elif encoder_type == "eegnet":
            model = WindowEncoderWrapper(eegnet(out_dim=output_dim, C=62, T=100), out_dim=output_dim, reduce=window_reduce).cuda()
        elif encoder_type == "shallownet":
            model = WindowEncoderWrapper(shallownet(out_dim=output_dim, C=62, T=100), out_dim=output_dim, reduce=window_reduce).cuda()
        elif encoder_type == "deepnet":
            model = WindowEncoderWrapper(deepnet(out_dim=output_dim, C=62, T=100), out_dim=output_dim, reduce=window_reduce).cuda()
        elif encoder_type == "tsconv":
            model = WindowEncoderWrapper(tsconv(out_dim=output_dim, C=62, T=100), out_dim=output_dim, reduce=window_reduce).cuda()
        elif encoder_type == "conformer":
            model = conformer(out_dim=output_dim).cuda()
        elif encoder_type == "mlp":
            model = mlpnet(out_dim=output_dim, input_dim=input_dim).cuda()
    elif feature_type in ["DE","PSD"]:
        if encoder_type == "mlp":
            model = mlpnet(out_dim=output_dim, input_dim=input_dim).cuda()
        elif encoder_type == "glfnet_mlp":
            model = glfnet_mlp(out_dim=output_dim, emb_dim=256, input_dim=input_dim).cuda()

    model = ReshapeWrapper(model, n_tokens=77)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for eeg, text in train_loader:
                eeg, text = eeg.cuda(non_blocking=True), text.cuda(non_blocking=True)
                optimizer.zero_grad()
                pred = model(eeg)
                if loss_type == "mse": loss = F.mse_loss(pred, text)
                elif loss_type == "cosine": loss = cosine_loss(pred, text)
                elif loss_type == "contrastive": loss = contrastive_loss(pred, text)
                elif loss_type == "mse+cosine": loss = mse_cosine_loss(pred, text)
                elif loss_type == "mse+contrastive": loss = mse_contrastive_loss(pred, text)
                elif loss_type == "cosine+contrastive": loss = cosine_contrastive_loss(pred, text)
                elif loss_type == "mse+cosine+contrastive": loss = mse_cosine_contrastive_loss(pred, text)
                else: raise ValueError("Unknown loss type")
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})
                pbar.update(1)
        print(f"[Epoch {epoch+1}] Avg {loss_type} loss: {total_loss/len(train_loader):.6f}")

    tag = f"{feature_type}_{encoder_type}_{loss_type}_{window_reduce}"
    ckpt_path   = os.path.join(save_root, f"semantic_predictor_{tag}.pt")
    scaler_path = os.path.join(save_root, f"scaler_{tag}.pkl")
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    with open(scaler_path, "wb") as f: pickle.dump(train_set.scaler, f)
    print(f"Model saved to {ckpt_path}\nScaler saved to {scaler_path}")
