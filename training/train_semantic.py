# ==========================================
# Semantic Predictor Training (Blocks 1–6 train, Block 7 test)
# ==========================================

import os, sys, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# === Repo imports ===
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
# Reshape Wrapper (force output into [B,77,768])
# ==========================================
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

def mse_cosine_loss(pred, target): 
    return F.mse_loss(pred, target) + cosine_loss(pred, target)

def mse_contrastive_loss(pred, target): 
    return F.mse_loss(pred, target) + contrastive_loss(pred, target)

def cosine_contrastive_loss(pred, target): 
    return cosine_loss(pred, target) + contrastive_loss(pred, target)

def mse_cosine_contrastive_loss(pred, target): 
    return F.mse_loss(pred, target) + cosine_loss(pred, target) + contrastive_loss(pred, target)

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    mode = input("\nMode (train / dry): ").strip()

    # Dummy shapes for dry run
    dummy_segments  = torch.randn(2, 1, 62, 400).cuda()   # (B,1,C,T)
    dummy_de        = torch.randn(2, 62, 5).cuda()        # (B,C,F)
    dummy_psd       = torch.randn(2, 62, 5).cuda()
    dummy_txt       = torch.randn(2, 77, 768).cuda()      # (B,77,768)

    output_dim = 77*768

    if mode == "dry":
        print("\n--- DRY RUN ---")
        # Segments
        for name, base in {
            "eegnet": eegnet(output_dim, 62, 400),
            "shallownet": shallownet(output_dim, 62, 400),
            "deepnet": deepnet(output_dim, 62, 400),
            "tsconv": tsconv(output_dim, 62, 400),
            "glmnet": glmnet(output_dim, 256, 62, 400),
            "mlp": mlpnet(output_dim, 62*400),
        }.items():
            out = ReshapeWrapper(base.cuda())(dummy_segments)
            print(f"Segments-{name}: {out.shape}")

        # DE
        for name, base in {
            "mlp": mlpnet(output_dim, 62*5),
            "glfnet_mlp": glfnet_mlp(output_dim, 256, 62*5),
        }.items():
            out = ReshapeWrapper(base.cuda())(dummy_de)
            print(f"DE-{name}: {out.shape}")

        # PSD
        for name, base in {
            "mlp": mlpnet(output_dim, 62*5),
            "glfnet_mlp": glfnet_mlp(output_dim, 256, 62*5),
        }.items():
            out = ReshapeWrapper(base.cuda())(dummy_psd)
            print(f"PSD-{name}: {out.shape}")

        print("\nDry run complete.")
        sys.exit(0)

    # ---- Training mode ----
    feature_type = input("\nEnter feature type (segments / DE / PSD): ").strip()
    bundle_path  = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_EEG_bundle.npz"
    save_root    = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"
    os.makedirs(save_root, exist_ok=True)

    if feature_type == "segments":
        print("\nEncoders: mlp / eegnet / shallownet / deepnet / tsconv / glmnet")
        encoder_type = input("Enter encoder type: ").strip()
    elif feature_type in ["DE","PSD"]:
        print("\nEncoders: mlp / glfnet_mlp")
        encoder_type = input("Enter encoder type: ").strip()
    else:
        raise ValueError("feature_type must be segments / DE / PSD")

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

    # Explicit reshape logic (assumes [62,400] for segments now)
    if feature_type == "segments":
        train_eeg = eeg_all[:, :6].reshape(-1, 1, 62, 400)
        test_eeg  = eeg_all[:, 6:].reshape(-1, 1, 62, 400)
    elif feature_type in ["DE","PSD"]:
        train_eeg = eeg_all[:, :6].reshape(-1, 62, 5)
        test_eeg  = eeg_all[:, 6:].reshape(-1, 62, 5)

    train_txt = txt_all[:, :6].reshape(-1, 77, 768)
    test_txt  = txt_all[:, 6:].reshape(-1, 77, 768)

    train_set = EEGTextDataset(train_eeg, train_txt, fit_scaler=True)
    test_set  = EEGTextDataset(test_eeg, test_txt, scaler=train_set.scaler, fit_scaler=False)

    # === Diagnostics: check scaling ===
    print("\n[Scaler diagnostics]")
    print(f"Train EEG scaled mean: {train_set.eeg.mean():.4f}, std: {train_set.eeg.std():.4f}")
    print(f"Test EEG scaled mean: {test_set.eeg.mean():.4f}, std: {test_set.eeg.std():.4f}")

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

    input_dim = train_set.eeg.shape[1] * np.prod(train_set.eeg.shape[2:])
    output_dim = 77*768

    # Build model
    if feature_type == "segments":
        if encoder_type == "glmnet": model = glmnet(output_dim, 256, 62, 400).cuda()
        elif encoder_type == "eegnet": model = eegnet(output_dim, 62, 400).cuda()
        elif encoder_type == "shallownet": model = shallownet(output_dim, 62, 400).cuda()
        elif encoder_type == "deepnet": model = deepnet(output_dim, 62, 400).cuda()
        elif encoder_type == "tsconv": model = tsconv(output_dim, 62, 400).cuda()
        elif encoder_type == "mlp": model = mlpnet(output_dim, input_dim).cuda()

    elif feature_type in ["DE","PSD"]:
        if encoder_type == "mlp": model = mlpnet(output_dim, input_dim).cuda()
        elif encoder_type == "glfnet_mlp": model = glfnet_mlp(output_dim, 256, input_dim).cuda()

    model = ReshapeWrapper(model, n_tokens=77)

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))

    # Training loop
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

    # Save
    tag = f"{feature_type}_{encoder_type}_{loss_type}"
    ckpt_path   = os.path.join(save_root, f"semantic_predictor_{tag}.pt")
    scaler_path = os.path.join(save_root, f"scaler_{tag}.pkl")
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    with open(scaler_path, "wb") as f: pickle.dump(train_set.scaler, f)
    print(f"Model saved to {ckpt_path}\nScaler saved to {scaler_path}")
