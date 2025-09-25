# ==========================================
# Semantic Predictor Training (77x768 Embeddings, Multi-Loss Options)
# ==========================================

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Add repo root so we can import core_files
repo_root = "/content/EEG2Video_my_version"
sys.path.append(repo_root)

from core_files.models import (
    eegnet, shallownet, deepnet, tsconv, conformer, mlpnet,
    glfnet, glfnet_mlp, glmnet
)

# ==========================================
# Dataset wrapper
# ==========================================
class EEGTextDataset(Dataset):
    def __init__(self, npz_files, feature_type="windows", scaler=None, fit_scaler=False, max_samples=None):
        eeg_all, text_all = [], []
        for f in npz_files:
            data = np.load(f, allow_pickle=True)

            if feature_type == "DE":
                eeg = data["EEG_DE"]
            elif feature_type == "PSD":
                eeg = data["EEG_PSD"]
            elif feature_type == "windows":
                eeg = data["EEG_windows"]
            else:
                raise ValueError("feature_type must be one of: DE / PSD / windows")

            text = data["BLIP_embeddings"]
            eeg_all.append(eeg)
            text_all.append(text)

        self.eeg = np.vstack(eeg_all)
        self.text = np.vstack(text_all)

        if max_samples:
            self.eeg = self.eeg[:max_samples]
            self.text = self.text[:max_samples]

        flat = self.eeg.reshape(self.eeg.shape[0], -1)
        if fit_scaler:
            self.scaler = StandardScaler().fit(flat)
        else:
            self.scaler = scaler
        flat = self.scaler.transform(flat)
        self.eeg = flat.reshape(self.eeg.shape)

    def __len__(self):
        return self.eeg.shape[0]

    def __getitem__(self, idx):
        eeg = self.eeg[idx]
        txt = self.text[idx]
        return torch.tensor(eeg, dtype=torch.float32), torch.tensor(txt, dtype=torch.float32)

# ==========================================
# Wrappers
# ==========================================
class WindowEncoderWrapper(nn.Module):
    def __init__(self, base_encoder, out_dim):
        super().__init__()
        self.base = base_encoder
    def forward(self, x):
        B, W, C, T = x.shape
        x = x.view(B*W, 1, C, T)
        feats = self.base(x)
        feats = feats.view(B, W, -1)
        return feats.mean(1)

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
# Training loop
# ==========================================
if __name__ == "__main__":
    bundle_dir = "/content/drive/MyDrive/EEG2Video_data/processed/SubjectBundles"
    save_root  = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"
    os.makedirs(save_root, exist_ok=True)

    feature_type = input("\nEnter feature type (DE / PSD / windows): ").strip()

    if feature_type == "windows":
        print("\nEncoders available for windows: mlp / eegnet / shallownet / deepnet / tsconv / conformer / glmnet")
        encoder_type = input("Enter encoder type: ").strip()
    elif feature_type in ["DE", "PSD"]:
        print("\nEncoders available for DE/PSD: mlp / glfnet / glfnet_mlp")
        encoder_type = input("Enter encoder type: ").strip()
    else:
        raise ValueError("feature_type must be one of: DE / PSD / windows")

    print("\nLoss options: mse / cosine / contrastive / mse+cosine / mse+contrastive / cosine+contrastive / mse+cosine+contrastive")
    loss_type = input("Enter loss type: ").strip()

    all_bundles = sorted([f for f in os.listdir(bundle_dir) if f.endswith("_train.npz")])
    subjects    = [f.replace("_train.npz", "") for f in all_bundles]

    print("\nSelect subject(s):")
    for idx, subj in enumerate(subjects):
        print(f"  [{idx}] {subj}")

    choice     = input("\nEnter subject indices (comma separated), 'all', or 'check': ").strip()
    num_epochs = int(input("\nEnter number of epochs (default 50): ") or 50)

    # Dry run
    if choice.lower() == "check":
        test_file = os.path.join(bundle_dir, all_bundles[0])
        dataset   = EEGTextDataset([test_file], feature_type=feature_type, fit_scaler=True, max_samples=2)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        eeg, txt = next(iter(dataloader))
        output_dim = 77*768

        if feature_type == "windows":
            if encoder_type == "glmnet":
                base = glmnet(out_dim=output_dim, emb_dim=256, C=62, T=100)
                model = WindowEncoderWrapper(base, out_dim=output_dim)
            elif encoder_type == "eegnet":
                model = WindowEncoderWrapper(eegnet(out_dim=output_dim, C=62, T=100), out_dim=output_dim)
            elif encoder_type == "shallownet":
                model = WindowEncoderWrapper(shallownet(out_dim=output_dim, C=62, T=100), out_dim=output_dim)
            elif encoder_type == "deepnet":
                model = WindowEncoderWrapper(deepnet(out_dim=output_dim, C=62, T=100), out_dim=output_dim)
            elif encoder_type == "tsconv":
                model = WindowEncoderWrapper(tsconv(out_dim=output_dim, C=62, T=100), out_dim=output_dim)
            elif encoder_type == "conformer":
                model = conformer(out_dim=output_dim)
            elif encoder_type == "mlp":
                model = mlpnet(out_dim=output_dim, input_dim=eeg[0].numel())
        elif feature_type in ["DE", "PSD"]:
            if encoder_type == "mlp":
                model = mlpnet(out_dim=output_dim, input_dim=eeg[0].numel())
            elif encoder_type == "glfnet":
                model = glfnet(out_dim=output_dim, emb_dim=256, C=62, T=5)
            elif encoder_type == "glfnet_mlp":
                model = glfnet_mlp(out_dim=output_dim, emb_dim=256, input_dim=eeg[0].numel())
        else:
            raise ValueError("Unknown feature/encoder combination")

        model = ReshapeWrapper(model, n_tokens=77)
        with torch.no_grad():
            out = model(eeg)
        print("\nDry run successful")
        print("EEG batch:", eeg.shape, "Text batch:", txt.shape, "Model output:", out.shape)
        sys.exit()

    # Select files
    if choice.lower() == "all":
        selected_files = [os.path.join(bundle_dir, f) for f in all_bundles]
        tag = f"all_{feature_type}_{encoder_type}_{loss_type}"
    else:
        selected_idx   = [int(c.strip()) for c in choice.split(",") if c.strip().isdigit()]
        selected_files = [os.path.join(bundle_dir, all_bundles[i]) for i in selected_idx]
        tag = "_".join([subjects[i] for i in selected_idx]) + f"_{feature_type}_{encoder_type}_{loss_type}"

    dataset    = EEGTextDataset(selected_files, feature_type=feature_type, fit_scaler=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

    output_dim = 77*768
    input_dim  = dataset.eeg.shape[1] * np.prod(dataset.eeg.shape[2:])

    if feature_type == "windows":
        if encoder_type == "glmnet":
            base = glmnet(out_dim=output_dim, emb_dim=256, C=62, T=100).cuda()
            model = WindowEncoderWrapper(base, out_dim=output_dim).cuda()
        elif encoder_type == "eegnet":
            model = WindowEncoderWrapper(eegnet(out_dim=output_dim, C=62, T=100), out_dim=output_dim).cuda()
        elif encoder_type == "shallownet":
            model = WindowEncoderWrapper(shallownet(out_dim=output_dim, C=62, T=100), out_dim=output_dim).cuda()
        elif encoder_type == "deepnet":
            model = WindowEncoderWrapper(deepnet(out_dim=output_dim, C=62, T=100), out_dim=output_dim).cuda()
        elif encoder_type == "tsconv":
            model = WindowEncoderWrapper(tsconv(out_dim=output_dim, C=62, T=100), out_dim=output_dim).cuda()
        elif encoder_type == "conformer":
            model = conformer(out_dim=output_dim).cuda()
        elif encoder_type == "mlp":
            model = mlpnet(out_dim=output_dim, input_dim=input_dim).cuda()
    elif feature_type in ["DE", "PSD"]:
        if encoder_type == "mlp":
            model = mlpnet(out_dim=output_dim, input_dim=input_dim).cuda()
        elif encoder_type == "glfnet":
            model = glfnet(out_dim=output_dim, emb_dim=256, C=62, T=5).cuda()
        elif encoder_type == "glfnet_mlp":
            model = glfnet_mlp(out_dim=output_dim, emb_dim=256, input_dim=input_dim).cuda()

    model = ReshapeWrapper(model, n_tokens=77)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(dataloader))

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for eeg, text in dataloader:
                eeg, text = eeg.cuda(non_blocking=True), text.cuda(non_blocking=True)
                optimizer.zero_grad()
                pred = model(eeg)

                if loss_type == "mse":
                    loss = F.mse_loss(pred, text)
                elif loss_type == "cosine":
                    loss = cosine_loss(pred, text)
                elif loss_type == "contrastive":
                    loss = contrastive_loss(pred, text)
                elif loss_type == "mse+cosine":
                    loss = mse_cosine_loss(pred, text)
                elif loss_type == "mse+contrastive":
                    loss = mse_contrastive_loss(pred, text)
                elif loss_type == "cosine+contrastive":
                    loss = cosine_contrastive_loss(pred, text)
                elif loss_type == "mse+cosine+contrastive":
                    loss = mse_cosine_contrastive_loss(pred, text)
                else:
                    raise ValueError("Unknown loss type")

                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})
                pbar.update(1)
        print(f"[Epoch {epoch+1}] Avg {loss_type} loss: {total_loss/len(dataloader):.6f}")

    ckpt_path   = os.path.join(save_root, f"semantic_predictor_{tag}.pt")
    scaler_path = os.path.join(save_root, f"scaler_{tag}.pkl")
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(dataset.scaler, f)
    print(f"Model saved to {ckpt_path}")
    print(f"Scaler saved to {scaler_path}")
