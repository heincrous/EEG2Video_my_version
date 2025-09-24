# ==========================================
# Semantic Predictor Training (Encoders + Loss Choice)
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

# Import encoders from core_files
from core_files.models import eegnet, shallownet, deepnet, tsconv, conformer, mlpnet


# ==========================================
# Dataset wrapper
# ==========================================
class EEGTextDataset(Dataset):
    def __init__(self, npz_files, feature_type="windows", scaler=None, fit_scaler=False, max_samples=None):
        eeg_all, text_all = [], []
        for f in npz_files:
            data = np.load(f, allow_pickle=True)

            if feature_type == "DE":
                eeg = data["EEG_DE"]   # (N,62,5)
            elif feature_type == "PSD":
                eeg = data["EEG_PSD"]  # (N,62,5)
            elif feature_type == "windows":
                eeg = data["EEG_windows"]  # (N,7,62,100)
            else:
                raise ValueError("feature_type must be one of: DE / PSD / windows")

            text = data["BLIP_embeddings"].reshape(data["BLIP_embeddings"].shape[0], -1)  # (N,59136)

            eeg_all.append(eeg)
            text_all.append(text)

        self.eeg = np.vstack(eeg_all)
        self.text = np.vstack(text_all)

        if max_samples:
            self.eeg = self.eeg[:max_samples]
            self.text = self.text[:max_samples]

        # Fit scaler on flattened EEG
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
# Window wrapper for encoders
# ==========================================
class WindowEncoderWrapper(nn.Module):
    def __init__(self, base_encoder, out_dim):
        super().__init__()
        self.base = base_encoder

    def forward(self, x):  # x: (B,7,62,100)
        B, W, C, T = x.shape
        x = x.view(B*W, 1, C, T)     # feed each window separately
        feats = self.base(x)         # (B*W, out_dim)
        feats = feats.view(B, W, -1)
        return feats.mean(1)         # average across 7 windows


# ==========================================
# Loss functions
# ==========================================
def cosine_loss(pred, target):
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return 1 - (pred * target).sum(dim=-1).mean()


# ==========================================
# Training loop
# ==========================================
if __name__ == "__main__":
    bundle_dir = "/content/drive/MyDrive/EEG2Video_data/processed/SubjectBundles"
    save_root  = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"
    os.makedirs(save_root, exist_ok=True)

    feature_type = input("\nEnter feature type (DE / PSD / windows): ").strip()
    encoder_type = input("\nEnter encoder type (mlp / eegnet / shallownet / deepnet / tsconv / conformer): ").strip()
    loss_type    = input("\nEnter loss type (mse / cosine): ").strip()

    all_bundles = sorted([f for f in os.listdir(bundle_dir) if f.endswith("_train.npz")])
    subjects    = [f.replace("_train.npz", "") for f in all_bundles]

    print("\nSelect subject(s):")
    for idx, subj in enumerate(subjects):
        print(f"  [{idx}] {subj}")

    choice     = input("\nEnter subject indices (comma separated), 'all', or 'check': ").strip()
    num_epochs = int(input("\nEnter number of epochs (default 50): ") or 50)

    # === Dry run ===
    if choice.lower() == "check":
        test_file = os.path.join(bundle_dir, all_bundles[0])
        dataset   = EEGTextDataset([test_file], feature_type=feature_type, fit_scaler=True, max_samples=10)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        eeg, txt = next(iter(dataloader))
        output_dim = txt.shape[1]

        print("\nDry run shapes:")
        print("EEG batch:", eeg.shape, "Text batch:", txt.shape)

        if feature_type in ["DE", "PSD"]:
            input_dim = dataset.eeg.shape[1] * dataset.eeg.shape[2]
            model = mlpnet(out_dim=output_dim, input_dim=input_dim)
        elif feature_type == "windows":
            if encoder_type == "mlp":
                input_dim = dataset.eeg.shape[1] * dataset.eeg.shape[2] * dataset.eeg.shape[3]
                model = mlpnet(out_dim=output_dim, input_dim=input_dim)
            elif encoder_type == "eegnet":
                model = WindowEncoderWrapper(eegnet(out_dim=output_dim, C=62, T=100), out_dim=output_dim)
            elif encoder_type == "shallownet":
                model = WindowEncoderWrapper(shallownet(out_dim=output_dim, C=62, T=100), out_dim=output_dim)
            elif encoder_type == "deepnet":
                model = WindowEncoderWrapper(deepnet(out_dim=output_dim, C=62, T=100), out_dim=output_dim)
            elif encoder_type == "tsconv":
                model = WindowEncoderWrapper(tsconv(out_dim=output_dim, C=62, T=100), out_dim=output_dim)
            elif encoder_type == "conformer":
                model = WindowEncoderWrapper(conformer(out_dim=output_dim), out_dim=output_dim)
            else:
                raise ValueError("Unknown encoder type")
        else:
            raise ValueError("Invalid feature type")

        with torch.no_grad():
            out = model(eeg)
        print("Forward pass OK — model output:", out.shape)
        print("\nDry run successful. Exiting.")
        exit()

    # === Select files ===
    if choice.lower() == "all":
        selected_files = [os.path.join(bundle_dir, f) for f in all_bundles]
        tag = f"all_{feature_type}_{encoder_type}_{loss_type}"
    else:
        selected_idx   = [int(c.strip()) for c in choice.split(",") if c.strip().isdigit()]
        selected_files = [os.path.join(bundle_dir, all_bundles[i]) for i in selected_idx]
        tag = "_".join([subjects[i] for i in selected_idx]) + f"_{feature_type}_{encoder_type}_{loss_type}"

    dataset    = EEGTextDataset(selected_files, feature_type=feature_type, fit_scaler=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

    output_dim = 77*768
    print(f"\nTraining Semantic Predictor ({encoder_type}) on {feature_type} features with {loss_type} loss")
    print(f"Samples: {len(dataset)}")

    # === Model selection ===
    if feature_type in ["DE", "PSD"]:
        input_dim = dataset.eeg.shape[1] * dataset.eeg.shape[2]
        model = mlpnet(out_dim=output_dim, input_dim=input_dim).cuda()
    elif feature_type == "windows":
        if encoder_type == "mlp":
            input_dim = dataset.eeg.shape[1] * dataset.eeg.shape[2] * dataset.eeg.shape[3]
            model = mlpnet(out_dim=output_dim, input_dim=input_dim).cuda()
        elif encoder_type == "eegnet":
            model = WindowEncoderWrapper(eegnet(out_dim=output_dim, C=62, T=100), out_dim=output_dim).cuda()
        elif encoder_type == "shallownet":
            model = WindowEncoderWrapper(shallownet(out_dim=output_dim, C=62, T=100), out_dim=output_dim).cuda()
        elif encoder_type == "deepnet":
            model = WindowEncoderWrapper(deepnet(out_dim=output_dim, C=62, T=100), out_dim=output_dim).cuda()
        elif encoder_type == "tsconv":
            model = WindowEncoderWrapper(tsconv(out_dim=output_dim, C=62, T=100), out_dim=output_dim).cuda()
        elif encoder_type == "conformer":
            model = WindowEncoderWrapper(conformer(out_dim=output_dim), out_dim=output_dim).cuda()
        else:
            raise ValueError("Unknown encoder type")
    else:
        raise ValueError("Invalid feature type")

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
                else:
                    loss = cosine_loss(pred, text)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})
                pbar.update(1)

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Avg {loss_type} loss: {avg_loss:.6f}")

    ckpt_path   = os.path.join(save_root, f"semantic_predictor_{tag}.pt")
    scaler_path = os.path.join(save_root, f"scaler_{tag}.pkl")
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(dataset.scaler, f)

    print(f"Model saved to: {ckpt_path}")
    print(f"Scaler saved to: {scaler_path}")
