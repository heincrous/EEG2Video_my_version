# ==========================================
# EEG → CLIP Semantic Predictor (DE-Only, MSE Loss)
# ==========================================
# Trains a semantic predictor on DE features for one subject.
# Uses ONLY MSE loss.
# Saves final evaluation metrics and model config in .txt format.
# ==========================================

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from einops import rearrange
from tqdm import tqdm
import importlib


# ==========================================
# Architectural Fine-Tuning Toggle
# ==========================================
# Choose ONE: "layer_width", "dropout", "activation", "normalization"
EXPERIMENT_TYPE = "activation"


# ==========================================
# Config block (tunable but defaults unchanged)
# ==========================================
CONFIG = {
    # === Model structure ===
    "dropout": 0.0,                      # Dropout probability. Default: 0.0
    "layer_widths": [10000, 10000, 10000, 10000],  # Hidden layer sizes. Default: four layers of 10,000 units
    "activation": "ReLU",                # Activation function. Default: "ReLU"
    "normalization": "None",             # Normalization type. Default: "None"

    # === Training and data parameters ===
    "feature_type": "EEG_DE_1per2s",     # EEG feature directory
    "subject_name": "sub1.npy",          # Subject file
    "class_subset": [0, 11, 24, 30, 33], # Subset of classes
    "subset_id": "1",                    # Subset identifier
    "epochs": 200,
    "batch_size": 128,
    "lr": 0.0005,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",

    # === Directory paths ===
    "eeg_root": "/content/drive/MyDrive/EEG2Video_data/processed",
    "clip_path": "/content/drive/MyDrive/EEG2Video_data/processed/CLIP_embeddings/CLIP_embeddings.npy",
    "result_root": "/content/drive/MyDrive/EEG2Video_results/semantic_predictor/architectural_fine-tuning",
}


# ==========================================
# Model
# ==========================================
class CLIPSemanticMLP(nn.Module):
    def __init__(self, input_dim, cfg=CONFIG):
        super().__init__()
        w = cfg["layer_widths"]
        p = cfg["dropout"]
        act_fn = getattr(nn, cfg["activation"])() if hasattr(nn, cfg["activation"]) else nn.ReLU()

        def norm_layer(size):
            if cfg["normalization"] == "BatchNorm":
                return nn.BatchNorm1d(size)
            elif cfg["normalization"] == "LayerNorm":
                return nn.LayerNorm(size)
            elif cfg["normalization"] == "GroupNorm":
                return nn.GroupNorm(4, size)
            else:
                return nn.Identity()

        layers = []
        in_dim = input_dim
        for width in w:
            layers += [nn.Linear(in_dim, width), norm_layer(width), act_fn]
            if p > 0:
                layers.append(nn.Dropout(p))
            in_dim = width
        layers.append(nn.Linear(in_dim, 77 * 768))
        self.mlp = nn.Sequential(*layers)

    def forward(self, eeg):
        return self.mlp(eeg)


class EEGTextDataset:
    def __init__(self, eeg, text):
        self.eeg = eeg
        self.text = text
    def __len__(self):
        return self.eeg.shape[0]
    def __getitem__(self, idx):
        return self.eeg[idx], self.text[idx]


# ==========================================
# Data Loading
# ==========================================
def load_de_data(cfg):
    eeg_path = os.path.join(cfg["eeg_root"], cfg["feature_type"], cfg["subject_name"])
    eeg = np.load(eeg_path, allow_pickle=True)
    clip = np.load(cfg["clip_path"], allow_pickle=True)

    if eeg.ndim == 6 and eeg.shape[3] == 2:
        eeg = eeg.mean(axis=3)
    elif eeg.ndim != 5:
        raise ValueError(f"Unexpected EEG shape: {eeg.shape}")

    print(f"Loaded EEG {cfg['subject_name']} | shape={eeg.shape}")
    print(f"Loaded CLIP shape={clip.shape}")
    return eeg, clip


# ==========================================
# Data Preparation
# ==========================================
def prepare_data(eeg, clip, cfg):
    eeg = eeg[:, cfg["class_subset"]]
    clip = clip[:, cfg["class_subset"]]

    train_eeg, val_eeg, test_eeg = eeg[:5], eeg[5:6], eeg[6:]
    train_clip, val_clip, test_clip = clip[:5], clip[5:6], clip[6:]

    flatten_eeg = lambda x: rearrange(x, "b c s ch t -> (b c s) (ch t)")
    flatten_clip = lambda x: rearrange(x, "b c s tok dim -> (b c s) (tok dim)")

    train_eeg, val_eeg, test_eeg = map(flatten_eeg, [train_eeg, val_eeg, test_eeg])
    train_clip, val_clip, test_clip = map(flatten_clip, [train_clip, val_clip, test_clip])

    scaler = StandardScaler()
    scaler.fit(train_eeg)
    train_eeg = scaler.transform(train_eeg)
    val_eeg = scaler.transform(val_eeg)
    test_eeg = scaler.transform(test_eeg)

    print(f"[Scaler] mean={np.mean(train_eeg):.5f}, std={np.std(train_eeg):.5f}")
    return train_eeg, val_eeg, test_eeg, train_clip, val_clip, test_clip


# ==========================================
# Evaluation
# ==========================================
def evaluate_model(model, eeg_flat, clip_flat, cfg):
    model.eval()
    device = cfg["device"]
    with torch.no_grad():
        eeg_tensor = torch.tensor(eeg_flat, dtype=torch.float32, device=device)
        gt_tensor = torch.tensor(clip_flat, dtype=torch.float32, device=device)
        preds_tensor = model(eeg_tensor)
        mse_loss = F.mse_loss(preds_tensor, gt_tensor).item()
        preds = preds_tensor.cpu().numpy()
    gt = clip_flat

    preds /= np.linalg.norm(preds, axis=1, keepdims=True) + 1e-8
    gt /= np.linalg.norm(gt, axis=1, keepdims=True) + 1e-8

    num_classes = len(cfg["class_subset"])
    samples_per_class = 5
    labels = np.repeat(np.arange(num_classes), samples_per_class)
    avg_cosine = np.mean(np.sum(preds * gt, axis=1))

    class_means = np.array([gt[labels == c].mean(axis=0) for c in range(num_classes)])
    class_means /= np.linalg.norm(class_means, axis=1, keepdims=True) + 1e-8

    sims = np.dot(preds, class_means.T)
    acc = (np.argmax(sims, axis=1) == labels).mean()

    within = [np.dot(preds[i], class_means[labels[i]]) for i in range(len(preds))]
    between = [np.mean(np.dot(class_means[np.arange(num_classes) != labels[i]], preds[i])) for i in range(len(preds))]
    avg_within, avg_between = np.mean(within), np.mean(between)
    fisher_score = np.sum((class_means - class_means.mean(axis=0)) ** 2) / (
        np.sum((preds - class_means[labels]) ** 2) + 1e-8
    )

    return mse_loss, avg_cosine, avg_within, avg_between, fisher_score, acc


# ==========================================
# Training
# ==========================================
def train_model(model, loader, val_eeg, val_clip, cfg):
    device = cfg["device"]
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"] * len(loader))

    for epoch in tqdm(range(1, cfg["epochs"] + 1)):
        model.train()
        total_loss = 0
        for eeg, clip in loader:
            eeg, clip = eeg.float().to(device), clip.float().to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(eeg), clip)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"[Epoch {epoch:03d}/{cfg['epochs']}] Avg Loss: {total_loss / len(loader):.6f}")


# ==========================================
# Save Results (.txt)
# ==========================================
def save_results(cfg, metrics):
    base_dir = os.path.join(cfg["result_root"], EXPERIMENT_TYPE)
    os.makedirs(base_dir, exist_ok=True)

    lw = "-".join(str(x) for x in cfg["layer_widths"])
    filename = (
        f"{EXPERIMENT_TYPE}_de_semantic_lw{lw}_do{cfg['dropout']}_act{cfg['activation']}_"
        f"norm{cfg['normalization']}_lr{cfg['lr']}_bs{cfg['batch_size']}_ep{cfg['epochs']}.txt"
    )
    save_path = os.path.join(base_dir, filename)

    mse, cos, within, between, fisher, acc = metrics
    with open(save_path, "w") as f:
        f.write("EEG → CLIP Semantic Predictor Summary\n")
        f.write("==========================================\n\n")
        f.write(f"Architectural Fine-Tuning Type: {EXPERIMENT_TYPE}\n\n")
        f.write("Model Configuration:\n")
        for k, v in cfg.items():
            f.write(f"{k}: {v}\n")
        f.write("\nFinal Evaluation Metrics:\n")
        f.write(f"MSE Loss: {mse:.6f}\n")
        f.write(f"Avg Cosine(pred, gt): {cos:.4f}\n")
        f.write(f"Within-Class Cosine: {within:.4f}\n")
        f.write(f"Between-Class Cosine: {between:.4f}\n")
        f.write(f"Δ (Within−Between): {within - between:.4f}\n")
        f.write(f"Fisher Score: {fisher:.4f}\n")
        f.write(f"Classification Accuracy: {acc * 100:.2f}%\n")

    print(f"\nSaved configuration and results to: {save_path}")


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    cfg = CONFIG
    eeg, clip = load_de_data(cfg)
    train_eeg, val_eeg, test_eeg, train_clip, val_clip, test_clip = prepare_data(eeg, clip, cfg)

    dataset = EEGTextDataset(train_eeg, train_clip)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

    model = CLIPSemanticMLP(input_dim=train_eeg.shape[1], cfg=cfg).to(cfg["device"])
    train_model(model, loader, val_eeg, val_clip, cfg)

    print("\n=== Final Test Metrics ===")
    metrics = evaluate_model(model, test_eeg, test_clip, cfg)
    save_results(cfg, metrics)
