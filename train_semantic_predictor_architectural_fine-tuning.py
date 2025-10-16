# ==========================================
# EEG → CLIP Semantic Predictor (DE-Only, MSE Loss)
# ==========================================
# For DE_1per1s (always 6D EEG: [7,40,5,2,62,100])
# Always averages across trials (axis=3)
# Uses only MSE loss and verified metric calculations
# Supports both "architectural" and "optimisation" modes
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


# ==========================================
# Experiment Mode Toggle
# ==========================================
EXPERIMENT_MODE = "architectural"  # or "optimisation"

if EXPERIMENT_MODE == "architectural":
    EXPERIMENT_TYPE = "activation"
    RESULT_ROOT = "/content/drive/MyDrive/EEG2Video_results/semantic_predictor/architectural_fine-tuning"
else:
    EXPERIMENT_TYPE = "scheduler"
    RESULT_ROOT = "/content/drive/MyDrive/EEG2Video_results/semantic_predictor/optimisation"


# ==========================================
# Config block
# ==========================================
CONFIG = {
    "dropout": 0.0,
    "layer_widths": [10000, 10000, 10000, 10000],
    "activation": "ReLU",
    "normalization": "None",

    "feature_type": "EEG_DE_1per1s",
    "subject_name": "sub1.npy",
    "class_subset": [0, 11, 24, 30, 33],
    "subset_id": "1",
    "epochs": 200,
    "batch_size": 128,
    "lr": 0.0005,
    "optimizer": "adam",
    "scheduler": "cosine",
    "weight_decay": 0.0,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",

    "eeg_root": "/content/drive/MyDrive/EEG2Video_data/processed",
    "clip_path": "/content/drive/MyDrive/EEG2Video_data/processed/CLIP_embeddings/CLIP_embeddings.npy",
    "result_root": RESULT_ROOT,
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
            if cfg["normalization"] == "LayerNorm":
                return nn.LayerNorm(size)
            if cfg["normalization"] == "GroupNorm":
                return nn.GroupNorm(4, size)
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
# Data Loading (always 6D averaging)
# ==========================================
def load_de_data(cfg):
    eeg_path = os.path.join(cfg["eeg_root"], cfg["feature_type"], cfg["subject_name"])
    clip_path = cfg["clip_path"]

    eeg = np.load(eeg_path, allow_pickle=True)
    clip = np.load(clip_path, allow_pickle=True)

    if eeg.ndim != 6:
        raise ValueError(f"Expected 6D EEG array, got shape {eeg.shape}")
    print("Averaging across trials (axis=3).")
    eeg = eeg.mean(axis=3)  # → shape [7, 40, 5, 62, 100]

    print(f"Loaded EEG {cfg['subject_name']} shape: {eeg.shape}")
    print(f"Loaded CLIP shape: {clip.shape}")
    return eeg, clip


# ==========================================
# Data Preparation
# ==========================================
def prepare_data(eeg, clip, cfg):
    eeg = eeg[:, cfg["class_subset"]]      # select classes
    clip = clip[:, cfg["class_subset"]]    # match classes

    train_eeg, val_eeg, test_eeg = eeg[:5], eeg[5:6], eeg[6:]
    train_clip, val_clip, test_clip = clip[:5], clip[5:6], clip[6:]

    flatten_eeg = lambda x: rearrange(x, "b c s ch t -> (b c s) (ch t)")
    flatten_clip = lambda x: rearrange(x, "b c s tok dim -> (b c s) (tok dim)")

    train_eeg, val_eeg, test_eeg = map(flatten_eeg, [train_eeg, val_eeg, test_eeg])
    train_clip, val_clip, test_clip = map(flatten_clip, [train_clip, val_clip, test_clip])

    scaler = StandardScaler()
    scaler.fit(eeg.reshape(-1, eeg.shape[-2] * eeg.shape[-1]))
    train_eeg = scaler.transform(train_eeg)
    val_eeg = scaler.transform(val_eeg)
    test_eeg = scaler.transform(test_eeg)

    print(f"[Scaler] mean={np.mean(train_eeg):.5f}, std={np.std(train_eeg):.5f}")
    return train_eeg, val_eeg, test_eeg, train_clip, val_clip, test_clip


# ==========================================
# Optimiser / Scheduler
# ==========================================
def build_optimizer(model, cfg):
    if cfg["optimizer"].lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=0.0)
    if cfg["optimizer"].lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    raise ValueError("Unsupported optimizer type.")


def build_scheduler(optimizer, cfg):
    if cfg["scheduler"].lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    if cfg["scheduler"].lower() == "constant":
        return None
    raise ValueError("Unsupported scheduler type.")


# ==========================================
# Evaluation (verified metric logic)
# ==========================================
def evaluate_model(model, eeg_flat, clip_flat, cfg):
    device = cfg["device"]
    model.eval()
    with torch.no_grad():
        eeg_tensor = torch.tensor(eeg_flat, dtype=torch.float32, device=device)
        gt_tensor = torch.tensor(clip_flat, dtype=torch.float32, device=device)
        preds_tensor = model(eeg_tensor)
        mse_loss = F.mse_loss(preds_tensor, gt_tensor).item()
        preds = preds_tensor.cpu().numpy()

    gt = clip_flat
    preds_norm = preds / (np.linalg.norm(preds, axis=1, keepdims=True) + 1e-8)
    gt_norm = gt / (np.linalg.norm(gt, axis=1, keepdims=True) + 1e-8)

    num_classes = len(cfg["class_subset"])
    samples_per_class = 5
    labels = np.repeat(np.arange(num_classes), samples_per_class)

    avg_cosine = float(np.mean(np.sum(preds_norm * gt_norm, axis=1)))

    # Compute class means
    class_means = np.zeros((num_classes, preds_norm.shape[1]))
    for c in range(num_classes):
        class_means[c] = gt_norm[labels == c].mean(axis=0)
        class_means[c] /= np.linalg.norm(class_means[c]) + 1e-8

    sims = np.dot(preds_norm, class_means.T)
    acc = float((np.argmax(sims, axis=1) == labels).mean())

    within = [np.dot(preds_norm[i], class_means[labels[i]]) for i in range(len(preds_norm))]
    between = [np.mean(np.dot(class_means[np.arange(num_classes) != labels[i]], preds_norm[i])) for i in range(len(preds_norm))]
    avg_within, avg_between = float(np.mean(within)), float(np.mean(between))

    # Fisher score = between-class variance / within-class variance
    global_mean = np.mean(class_means, axis=0)
    sb = np.sum((class_means - global_mean) ** 2)
    sw = np.sum([(preds_norm[labels == c] - class_means[c]) ** 2 for c in range(num_classes)])
    fisher_score = float(sb / (sw + 1e-8))

    print(
        f"  MSE Loss: {mse_loss:.6f}\n"
        f"  Avg cosine(pred,gt): {avg_cosine:.4f}\n"
        f"  Within-class cosine: {avg_within:.4f}\n"
        f"  Between-class cosine: {avg_between:.4f}\n"
        f"  Fisher Score: {fisher_score:.4f}\n"
        f"  Δ (Within−Between): {avg_within - avg_between:.4f}\n"
        f"  Classification Accuracy: {acc*100:.2f}%"
    )

    return mse_loss, avg_cosine, avg_within, avg_between, fisher_score, acc


# ==========================================
# Training
# ==========================================
def train_model(model, train_loader, val_eeg, val_clip, cfg):
    device = cfg["device"]
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    for epoch in tqdm(range(1, cfg["epochs"] + 1)):
        model.train()
        total_loss = 0
        for eeg, clip in train_loader:
            eeg, clip = eeg.float().to(device), clip.float().to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(eeg), clip)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"\n[Epoch {epoch:03d}/{cfg['epochs']}] Avg Loss: {avg_loss:.6f}")
            evaluate_model(model, val_eeg, val_clip, cfg)


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

    print("\nFinal Test Metrics:")
    mse, cos, within, between, fisher, acc = evaluate_model(model, test_eeg, test_clip, cfg)

    exp_dir = os.path.join(cfg["result_root"], EXPERIMENT_TYPE)
    os.makedirs(exp_dir, exist_ok=True)
    filename = (
        f"{EXPERIMENT_TYPE}_opt{cfg['optimizer']}_sched{cfg['scheduler']}_"
        f"lr{cfg['lr']}_wd{cfg['weight_decay']}_bs{cfg['batch_size']}_ep{cfg['epochs']}.txt"
    )
    save_path = os.path.join(exp_dir, filename)

    with open(save_path, "w") as f:
        f.write(f"EEG → CLIP Semantic Predictor ({'Architectural Fine-Tuning' if EXPERIMENT_MODE == 'architectural' else 'Optimisation'})\n")
        f.write("==========================================\n\n")
        f.write(f"Experiment Type: {EXPERIMENT_TYPE}\n\n")
        f.write("Configuration Used:\n")
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
