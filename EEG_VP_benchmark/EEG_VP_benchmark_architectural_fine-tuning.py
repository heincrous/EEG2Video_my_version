# ==========================================
# EEG CLASSIFICATION (EEG → CATEGORY LABELS)
# ==========================================
# Input: Preprocessed EEG DE features
# Process: Train MLP encoder (GLFNet-MLP) to classify 40 visual categories
# Output: Accuracy metrics (.txt)
# ==========================================

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import importlib

import torch
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm


# ==========================================
# Default Configuration
# ==========================================
"""
DEFAULT MODEL CONFIGURATION

Architecture:
Encoder: GLFNet-MLP
Embedding dimension: 64
Input dimension: 310 (62 × 5)
Layer width: [512, 256]
Dropout: 0.0
Activation: ELU
Normalisation: BatchNorm

Optimisation:
Learning rate: 0.0005
Optimiser: Adam
Weight decay: 0.0
Scheduler: constant

Training:
Epochs: 100
Batch size: 128
Loss: CrossEntropy
"""

# ==========================================
# Experiment Settings
# ==========================================
EXPERIMENT_MODE = "architectural"  # Classification only
EXPERIMENT_TYPE = "normalisation"     # Layer_width, dropout, activation, or normalisation

RESULT_ROOT = "/content/drive/MyDrive/EEG2Video_results/EEG_VP_benchmark/architectural_fine-tuning"


# ==========================================
# Configuration Table
# ==========================================
CONFIG = {
    "feature_type": "EEG_DE_1per1s",
    "encoder_name": "glfnet_mlp",
    "subjects_to_train": [
        "sub1_session2.npy", "sub1.npy", "sub18.npy", "sub19.npy",
        "sub6.npy", "sub15.npy", "sub20.npy", "sub7.npy", "sub10.npy", "sub13.npy"
    ],
    "num_classes": 40,
    "channels": 62,
    "time_len": 5,
    "num_blocks": 7,
    "clips_per_class": 5,
    "emb_dim": 64,
    "batch_size": 128,
    "num_epochs": 100,
    "lr": 0.0005,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data_root": "/content/drive/MyDrive/EEG2Video_data/processed/",
    "de_dir": "EEG_DE_1per1s/",
    "result_root": RESULT_ROOT,
}


# ==========================================
# Model Definition
# ==========================================
class EEGClassifier(nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        from models.glfnet_mlp import glfnet_mlp
        self.model = glfnet_mlp(out_dim=cfg["num_classes"], emb_dim=cfg["emb_dim"], input_dim=input_dim)

    def forward(self, x):
        return self.model(x)


# ==========================================
# Data Handling
# ==========================================
def create_labels(num_classes, clips_per_class, blocks=1):
    block_labels = np.repeat(np.arange(num_classes), clips_per_class)
    return np.tile(block_labels, blocks)


def load_feature_data(sub_file, cfg):
    path = os.path.join(cfg["data_root"], cfg["de_dir"], sub_file)
    data = np.load(path)
    print(f"Loaded {sub_file} | shape {data.shape}")
    return data


def preprocess_data(data, cfg):
    """Preprocess DE EEG features (7,40,5,2,62,5) into train/val/test."""
    data = data.mean(axis=3)  # average trials -> (7,40,5,62,5)
    b, c, d, f, g = data.shape
    data = data.reshape(b, c * d, f, g)

    # Normalisation
    scaler = StandardScaler()
    flat_all = data.reshape(-1, g)
    scaler.fit(flat_all)
    scaled = scaler.transform(flat_all).reshape(b, c * d, f, g)

    # Split by blocks
    train_data, val_data, test_data = scaled[:5], scaled[5:6], scaled[6:7]

    labels = {
        "train": create_labels(cfg["num_classes"], cfg["clips_per_class"], 5),
        "val":   create_labels(cfg["num_classes"], cfg["clips_per_class"]),
        "test":  create_labels(cfg["num_classes"], cfg["clips_per_class"]),
    }

    processed = {
        "train": train_data.reshape(-1, f, g),
        "val":   val_data.reshape(-1, f, g),
        "test":  test_data.reshape(-1, f, g),
    }
    return processed, labels


def Get_Dataloader(data_split, labels, istrain, batch_size):
    X = torch.tensor(data_split, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return data.DataLoader(data.TensorDataset(X, y), batch_size=batch_size, shuffle=istrain)


# ==========================================
# Evaluation Utilities
# ==========================================
def topk_accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size).item())
        return res


def train_and_eval(model, train_iter, test_iter, cfg):
    device = cfg["device"]
    model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()

    for epoch in tqdm(range(1, cfg["num_epochs"] + 1)):
        model.train()
        total_loss, total_acc, total_count = 0, 0, 0

        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item() * X.shape[0]
            total_acc += (y_hat.argmax(1) == y).sum().item()
            total_count += X.shape[0]

        if epoch % 10 == 0:
            avg_loss = total_loss / total_count
            avg_acc = total_acc / total_count
            print(f"[Epoch {epoch}/{cfg['num_epochs']}] Loss={avg_loss:.4f} | Acc={avg_acc:.4f}")

    # Evaluation
    model.eval()
    top1_all, top5_all = [], []
    with torch.no_grad():
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            top1, top5 = topk_accuracy(y_hat, y)
            top1_all.append(top1)
            top5_all.append(top5)

    return np.mean(top1_all), np.mean(top5_all)


# ==========================================
# Save Results
# ==========================================
def save_results(cfg, top1_list, top5_list, exp_type):
    mean_top1, std_top1 = np.mean(top1_list), np.std(top1_list)
    mean_top5, std_top5 = np.mean(top5_list), np.std(top5_list)
    exp_dir = os.path.join(cfg["result_root"], exp_type)
    os.makedirs(exp_dir, exist_ok=True)

    # Extract architecture details from the model definition if available
    try:
        model_module = importlib.import_module(f"models.{cfg['encoder_name']}")
        model_cfg = getattr(model_module, "CONFIG", {})
    except Exception:
        model_cfg = {}

    layer_widths = model_cfg.get("layer_widths", model_cfg.get("layer_width", "NA"))
    if isinstance(layer_widths, (list, tuple)):
        layer_widths = "-".join(str(x) for x in layer_widths)

    dropout = model_cfg.get("dropout", "NA")
    activation = model_cfg.get("activation", "NA")
    norm = model_cfg.get("normalisation", model_cfg.get("norm", "NA"))

    # Create filename based on architecture configuration
    fname = f"{exp_type}_glfnet_mlp_lw{layer_widths}_do{dropout}_act{activation}_norm{norm}.txt"
    save_path = os.path.join(exp_dir, fname)

    with open(save_path, "w") as f:
        f.write("EEG→Category Classifier (Architectural Fine-Tuning)\n")
        f.write("=" * 60 + "\n\n")
        f.write("Configuration Used:\n")
        for k, v in cfg.items():
            f.write(f"{k}: {v}\n")

        f.write("\nModel Architecture Parameters:\n")
        f.write(f"Layer Widths: {layer_widths}\n")
        f.write(f"Dropout: {dropout}\n")
        f.write(f"Activation: {activation}\n")
        f.write(f"Normalisation: {norm}\n")

        f.write("\nFinal Evaluation Metrics:\n")
        f.write(f"Mean Top-1: {mean_top1:.4f} ± {std_top1:.4f}\n")
        f.write(f"Mean Top-5: {mean_top5:.4f} ± {std_top5:.4f}\n")

        f.write("\nSubject-Wise Results:\n")
        for s, t1, t5 in zip(cfg["subjects_to_train"], top1_list, top5_list):
            f.write(f"{s:15s} | Top-1: {t1:.4f} | Top-5: {t5:.4f}\n")

    print(f"[Saved] {save_path}")


# ==========================================
# Cleanup Utilities
# ==========================================
def clean_old_results(cfg, exp_type):
    """Remove only result files with the same architecture configuration."""
    exp_dir = os.path.join(cfg["result_root"], exp_type)
    if not os.path.exists(exp_dir):
        return

    try:
        model_module = importlib.import_module(f"models.{cfg['encoder_name']}")
        model_cfg = getattr(model_module, "CONFIG", {})
    except Exception:
        model_cfg = {}

    # Extract key identifiers
    layer_widths = model_cfg.get("layer_widths", model_cfg.get("layer_width", "NA"))
    if isinstance(layer_widths, (list, tuple)):
        layer_widths = "-".join(str(x) for x in layer_widths)
    dropout = model_cfg.get("dropout", "NA")
    activation = model_cfg.get("activation", "NA")
    norm = model_cfg.get("normalisation", model_cfg.get("norm", "NA"))

    # Unique identifier substring for matching same setup files
    signature = f"lw{layer_widths}_do{dropout}_act{activation}_norm{norm}"
    deleted = []

    # Remove only files matching the same architecture configuration
    for f in os.listdir(exp_dir):
        if f.endswith(".txt") and signature in f:
            os.remove(os.path.join(exp_dir, f))
            deleted.append(f)

    if deleted:
        print(f"[Cleanup] Removed old matching result files: {deleted}")
    else:
        print("[Cleanup] No matching result files found to delete.")


# ==========================================
# Main
# ==========================================
def main():
    cfg = CONFIG
    clean_old_results(cfg, EXPERIMENT_TYPE)
    subjects = cfg["subjects_to_train"]

    all_top1, all_top5 = [], []
    for sub in subjects:
        print(f"\n=== Subject: {sub} ===")
        data = load_feature_data(sub, cfg)
        processed, labels = preprocess_data(data, cfg)
        train_iter = Get_Dataloader(processed["train"], labels["train"], True, cfg["batch_size"])
        test_iter  = Get_Dataloader(processed["test"],  labels["test"],  False, cfg["batch_size"])
        model = EEGClassifier(input_dim=310, cfg=cfg).to(cfg["device"])
        t1, t5 = train_and_eval(model, train_iter, test_iter, cfg)
        print(f"[Test Results] Top-1={t1:.4f} | Top-5={t5:.4f}")
        all_top1.append(t1)
        all_top5.append(t5)


    print(f"\n=== Final Results ===")
    print(f"Mean Top-1: {np.mean(all_top1):.4f} ± {np.std(all_top1):.4f}")
    print(f"Mean Top-5: {np.mean(all_top5):.4f} ± {np.std(all_top5):.4f}")

    save_results(cfg, all_top1, all_top5, EXPERIMENT_TYPE)


if __name__ == "__main__":
    main()
