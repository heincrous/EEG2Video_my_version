# ==========================================
# DYNAMIC PREDICTOR (EEG → Motion Classification)
# ==========================================
# Input: Preprocessed EEG DE features and optical flow labels
# Process: Train GLFNet-MLP to classify Fast vs Slow motion
# Output: Accuracy metrics (.txt)
# ==========================================

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from models.glfnet_mlp import glfnet_mlp


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
EXPERIMENT_MODE = "optimisation"
EXPERIMENT_TYPE = "learning_rate"

RESULT_ROOT = "/content/drive/MyDrive/EEG2Video_results/dynamic_predictor/optimisation"

CONFIG = {
    "feature_type": "EEG_DE_1per1s",
    "subjects_to_train": [],
    "epochs": 100,
    "batch_size": 128,
    "lr": 0.0001,
    "optimiser": "adamw",
    "weight_decay": 0.25,
    "scheduler": "cosine",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "emb_dim": 64,
    "input_dim": 62 * 5,
    "num_classes": 2,
    "channels": 62,
    "time_len": 5,
    "result_root": RESULT_ROOT,
    "data_root": "/content/drive/MyDrive/EEG2Video_data/processed",
    "optical_flow_path": "/content/drive/MyDrive/EEG2Video_data/processed/meta-info/All_video_optical_flow_score_byclass.npy",
}


# ==========================================
# Dataset
# ==========================================
class EEGBinaryDataset(Dataset):
    def __init__(self, eeg, labels):
        self.eeg = eeg
        self.labels = labels

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        return self.eeg[idx], self.labels[idx]


# ==========================================
# Data Handling
# ==========================================
def create_binary_labels(flow_path):
    flow = np.load(flow_path).reshape(-1)
    threshold = np.median(flow)
    labels = (flow > threshold).astype(int)
    return labels


def load_de_data(cfg, sub_file):
    eeg_path = os.path.join(cfg["data_root"], cfg["feature_type"], sub_file)
    eeg = np.load(eeg_path, allow_pickle=True)

    if eeg.ndim != 6:
        raise ValueError(f"Expected 6D EEG array [7,40,5,2,62,5], got {eeg.shape}")

    eeg = eeg.mean(axis=3)  # average trials -> [7,40,5,62,5]
    print(f"Loaded {sub_file} shape: {eeg.shape}")
    return eeg


def preprocess_data(eeg):
    b, c, s, ch, f = eeg.shape
    eeg = eeg.reshape(b, c * s, ch, f)

    scaler = StandardScaler()
    flat = eeg.reshape(-1, f)
    scaler.fit(flat)
    eeg = scaler.transform(flat).reshape(b, c * s, ch, f)
    return eeg


def split_data(eeg):
    train, val, test = eeg[:5], eeg[5:6], eeg[6:]
    return (
        train.reshape(-1, eeg.shape[2], eeg.shape[3]),
        val.reshape(-1, eeg.shape[2], eeg.shape[3]),
        test.reshape(-1, eeg.shape[2], eeg.shape[3]),
    )


# ==========================================
# Optimiser and Scheduler
# ==========================================
def build_optimiser(model, cfg):
    opt = cfg["optimiser"].lower()
    if opt == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if opt == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    raise ValueError("Unsupported optimiser type.")


def build_scheduler(optimiser, cfg):
    sched = cfg["scheduler"].lower()
    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg["epochs"])
    if sched == "constant":
        return None
    raise ValueError("Unsupported scheduler type.")


# ==========================================
# Evaluation
# ==========================================
def evaluate_model(model, loader, cfg):
    device = cfg["device"]
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"[Evaluation] Accuracy: {acc:.4f}")
    return acc


# ==========================================
# Training Loop
# ==========================================
def train_model(model, train_loader, test_loader, cfg):
    device = cfg["device"]
    optimiser = build_optimiser(model, cfg)
    scheduler = build_scheduler(optimiser, cfg)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in tqdm(range(1, cfg["epochs"] + 1)):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimiser.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimiser.step()
            if scheduler:
                scheduler.step()

            epoch_loss += loss.item() * X.size(0)
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)

        if epoch % 10 == 0:
            avg_loss = epoch_loss / total
            avg_acc = correct / total
            print(f"\n[Epoch {epoch}/{cfg['epochs']}] AvgLoss={avg_loss:.6f} | Train Acc={avg_acc:.4f}")

    test_acc = evaluate_model(model, test_loader, cfg)
    return test_acc


# ==========================================
# Save Results
# ==========================================
def save_results(cfg, acc_list, subjects, exp_type):
    mean_acc = np.mean(acc_list)
    exp_dir = os.path.join(cfg["result_root"], exp_type)
    os.makedirs(exp_dir, exist_ok=True)

    fname = (
        f"{exp_type}_opt{cfg['optimiser']}_wd{cfg['weight_decay']}"
        f"_sched{cfg['scheduler']}_lr{cfg['lr']}.txt"
    )
    save_path = os.path.join(exp_dir, fname)

    with open(save_path, "w") as f:
        f.write(f"EEG→Dynamic Predictor ({EXPERIMENT_MODE.capitalize()})\n")
        f.write("=" * 60 + "\n\n")
        f.write("Configuration Used:\n")
        for k, v in cfg.items():
            f.write(f"{k}: {v}\n")

        f.write("\nFinal Evaluation Metrics:\n")
        f.write(f"Mean Accuracy: {mean_acc:.4f}\n\n")
        f.write("Subject-Wise Accuracy:\n")
        for sub, acc in zip(subjects, acc_list):
            f.write(f"{sub:15s} | Accuracy: {acc:.4f}\n")

    print(f"Saved results to: {save_path}")


# ==========================================
# Cleanup Module
# ==========================================
def clean_old_result_files(cfg, exp_type):
    exp_dir = os.path.join(cfg["result_root"], exp_type)
    if not os.path.exists(exp_dir):
        return

    target_name = (
        f"{exp_type}_opt{cfg['optimiser']}_wd{cfg['weight_decay']}"
        f"_sched{cfg['scheduler']}_lr{cfg['lr']}.txt"
    )

    target_path = os.path.join(exp_dir, target_name)
    if os.path.exists(target_path):
        os.remove(target_path)
        print(f"[Cleanup] Removed old result file: {target_path}")
    else:
        print(f"[Cleanup] No existing .txt result file named {target_name}.")


# ==========================================
# Main
# ==========================================
def main():
    print(f"\n=== EEG→Dynamic Predictor ({EXPERIMENT_TYPE}) ===")
    cfg = CONFIG.copy()
    subjects = cfg["subjects_to_train"]
    all_acc = []

    clean_old_result_files(cfg, EXPERIMENT_TYPE)
    labels_all = create_binary_labels(cfg["optical_flow_path"])

    for sub in subjects:
        print(f"\n=== Running Dynamic Predictor for {sub} ===")

        # Load and preprocess data
        eeg = load_de_data(cfg, sub)
        eeg = preprocess_data(eeg)
        train_eeg, _, test_eeg = split_data(eeg)

        samples_per_block = 40 * 5
        train_labels = labels_all[:samples_per_block * 5]
        test_labels = labels_all[samples_per_block * 6 : samples_per_block * 7]

        train_dataset = EEGBinaryDataset(
            torch.tensor(train_eeg, dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.long)
        )
        test_dataset = EEGBinaryDataset(
            torch.tensor(test_eeg, dtype=torch.float32),
            torch.tensor(test_labels, dtype=torch.long)
        )

        train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False)

        # Model setup
        model = glfnet_mlp(out_dim=2, emb_dim=cfg["emb_dim"], input_dim=cfg["input_dim"]).to(cfg["device"])

        # Train model
        acc = train_model(model, train_loader, test_loader, cfg)
        all_acc.append(acc)

    # Save results
    save_results(cfg, all_acc, subjects, EXPERIMENT_TYPE)


if __name__ == "__main__":
    main()
