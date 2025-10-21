# ==========================================
# EEG CLASSIFICATION (FULL MODULAR + FUNCTIONAL DRY RUN)
# ==========================================
# Input: Preprocessed EEG features (segment, DE, or PSD)
# Process: Train or dry-run multiple encoder architectures per subject
# Output: Top-1 and Top-5 accuracies (.txt)
# ==========================================

import os
import numpy as np
import importlib
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils import data
from einops import rearrange
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# Encoder Imports
# ==========================================
from models.shallownet import shallownet
from models.eegnet import eegnet
from models.deepnet import deepnet
from models.conformer import conformer
from models.tsconv import tsconv
from models.glfnet import glfnet
from models.mlpnet import mlpnet
from models.glfnet_mlp import glfnet_mlp

# ==========================================
# Default Configuration
# ==========================================
"""
DEFAULT MODEL CONFIGURATION
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
EXPERIMENT_MODE = "classification"
RESULT_ROOT = "/content/drive/MyDrive/EEG2Video_results/EEG_VP_benchmark/classification"


# ==========================================
# Configuration Table
# ==========================================
CONFIG = {
    "feature_type": "psd",
    "encoder_name": "glfnet_mlp",
    "subjects_to_train": [
        "sub1_session2.npy", "sub1.npy", "sub18.npy", "sub19.npy",
        "sub6.npy", "sub15.npy", "sub20.npy", "sub7.npy", "sub10.npy", "sub13.npy"
    ],
    "num_classes": 40,
    "channels": 62,
    "time_len": 400,
    "num_blocks": 7,
    "clips_per_class": 5,
    "data_root": "/content/drive/MyDrive/EEG2Video_data/processed/",
    "segment_dir": "EEG_segments/",
    "de_dir": "EEG_DE_1per1s/",
    "psd_dir": "EEG_PSD_1per1s/",
    "batch_size": 128,
    "num_epochs": 100,
    "lr": 0.0005,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "result_root": RESULT_ROOT,
    "dry_run": False,
    "save_confusion_matrices": True,   # Set False to skip confusion matrices
}


# ==========================================
# Data Handling
# ==========================================
def create_labels(num_classes, clips_per_class, blocks=1):
    block_labels = np.repeat(np.arange(num_classes), clips_per_class)
    return np.tile(block_labels, blocks)


def load_feature_data(sub_file, cfg):
    if cfg["feature_type"] == "segment":
        path = os.path.join(cfg["data_root"], cfg["segment_dir"], sub_file)
    elif cfg["feature_type"] == "de":
        path = os.path.join(cfg["data_root"], cfg["de_dir"], sub_file)
    elif cfg["feature_type"] == "psd":
        path = os.path.join(cfg["data_root"], cfg["psd_dir"], sub_file)
    else:
        raise ValueError("Unsupported feature type")

    data = np.load(path)
    print(f"Loaded {sub_file} | shape {data.shape}")
    return data


def preprocess_data(data, cfg):
    """Preprocess EEG feature array into train/val/test splits with scaling."""
    if cfg["feature_type"] == "segment":
        b, c, d, f, g = data.shape
    else:
        data = data.mean(axis=3)
        b, c, d, f, g = data.shape

    data = data.reshape(b, c * d, f, g)

    scaler = StandardScaler()
    flat_all = data.reshape(-1, g)
    scaler.fit(flat_all)
    scaled = scaler.transform(flat_all).reshape(b, c * d, f, g)

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


def Get_Dataloader(features, labels, istrain, batch_size):
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return data.DataLoader(data.TensorDataset(X, y), batch_size=batch_size, shuffle=istrain)


# ==========================================
# Model Selection
# ==========================================
def select_model(cfg):
    ft = cfg["feature_type"]
    en = cfg["encoder_name"]

    if ft == "segment" and en in ["mlpnet", "glfnet_mlp"]:
        en = "glfnet"
    elif ft in ["de", "psd"] and en not in ["mlpnet", "glfnet_mlp"]:
        en = "glfnet_mlp"

    model_map = {
        "shallownet": shallownet,
        "eegnet": eegnet,
        "deepnet": deepnet,
        "conformer": conformer,
        "tsconv": tsconv,
        "glfnet": glfnet,
        "mlpnet": mlpnet,
        "glfnet_mlp": glfnet_mlp,
    }

    ModelClass = model_map[en]
    if en == "glfnet":
        model = ModelClass(out_dim=cfg["num_classes"], emb_dim=64, C=cfg["channels"], T=cfg["time_len"])
    elif en in ["shallownet", "deepnet", "eegnet", "tsconv"]:
        model = ModelClass(out_dim=cfg["num_classes"], C=cfg["channels"], T=cfg["time_len"])
    elif en == "conformer":
        model = ModelClass(out_dim=cfg["num_classes"])
    elif en == "glfnet_mlp":
        model = ModelClass(out_dim=cfg["num_classes"], emb_dim=64, input_dim=310)
    elif en == "mlpnet":
        model = ModelClass(out_dim=cfg["num_classes"], input_dim=310)
    else:
        raise ValueError(f"Unknown encoder: {en}")

    print(f"Using encoder: {en} | Feature type: {ft}")
    return model


# ==========================================
# Functional Dry Run
# ==========================================
def simulate_preprocessed_dummy(ft, cfg):
    """Generate correctly shaped dummy tensors to test model compatibility."""
    C, T = cfg["channels"], cfg["time_len"]
    device = cfg["device"]

    if ft == "segment":
        raw = torch.randn(7, 40, 5, C, T)
        flat = rearrange(raw, "b c d f g -> (b c d) f g").unsqueeze(1)
    else:
        raw = torch.randn(7, 40, 5, 2, C, 5)
        flat = rearrange(raw.mean(axis=3), "b c d f g -> (b c d) f g")

    flat = (flat - flat.mean()) / (flat.std() + 1e-6)
    return flat.to(device)


def dry_run_all_models(cfg):
    """Run all compatible model-feature pairs for forward validation."""
    print("\n=== Running Full Functional Dry Run ===")
    device = cfg["device"]

    model_groups = {
        "segment": ["shallownet", "deepnet", "eegnet", "tsconv", "conformer", "glfnet"],
        "de": ["mlpnet", "glfnet_mlp"],
        "psd": ["mlpnet", "glfnet_mlp"],
    }

    for ft, models in model_groups.items():
        dummy_input = simulate_preprocessed_dummy(ft, cfg)
        print(f"\n--- Feature: {ft.upper()} --- Input shape: {list(dummy_input.shape)}")
        for name in models:
            try:
                tmp_cfg = cfg.copy()
                tmp_cfg["feature_type"] = ft
                tmp_cfg["encoder_name"] = name
                model = select_model(tmp_cfg).to(device)
                model.eval()
                with torch.no_grad():
                    out = model(dummy_input)
                print(f"{name:12s} | Output: {list(out.shape)} | OK")
            except Exception as e:
                print(f"{name:12s} | ERROR: {str(e)}")

    print("\n=== Dry Run Complete ===\n")


# ==========================================
# Training and Evaluation
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
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in tqdm(range(1, cfg["num_epochs"] + 1)):
        total_loss, total_acc, total_count = 0.0, 0.0, 0
        model.train()
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            if cfg["feature_type"] == "segment":
                X = X.unsqueeze(1)
            optimiser.zero_grad()
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * X.shape[0]
            total_acc += (y_hat.argmax(1) == y).sum().item()
            total_count += X.shape[0]

        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] Loss={total_loss/total_count:.4f} | Acc={total_acc/total_count:.4f}")

    model.eval()
    top1_all, top5_all = [], []
    with torch.no_grad():
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            if cfg["feature_type"] == "segment":
                X = X.unsqueeze(1)
            y_hat = model(X)
            t1, t5 = topk_accuracy(y_hat, y)
            top1_all.append(t1)
            top5_all.append(t5)

    return np.mean(top1_all), np.mean(top5_all)


# ==========================================
# Confusion Matrix Generation and Plotting
# ==========================================
def plot_confusion_matrix(model, dataloader, cfg, subject_name):
    """Compute and save confusion matrix for a trained model."""
    if not cfg.get("save_confusion_matrices", False):
        return

    device = cfg["device"]
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            if cfg["feature_type"] == "segment":
                X = X.unsqueeze(1)
            preds = model(X).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    # === Save under feature-type subfolder ===
    plot_dir = os.path.join(cfg["result_root"], "plots", cfg["feature_type"])
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        cmap="Blues",
        annot=False,
        xticklabels=False,
        yticklabels=False,
        cbar=True,
        square=True,
        vmin=0,
        vmax=0.3,
    )
    plt.title(f"Confusion Matrix – {cfg['encoder_name']} ({subject_name})", fontsize=14, pad=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(plot_dir, f"{cfg['encoder_name']}_confmat_{subject_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved Confusion Matrix] {save_path}")
    return cm


# ==========================================
# Average Confusion Matrix Across Subjects
# ==========================================
def plot_average_confusion_matrix(all_cms, cfg):
    """Compute and save an averaged confusion matrix across all subjects."""
    if not all_cms or not cfg.get("save_confusion_matrices", False):
        return

    avg_cm = np.mean(all_cms, axis=0)
    avg_cm_norm = avg_cm / (avg_cm.sum(axis=1, keepdims=True) + 1e-8)

    plot_dir = os.path.join(cfg["result_root"], "plots", cfg["feature_type"])
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        avg_cm_norm,
        cmap="Blues",
        annot=False,
        xticklabels=False,
        yticklabels=False,
        cbar=True,
        square=True,
        vmin=0,
        vmax=0.3,
    )
    plt.title(f"Average Confusion Matrix – {cfg['encoder_name']}", fontsize=14, pad=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(plot_dir, f"{cfg['encoder_name']}_avg_confmat.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved Average Confusion Matrix] {save_path}")


# ==========================================
# Save Results
# ==========================================
def save_results(cfg, subjects, all_top1, all_top5):
    mean_top1, std_top1 = np.mean(all_top1), np.std(all_top1)
    mean_top5, std_top5 = np.mean(all_top5), np.std(all_top5)
    exp_dir = os.path.join(cfg["result_root"], cfg["feature_type"])
    os.makedirs(exp_dir, exist_ok=True)

    try:
        model_module = importlib.import_module(f"models.{cfg['encoder_name']}")
        model_cfg = getattr(model_module, "CONFIG", {})
    except Exception:
        model_cfg = {}

    lw = "NA"
    if "layer_widths" in model_cfg:
        lw = "-".join(str(x) for x in model_cfg["layer_widths"])
    elif "layer_width" in model_cfg:
        lw = str(model_cfg["layer_width"])

    dropout = model_cfg.get("dropout", "NA")
    activation = model_cfg.get("activation", "NA")
    norm = model_cfg.get("normalisation", model_cfg.get("normalisation", "NA"))

    filename = f"{cfg['encoder_name']}.txt"
    save_path = os.path.join(exp_dir, filename)

    with open(save_path, "w") as f:
        f.write("EEG Classification Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write("Configuration:\n")
        for k, v in cfg.items():
            f.write(f"{k}: {v}\n")

        f.write("\nFinal Metrics:\n")
        f.write(f"Mean Top-1: {mean_top1:.4f} ± {std_top1:.4f}\n")
        f.write(f"Mean Top-5: {mean_top5:.4f} ± {std_top5:.4f}\n\n")

        f.write("Subject-Wise Results:\n")
        for s, t1, t5 in zip(subjects, all_top1, all_top5):
            f.write(f"{s:15s} | Top-1: {t1:.4f} | Top-5: {t5:.4f}\n")

    print(f"[Saved Results] {save_path}")


# ==========================================
# Cleanup Utilities
# ==========================================
def clean_old_results(cfg):
    """
    Remove only result and plot files matching the same encoder
    and feature-type configuration.
    """
    exp_dir = os.path.join(cfg["result_root"], cfg["feature_type"])
    plot_dir = os.path.join(cfg["result_root"], "plots", cfg["feature_type"])

    try:
        model_module = importlib.import_module(f"models.{cfg['encoder_name']}")
        model_cfg = getattr(model_module, "CONFIG", {})
    except Exception:
        model_cfg = {}

    lw = model_cfg.get("layer_widths", model_cfg.get("layer_width", "NA"))
    if isinstance(lw, (list, tuple)):
        lw = "-".join(str(x) for x in lw)
    dropout = model_cfg.get("dropout", "NA")
    activation = model_cfg.get("activation", "NA")
    norm = model_cfg.get("normalisation", model_cfg.get("norm", "NA"))
    signature = f"{cfg['encoder_name']}_w{lw}_d{dropout}_a{activation}_n{norm}"

    deleted = []

    # Delete matching results
    if os.path.exists(exp_dir):
        for f in os.listdir(exp_dir):
            if f.endswith(".txt") and signature in f:
                os.remove(os.path.join(exp_dir, f))
                deleted.append(f)

    # Delete matching confusion matrices
    if os.path.exists(plot_dir):
        for f in os.listdir(plot_dir):
            if f.endswith(".png") and cfg["encoder_name"] in f:
                os.remove(os.path.join(plot_dir, f))
                deleted.append(f)

    if deleted:
        print(f"[Cleanup] Removed old matching files: {deleted}")
    else:
        print("[Cleanup] No matching result or plot files found.")


# ==========================================
# Main
# ==========================================
def main(cfg):
    if cfg["dry_run"]:
        dry_run_all_models(cfg)
        return

    clean_old_results(cfg)

    if cfg["feature_type"] == "segment":
        data_dir = os.path.join(cfg["data_root"], cfg["segment_dir"])
    elif cfg["feature_type"] == "de":
        data_dir = os.path.join(cfg["data_root"], cfg["de_dir"])
    else:
        data_dir = os.path.join(cfg["data_root"], cfg["psd_dir"])

    subjects = [s for s in os.listdir(data_dir) if s in cfg["subjects_to_train"]]
    print(f"\nTraining subjects: {subjects}\n")

    all_top1, all_top5 = [], []
    all_confmats = []

    for sub in subjects:
        print(f"\n=== Subject: {sub} ===")
        data = load_feature_data(sub, cfg)
        processed, labels = preprocess_data(data, cfg)

        train_iter = Get_Dataloader(processed["train"], labels["train"], True, cfg["batch_size"])
        test_iter  = Get_Dataloader(processed["test"],  labels["test"],  False, cfg["batch_size"])
        model = select_model(cfg)
        top1, top5 = train_and_eval(model, train_iter, test_iter, cfg)
        print(f"[Test Block-7] Top-1={top1:.4f} | Top-5={top5:.4f}")
        cm = plot_confusion_matrix(model, test_iter, cfg, subject_name=sub)
        if cm is not None:
            all_confmats.append(cm)
        all_top1.append(top1)
        all_top5.append(top5)

    plot_average_confusion_matrix(all_confmats, cfg)
    print("\n=== Final Results ===")
    print(f"Mean Top-1: {np.mean(all_top1):.4f} ± {np.std(all_top1):.4f}")
    print(f"Mean Top-5: {np.mean(all_top5):.4f} ± {np.std(all_top5):.4f}")

    save_results(cfg, subjects, all_top1, all_top5)


if __name__ == "__main__":
    main(CONFIG)
