# ==========================================
# EEG Classification Script (DE-Only + Experiment Type)
# ==========================================
# Performs subject-wise EEG classification using DE features only.
# Uses GLFNet-MLP encoder and saves results under experiment-type folders.
# ==========================================


# === Imports ===
import os
import numpy as np
import importlib
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils import data


# === Encoder Import ===
from models.glfnet_mlp import glfnet_mlp


# ==========================================
# Architectural Fine-Tuning Toggle
# ==========================================
# Choose ONE: "layer_width", "dropout", "activation", "normalisation"
EXPERIMENT_TYPE = "activation"


# ==========================================
# 1. Configuration Table
# ==========================================
CONFIG = {
    "feature_type": "de",
    "encoder_name": "glfnet_mlp",
    "subjects_to_train": [
        "sub1_session2.npy",
        "sub1.npy",
        "sub18.npy",
        "sub19.npy",
        "sub6.npy",
        "sub15.npy",
        "sub20.npy",
        "sub7.npy",
        "sub10.npy",
        "sub13.npy",
    ],
    "dry_run": False,

    # --- Data parameters ---
    "num_classes": 40,
    "channels": 62,
    "time_len": 5,
    "num_blocks": 7,
    "clips_per_class": 5,

    # --- Paths ---
    "data_root": "/content/drive/MyDrive/EEG2Video_data/processed/",
    "de_dir": "EEG_DE_1per1s/",

    # --- Model parameters ---
    "emb_dim": 64,

    # --- Training parameters ---
    "batch_size": 128,
    "num_epochs": 100,
    "lr": 0.0005,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ==========================================
# 2. Data Handling
# ==========================================
def create_labels(num_classes, clips_per_class, blocks=1):
    block_labels = np.repeat(np.arange(num_classes), clips_per_class)
    return np.tile(block_labels, blocks)


def load_feature_data(sub_file, cfg):
    path = os.path.join(cfg["data_root"], cfg["de_dir"], sub_file)
    data = np.load(path)  # (7, 40, 5, 2, 62, 5)
    print(f"Loaded {sub_file} | shape {data.shape}")
    return data


def preprocess_data(data, cfg):
    """Preprocess DE EEG features (7,40,5,2,62,5) into train/val/test."""
    data = data.mean(axis=3)  # average trials -> (7,40,5,62,5)
    b, c, d, f, g = data.shape
    data = data.reshape(b, c * d, f, g)

    # normalization
    scaler = StandardScaler()
    flat_all = data.reshape(-1, g)
    scaler.fit(flat_all)
    scaled = scaler.transform(flat_all).reshape(b, c * d, f, g)

    # split by blocks
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
# 3. Training and Evaluation
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(cfg["num_epochs"]):
        model.train()
        total_loss, total_acc, total_count = 0, 0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.shape[0]
            total_acc += (y_hat.argmax(1) == y).sum().item()
            total_count += X.shape[0]
        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch+1}] Loss={total_loss/total_count:.4f} | Acc={total_acc/total_count:.4f}")

    # evaluation
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
# 4. Main Execution
# ==========================================
def main(cfg):
    base_dir = "/content/drive/MyDrive/EEG2Video_results/EEG_VP_benchmark"
    fine_dir = os.path.join(base_dir, "architectural_fine-tuning", EXPERIMENT_TYPE)
    os.makedirs(fine_dir, exist_ok=True)

    subjects = [s for s in os.listdir(os.path.join(cfg["data_root"], cfg["de_dir"])) if s in cfg["subjects_to_train"]]
    print(f"\nTraining subjects: {subjects}\n")

    all_top1, all_top5 = [], []

    for sub in subjects:
        print(f"\n=== Subject: {sub} ===")
        data = load_feature_data(sub, cfg)
        processed, labels = preprocess_data(data, cfg)
        train_iter = Get_Dataloader(processed["train"], labels["train"], True, cfg["batch_size"])
        test_iter  = Get_Dataloader(processed["test"],  labels["test"],  False, cfg["batch_size"])

        model = glfnet_mlp(out_dim=cfg["num_classes"], emb_dim=cfg["emb_dim"], input_dim=310)
        top1, top5 = train_and_eval(model, train_iter, test_iter, cfg)
        print(f"Test (Block 7): Top-1={top1:.4f}, Top-5={top5:.4f}")
        all_top1.append(top1)
        all_top5.append(top5)

    mean_top1, std_top1 = np.mean(all_top1), np.std(all_top1)
    mean_top5, std_top5 = np.mean(all_top5), np.std(all_top5)

    print("\n=== Final Results ===")
    print(f"Mean Top-1: {mean_top1:.4f} ± {std_top1:.4f}")
    print(f"Mean Top-5: {mean_top5:.4f} ± {std_top5:.4f}")

    # ==========================================
    # Save Configuration and Results (Auto)
    # ==========================================
    model_cfg = getattr(importlib.import_module("models.glfnet_mlp"), "CONFIG", {})

    lw = "-".join(str(x) for x in model_cfg.get("layer_widths", [])) if "layer_widths" in model_cfg else "NA"
    dropout = model_cfg.get("dropout", "NA")
    activation = model_cfg.get("activation", "NA")
    norm = model_cfg.get("normalization", model_cfg.get("normalisation", "NA"))

    filename = (
        f"{EXPERIMENT_TYPE}_de_mlp_lw{lw}_do{dropout}_act{activation}_norm{norm}_"
        f"emb{cfg['emb_dim']}_lr{cfg['lr']}_bs{cfg['batch_size']}_ep{cfg['num_epochs']}.txt"
    )

    save_path = os.path.join(fine_dir, filename)
    with open(save_path, "w") as f:
        f.write("EEG DE Classification Summary\n")
        f.write("==========================================\n\n")
        f.write(f"Architectural Fine-Tuning Type: {EXPERIMENT_TYPE}\n\n")
        f.write("Configuration Used:\n")
        for k, v in cfg.items():
            f.write(f"{k}: {v}\n")
        f.write("\nModel Config:\n")
        for k, v in model_cfg.items():
            f.write(f"{k}: {v}\n")
        f.write("\nFinal Results:\n")
        f.write(f"Mean Top-1: {mean_top1:.4f} ± {std_top1:.4f}\n")
        f.write(f"Mean Top-5: {mean_top5:.4f} ± {std_top5:.4f}\n\n")
        f.write("Subject-Wise Results:\n")
        for sub, t1, t5 in zip(subjects, all_top1, all_top5):
            f.write(f"{sub:15s} | Top-1: {t1:.4f} | Top-5: {t5:.4f}\n")

    print(f"\nSaved configuration and results to: {save_path}")


if __name__ == "__main__":
    main(CONFIG)
