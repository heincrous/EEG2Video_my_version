# ==========================================
# EEG Classification Script (DE-Only Version)
# ==========================================
# Performs subject-wise EEG classification using DE features.
# Uses the GLFNet-MLP encoder and follows the same modular format
# and auto-saving system as the full benchmark script.
# ==========================================


# === Imports ===
import os
import numpy as np
import importlib
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils import data
from einops import rearrange


# === Encoder Import ===
from models.glfnet_mlp import glfnet_mlp


# ==========================================
# 1. Configuration Table
# ==========================================
CONFIG = {
    "feature_type": "de",  # fixed
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
    "time_len": 5,  # DE features per channel
    "num_blocks": 7,
    "clips_per_class": 5,

    # --- Paths ---
    "data_root": "/content/drive/MyDrive/EEG2Video_data/processed/",
    "de_dir": "EEG_DE_1per1s/",

    # --- Training parameters ---
    "batch_size": 256,
    "num_epochs": 100,
    "lr": 0.0005,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ==========================================
# 2. Data Handling Module
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
    """Preprocess DE data into train/val/test splits with normalization."""
    data = data.mean(axis=3)  # average over trials -> (7,40,5,62,5)
    b, c, d, f, g = data.shape  # (7,40,5,62,5)
    data = data.reshape(b, c * d, f, g)  # (7,200,62,5)

    # global normalization
    scaler = StandardScaler()
    flat_all = data.reshape(-1, g)
    scaler.fit(flat_all)
    scaled = scaler.transform(flat_all).reshape(b, c * d, f, g)

    # split by blocks (5 train, 1 val, 1 test)
    train_data = scaled[:5]
    val_data   = scaled[5:6]
    test_data  = scaled[6:7]

    train_labels = create_labels(cfg["num_classes"], cfg["clips_per_class"], 5)
    val_labels   = create_labels(cfg["num_classes"], cfg["clips_per_class"])
    test_labels  = create_labels(cfg["num_classes"], cfg["clips_per_class"])

    train_data = train_data.reshape(-1, f, g)
    val_data   = val_data.reshape(-1, f, g)
    test_data  = test_data.reshape(-1, f, g)

    return train_data, val_data, test_data, train_labels, val_labels, test_labels


def Get_Dataloader(features, labels, istrain, batch_size):
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return data.DataLoader(data.TensorDataset(X, y), batch_size=batch_size, shuffle=istrain)


# ==========================================
# 3. Training and Evaluation Module
# ==========================================
def cal_accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, axis=1)
    return torch.sum(y_hat == y).item()


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


def train_and_eval(model, train_iter, val_iter, test_iter, cfg):
    device = cfg["device"]
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(cfg["num_epochs"]):
        model.train()
        total_loss, total_acc, total_count = 0, 0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.shape[0]
            total_acc += cal_accuracy(y_hat, y)
            total_count += X.shape[0]
        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch+1}] Loss {total_loss/total_count:.4f} | Acc {total_acc/total_count:.4f}")

    # --- Evaluation ---
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
    data_dir = os.path.join(cfg["data_root"], cfg["de_dir"])
    subjects = [s for s in os.listdir(data_dir) if s in cfg["subjects_to_train"]]
    print(f"\nTraining subjects: {subjects}\n")

    all_top1, all_top5 = [], []

    for sub in subjects:
        print(f"\n=== Subject: {sub} ===")
        data = load_feature_data(sub, cfg)
        train_data, val_data, test_data, train_labels, val_labels, test_labels = preprocess_data(data, cfg)

        train_iter = Get_Dataloader(train_data, train_labels, True, cfg["batch_size"])
        val_iter   = Get_Dataloader(val_data, val_labels, False, cfg["batch_size"])
        test_iter  = Get_Dataloader(test_data, test_labels, False, cfg["batch_size"])

        model = glfnet_mlp(out_dim=cfg["num_classes"], emb_dim=64, input_dim=310)
        print(f"Using encoder: {cfg['encoder_name']} | Feature type: {cfg['feature_type']}")

        top1, top5 = train_and_eval(model, train_iter, val_iter, test_iter, cfg)
        print(f"Test (Block 7): Top-1={top1:.4f}, Top-5={top5:.4f}\n")
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
    base_dir = "/content/drive/MyDrive/EEG2Video_results/EEG_VP_benchmark"
    feature_dir = os.path.join(base_dir, "de")
    os.makedirs(feature_dir, exist_ok=True)

    try:
        model_module = importlib.import_module("models.glfnet_mlp")
        model_cfg = getattr(model_module, "CONFIG", {})
    except Exception:
        model_cfg = {}

    if "layer_widths" in model_cfg:
        lw_value = model_cfg["layer_widths"]
        layer_widths = "-".join(str(x) for x in lw_value)
    elif "layer_width" in model_cfg:
        layer_widths = str(model_cfg["layer_width"])
    else:
        layer_widths = "NA"

    dropout = model_cfg.get("dropout", "NA")
    activation = model_cfg.get("activation", "NA")
    normalisation = model_cfg.get("normalization", model_cfg.get("normalisation", "NA"))

    print(cfg["encoder_name"], "CONFIG seen by writer:")
    for k, v in model_cfg.items():
        print(f"  {k!r}: {v!r}")

    filename = (
        f"{cfg['encoder_name']}_lw{layer_widths}_do{dropout}_"
        f"act{activation}_norm{normalisation}_"
        f"lr{cfg['lr']}_bs{cfg['batch_size']}_ep{cfg['num_epochs']}.txt"
    )

    save_path = os.path.join(feature_dir, filename)
    with open(save_path, "w") as f:
        f.write("EEG DE Classification Summary\n")
        f.write("==========================================\n\n")
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
