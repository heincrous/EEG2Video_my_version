# ==========================================
# EEG → Dynamic Predictor (DE-only, Fast/Slow Classification)
# ==========================================
# Performs subject-wise EEG classification using DE features only.
# Uses GLFNet-MLP encoder (emb_dim=64) and optical flow labels for Fast/Slow.
# Supports optimizer, scheduler, and learning rate experiments.
# ==========================================


# === Imports ===
import os
import numpy as np
import importlib
import torch
import torch.nn as nn
from torch.utils import data
from sklearn.preprocessing import StandardScaler


# === Encoder Import ===
from models.glfnet_mlp import glfnet_mlp


# ==========================================
# Experiment Toggle
# ==========================================
# Choose one: "optimizer", "scheduler", "learning_rate"
EXPERIMENT_TYPE = "optimizer"


# ==========================================
# 1. Configuration Table
# ==========================================
CONFIG = {
    "feature_type"      : "DE",
    "encoder_name"      : "glfnet_mlp",
    "subjects_to_train" : ["sub1.npy"],

    # --- Data parameters ---
    "num_classes"       : 2,
    "channels"          : 62,
    "time_len"          : 5,
    "num_blocks"        : 7,
    "clips_per_class"   : 5,

    # --- Paths ---
    "data_root"         : "/content/drive/MyDrive/EEG2Video_data/processed/",
    "de_dir"            : "EEG_DE_1per1s/",
    "optical_flow_path" : "/content/drive/MyDrive/EEG2Video_data/processed/meta-info/All_video_optical_flow_score_byclass.npy",
    "save_root"         : "/content/drive/MyDrive/EEG2Video_results/dynamic_predictor/",
    "checkpoint_dir"    : "/content/drive/MyDrive/EEG2Video_checkpoints/dynamic_predictor/",

    # --- Model parameters ---
    "emb_dim"           : 64,
    "input_dim"         : 62 * 5,

    # --- Training parameters ---
    "batch_size"        : 128,
    "num_epochs"        : 100,
    "lr"                : 0.0005,
    "optimizer"         : "adamw",          # ["adam", "adamw"]
    "weight_decay"      : 0.25,
    "scheduler"         : "constant",       # ["constant", "cosine"]
    "device"            : "cuda" if torch.cuda.is_available() else "cpu",
}


# ==========================================
# 2. Data Handling
# ==========================================
def create_binary_labels(flow_path):
    """Generate binary fast/slow labels using optical flow threshold."""
    flow = np.load(flow_path).reshape(-1)  # (1400,)
    threshold = np.median(flow)
    labels = (flow > threshold).astype(int)
    return labels


def load_feature_data(sub_file, cfg):
    path = os.path.join(cfg["data_root"], cfg["de_dir"], sub_file)
    data = np.load(path)  # (7,40,5,2,62,5)
    print(f"Loaded {sub_file} | shape {data.shape}")
    return data


def preprocess_data(data, cfg):
    """Preprocess DE EEG features (7,40,5,2,62,5) into train/val/test."""
    data = data.mean(axis=3)  # average trials -> (7,40,5,62,5)
    b, c, d, f, g = data.shape
    data = data.reshape(b, c * d, f, g)

    scaler = StandardScaler()
    flat = data.reshape(-1, g)
    scaler.fit(flat)
    scaled = scaler.transform(flat).reshape(b, c * d, f, g)

    train_data, val_data, test_data = scaled[:5], scaled[5:6], scaled[6:7]
    processed = {
        "train": train_data.reshape(-1, f, g),
        "val":   val_data.reshape(-1, f, g),
        "test":  test_data.reshape(-1, f, g),
    }
    return processed


def Get_Dataloader(data_split, labels, istrain, batch_size):
    X = torch.tensor(data_split, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return data.DataLoader(data.TensorDataset(X, y), batch_size=batch_size, shuffle=istrain)


# ==========================================
# 3. Training and Evaluation
# ==========================================
def build_optimizer(model, cfg):
    if cfg["optimizer"].lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    elif cfg["optimizer"].lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        raise ValueError("Unknown optimizer type. Choose 'adam' or 'adamw'.")


def build_scheduler(optimizer, cfg):
    if cfg["scheduler"].lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"])
    elif cfg["scheduler"].lower() == "constant":
        return None
    else:
        raise ValueError("Unknown scheduler type. Choose 'constant' or 'cosine'.")


def train_and_eval(model, train_iter, test_iter, cfg):
    device = cfg["device"]
    model.to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
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
            if scheduler:
                scheduler.step()
            total_loss += loss.item() * X.shape[0]
            total_acc += (y_hat.argmax(1) == y).sum().item()
            total_count += X.shape[0]
        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch+1}] Loss={total_loss/total_count:.4f} | Acc={total_acc/total_count:.4f}")

    # evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            correct += (y_hat.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


# ==========================================
# 4. Main Execution
# ==========================================
def main(cfg):
    exp_dir = os.path.join(cfg["save_root"], EXPERIMENT_TYPE)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    subjects = [s for s in os.listdir(os.path.join(cfg["data_root"], cfg["de_dir"])) if s in cfg["subjects_to_train"]]
    print(f"\nTraining subjects: {subjects}\n")

    labels_all = create_binary_labels(cfg["optical_flow_path"])
    all_acc = []

    for sub in subjects:
        print(f"\n=== Subject: {sub} ===")
        data = load_feature_data(sub, cfg)
        processed = preprocess_data(data, cfg)

        # Split by blocks: 5 train, 1 val, 1 test
        samples_per_block = 40 * 5
        train_labels = labels_all[:samples_per_block * 5]   # 1000
        test_labels  = labels_all[samples_per_block * 6 : samples_per_block * 7]  # 200

        train_iter = Get_Dataloader(processed["train"], train_labels, True, cfg["batch_size"])
        test_iter  = Get_Dataloader(processed["test"],  test_labels,  False, cfg["batch_size"])

        model = glfnet_mlp(out_dim=2, emb_dim=cfg["emb_dim"], input_dim=cfg["input_dim"])
        acc = train_and_eval(model, train_iter, test_iter, cfg)
        print(f"Test Accuracy: {acc:.4f}")
        all_acc.append(acc)

        ckpt_path = os.path.join(cfg["checkpoint_dir"], f"dynpred_{EXPERIMENT_TYPE}_{sub.replace('.npy','')}.pt")
        torch.save({"state_dict": model.state_dict()}, ckpt_path)

    mean_acc = np.mean(all_acc)
    print("\n=== Final Results ===")
    print(f"Mean Accuracy: {mean_acc:.4f}")


    # ==========================================
    # Save Configuration and Results
    # ==========================================
    model_cfg = getattr(importlib.import_module("models.glfnet_mlp"), "CONFIG", {})
    filename = (
        f"{EXPERIMENT_TYPE}_opt{cfg['optimizer']}_sched{cfg['scheduler']}_"
        f"lr{cfg['lr']}_wd{cfg['weight_decay']}_bs{cfg['batch_size']}_ep{cfg['num_epochs']}.txt"
    )
    save_path = os.path.join(exp_dir, filename)

    with open(save_path, "w") as f:
        f.write("EEG DE → Dynamic Predictor (Fast/Slow)\n")
        f.write("==========================================\n\n")
        f.write(f"Experiment Type: {EXPERIMENT_TYPE}\n\n")
        f.write("Configuration Used:\n")
        for k, v in cfg.items():
            f.write(f"{k}: {v}\n")
        f.write("\nModel Config:\n")
        for k, v in model_cfg.items():
            f.write(f"{k}: {v}\n")
        f.write("\nFinal Results:\n")
        f.write(f"Mean Accuracy: {mean_acc:.4f}\n\n")
        f.write("Subject-Wise Results:\n")
        for sub, acc in zip(subjects, all_acc):
            f.write(f"{sub:15s} | Acc: {acc:.4f}\n")

    print(f"\nSaved configuration and results to: {save_path}")


if __name__ == "__main__":
    main(CONFIG)
