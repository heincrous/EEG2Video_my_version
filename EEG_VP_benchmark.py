# ==========================================
# EEG Classification Script (Full Modular + Functional Dry Run)
# ==========================================
# Performs comprehensive dry-run forward passes on all models
# with simulated preprocessing for all feature types.
# Can also train EEG classifiers per subject when dry_run=False.
# ==========================================


# === Imports ===
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils import data
from einops import rearrange

# === Encoder Imports ===
from models.shallownet import shallownet
from models.eegnet import eegnet
from models.deepnet import deepnet
from models.conformer import conformer
from models.tsconv import tsconv
from models.glfnet import glfnet
from models.mlpnet import mlpnet
from models.glfnet_mlp import glfnet_mlp


# ==========================================
# 1. Configuration Table
# ==========================================
CONFIG = {
    "feature_type": "de",       # "window", "de", or "psd"
    "encoder_name": "glfnet",   # used for training
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
        "sub13.npy"
    ],
    # "subjects_to_train": "all",
    "dry_run": False,            # perform preprocessing + forward passes

    # --- Data parameters ---
    "num_classes": 40,
    "channels": 62,
    "time_len": 100,
    "num_blocks": 7,
    "clips_per_class": 5,

    # --- Paths ---
    "data_root": "/content/drive/MyDrive/EEG2Video_data/processed/",
    "window_dir": "EEG_window_100/",
    "de_dir": "EEG_DE_1per1s/",
    "psd_dir": "EEG_PSD_1per1s/",

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
    dir_map = {
        "window": cfg["window_dir"],
        "de": cfg["de_dir"],
        "psd": cfg["psd_dir"],
    }
    path = os.path.join(cfg["data_root"], dir_map[cfg["feature_type"]], sub_file)
    data = np.load(path)
    print(f"Loaded {sub_file} | shape {data.shape}")
    return data


def preprocess_data(data, cfg):
    """Split EEG data into train/val/test (blocks 1–5, 6, 7) using ONE global StandardScaler, preserving structure."""
    if cfg["feature_type"] == "window":
        data = data.mean(axis=3)  # (7, 40, 5, 62, 100)
    else:
        data = data.mean(axis=3)  # (7, 40, 5, 62, 5)

    # reshape to (blocks, samples_per_block, channels, features)
    b, c, d, f, g = data.shape  # (7, 40, 5, 62, 5)
    data = data.reshape(b, c * d, f, g)  # (7, 200, 62, 5)

    # fit one scaler on all blocks
    scaler = StandardScaler()
    flat_all = data.reshape(-1, g)  # flatten feature dimension only
    scaler.fit(flat_all)
    scaled = scaler.transform(flat_all).reshape(b, c * d, f, g)

    # now split cleanly by blocks
    train_data = scaled[:5]
    val_data   = scaled[5:6]
    test_data  = scaled[6:7]

    train_labels = create_labels(cfg["num_classes"], cfg["clips_per_class"], 5)
    val_labels   = create_labels(cfg["num_classes"], cfg["clips_per_class"])
    test_labels  = create_labels(cfg["num_classes"], cfg["clips_per_class"])

    # reshape to (samples, channels, features)
    train_data = train_data.reshape(-1, f, g)
    val_data   = val_data.reshape(-1, f, g)
    test_data  = test_data.reshape(-1, f, g)

    return train_data, val_data, test_data, train_labels, val_labels, test_labels


def Get_Dataloader(features, labels, istrain, batch_size):
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    return data.DataLoader(data.TensorDataset(features, labels), batch_size=batch_size, shuffle=istrain)


# ==========================================
# 3. Model Selection Module
# ==========================================
def select_model(cfg):
    ft = cfg["feature_type"]
    en = cfg["encoder_name"]

    # enforce correct encoder usage by feature type
    if ft == "window" and en in ["mlpnet", "glfnet_mlp"]:
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
# 4. Functional Dry Run Module (Fixed + Filtered)
# ==========================================
def simulate_preprocessed_dummy(ft, cfg):
    """Simulate realistic preprocessing and produce correctly shaped tensors."""
    C = cfg["channels"]
    T = cfg["time_len"]
    device = cfg["device"]

    if ft == "window":
        # Raw EEG window data: (7 blocks, 40 classes, 5 clips, 7 subjects, 62 ch, 100 time)
        raw = torch.randn(7, 40, 5, 7, C, T)
        avgd = raw.mean(axis=3)  # → (7, 40, 5, 62, 100)

        # Merge block, class, and clip → (1400, 62, 100)
        flat = rearrange(avgd, "b c d f g -> (b c d) f g")

        # Normalize
        flat = (flat - flat.mean()) / (flat.std() + 1e-6)

        # Add channel dimension for CNN input → (1400, 1, 62, 100)
        flat = flat.unsqueeze(1)

        return flat.to(device)

    else:
        # DE / PSD simulated: (7, 40, 5, 2, 62, 5)
        raw = torch.randn(7, 40, 5, 2, C, 5)
        avgd = raw.mean(axis=3)  # → (7, 40, 5, 62, 5)

        # Merge block, class, and clip → (1400, 62, 5)
        flat = rearrange(avgd, "b c d f g -> (b c d) f g")

        # Normalize
        flat = (flat - flat.mean()) / (flat.std() + 1e-6)

        return flat.to(device)


def dry_run_all_models(cfg):
    """Perform full preprocessing-simulated forward passes for all valid model-feature pairs."""
    print("\n=== Running Full Functional Dry Run (All Models + Features) ===")
    device = cfg["device"]

    # mapping which models to test per feature type
    model_groups = {
        "window": ["shallownet", "deepnet", "eegnet", "tsconv", "conformer", "glfnet"],
        "de": ["mlpnet", "glfnet_mlp"],
        "psd": ["mlpnet", "glfnet_mlp"],
    }

    for ft, model_names in model_groups.items():
        print(f"\n--- Testing feature type: {ft.upper()} ---")
        dummy_input = simulate_preprocessed_dummy(ft, cfg)
        print(f"Dummy preprocessed input shape: {list(dummy_input.shape)}")

        for name in model_names:
            try:
                temp_cfg = cfg.copy()
                temp_cfg["feature_type"] = ft
                temp_cfg["encoder_name"] = name
                model = select_model(temp_cfg).to(device)
                model.eval()
                with torch.no_grad():
                    out = model(dummy_input)
                print(f"{name:12s} | Output shape: {list(out.shape)} | OK")
            except Exception as e:
                print(f"{name:12s} | ERROR: {str(e)}")

    print("\n=== Dry Run Complete ===\n")


# ==========================================
# 5. Training and Evaluation Module
# ==========================================
def cal_accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, axis=1)
    cmp = (y_hat == y)
    return torch.sum(cmp, dim=0)


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0)
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = cfg["num_epochs"]

    for epoch in range(num_epochs):
        total_loss, total_acc, total_count = 0, 0, 0
        model.train()
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

    model.eval()
    top1_all, top5_all = [], []
    with torch.no_grad():
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            top1, top5 = topk_accuracy(y_hat, y, topk=(1, 5))
            top1_all.append(top1)
            top5_all.append(top5)

    print(f"Top-1 Acc: {np.mean(top1_all):.4f} | Top-5 Acc: {np.mean(top5_all):.4f}")
    return np.mean(top1_all), np.mean(top5_all)


# ==========================================
# 6. Main Execution
# ==========================================
def main(cfg):
    if cfg["dry_run"]:
        dry_run_all_models(cfg)
        return

    dir_map = {
        "window": cfg["window_dir"],
        "de": cfg["de_dir"],
        "psd": cfg["psd_dir"],
    }
    data_dir = os.path.join(cfg["data_root"], dir_map[cfg["feature_type"]])
    subjects = os.listdir(data_dir)
    if cfg["subjects_to_train"] != "all":
        subjects = [s for s in subjects if s in cfg["subjects_to_train"]]

    print(f"\nTraining subjects: {subjects}\n")
    all_top1, all_top5 = [], []

    for sub in subjects:
        data = load_feature_data(sub, cfg)

        # Split into train, val, test
        train_data, val_data, test_data, train_labels, val_labels, test_labels = preprocess_data(data, cfg)

        train_iter = Get_Dataloader(train_data, train_labels, True,  cfg["batch_size"])
        val_iter   = Get_Dataloader(val_data,   val_labels,   False, cfg["batch_size"])
        test_iter  = Get_Dataloader(test_data,  test_labels,  False, cfg["batch_size"])

        model = select_model(cfg)
        print(f"\n=== Training {sub} ===")

        # Train model once, monitor val loss
        top1_val, top5_val = train_and_eval(model, train_iter, val_iter, cfg)
        print(f"Validation (Block 6): Top-1={top1_val:.4f}, Top-5={top5_val:.4f}")

        # Evaluate *only*, no retraining on test
        model.eval()
        top1_all, top5_all = [], []
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(cfg["device"]), y.to(cfg["device"])
                y_hat = model(X)
                top1, top5 = topk_accuracy(y_hat, y, topk=(1, 5))
                top1_all.append(top1)
                top5_all.append(top5)

        top1_test = np.mean(top1_all)
        top5_test = np.mean(top5_all)
        print(f"Test (Block 7): Top-1={top1_test:.4f}, Top-5={top5_test:.4f}\n")

        all_top1.append(top1_test)
        all_top5.append(top5_test)


    # ==========================================
    # Final Results Summary
    # ==========================================
    print("=== Final Results ===")
    print(f"Mean Top-1: {np.mean(all_top1):.4f} ± {np.std(all_top1):.4f}")
    print(f"Mean Top-5: {np.mean(all_top5):.4f} ± {np.std(all_top5):.4f}")

    # ==========================================
    # Ranked Subject Performance Summary
    # ==========================================
    print("\n=== Ranked Subject Performance (Top-1 Accuracy) ===")
    subject_scores = list(zip(subjects, all_top1, all_top5))
    subject_scores.sort(key=lambda x: x[1], reverse=True)

    for rank, (sub, top1, top5) in enumerate(subject_scores, 1):
        print(f"{rank:2d}. {sub:15s} | Top-1: {top1:.4f} | Top-5: {top5:.4f}")

    best_sub, best_top1, best_top5 = subject_scores[0]
    print(f"\nBest Performing Subject: {best_sub} | Top-1: {best_top1:.4f} | Top-5: {best_top5:.4f}")


if __name__ == "__main__":
    main(CONFIG)
