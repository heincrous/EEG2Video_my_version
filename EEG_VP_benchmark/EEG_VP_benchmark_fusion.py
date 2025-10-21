# ==========================================
# EEG FUSION CLASSIFICATION (SEGMENT + DE + PSD)
# ==========================================
# Input: Preprocessed EEG feature types (segment, DE, PSD)
# Process: Encode each feature stream with GLFNet / GLFNet-MLP and fuse via Attention
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
EXPERIMENT_MODE = "fusion"
RESULT_ROOT = "/content/drive/MyDrive/EEG2Video_results/EEG_VP_benchmark/fusion"


# ==========================================
# Configuration Table
# ==========================================
CONFIG = {
    "feature_types": ["de", "psd", "segment"],
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
    "emb_dim": 64,
    "fusion_type": "attention",
    "batch_size": 512,
    "num_epochs": 100,
    "lr": 0.0005,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "result_root": RESULT_ROOT,
    "dry_run": False,
}


# ==========================================
# Data Handling
# ==========================================
def create_labels(num_classes, clips_per_class, blocks=1):
    block_labels = np.repeat(np.arange(num_classes), clips_per_class)
    return np.tile(block_labels, blocks)


def load_feature_data(sub_file, cfg):
    feature_data = {}
    for ft in cfg["feature_types"]:
        if ft == "segment":
            path = os.path.join(cfg["data_root"], cfg["segment_dir"], sub_file)
        elif ft == "de":
            path = os.path.join(cfg["data_root"], cfg["de_dir"], sub_file)
        elif ft == "psd":
            path = os.path.join(cfg["data_root"], cfg["psd_dir"], sub_file)
        else:
            raise ValueError(f"Unsupported feature type: {ft}")
        data = np.load(path)
        print(f"Loaded {sub_file} [{ft}] | shape {data.shape}")
        feature_data[ft] = data
    return feature_data


def preprocess_data(feature_data, cfg):
    """Preprocess each EEG feature type independently but split identically."""
    processed, labels_dict = {}, None

    for ft, data in feature_data.items():
        if ft == "segment":
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

        if labels_dict is None:
            labels_dict = {
                "train": create_labels(cfg["num_classes"], cfg["clips_per_class"], 5),
                "val":   create_labels(cfg["num_classes"], cfg["clips_per_class"]),
                "test":  create_labels(cfg["num_classes"], cfg["clips_per_class"]),
            }

        processed[ft] = {
            "train": train_data.reshape(-1, f, g),
            "val":   val_data.reshape(-1, f, g),
            "test":  test_data.reshape(-1, f, g),
        }

    return processed, labels_dict


def Get_Dataloader(feature_dict, labels, istrain, batch_size):
    tensors = [torch.tensor(v, dtype=torch.float32) for v in feature_dict.values()]
    labels = torch.tensor(labels, dtype=torch.long)
    dataset = data.TensorDataset(*tensors, labels)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=istrain)


# ==========================================
# Model Definition
# ==========================================
from models.glfnet import glfnet
from models.glfnet_mlp import glfnet_mlp


class AttentionFusion(nn.Module):
    def __init__(self, emb_dim, num_classes):
        super().__init__()
        self.query = nn.Linear(emb_dim, emb_dim)
        self.key   = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)
        self.softmax = nn.Softmax(dim=1)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, features):
        Q, K, V = self.query(features), self.key(features), self.value(features)
        attn = self.softmax(torch.bmm(Q, K.transpose(1, 2)) / (features.size(-1) ** 0.5))
        fused = torch.bmm(attn, V).mean(dim=1)
        return self.classifier(fused)


def build_fusion_model(cfg):
    emb_dim, device = cfg["emb_dim"], cfg["device"]
    encoders = {}

    if "segment" in cfg["feature_types"]:
        encoders["segment"] = glfnet(out_dim=emb_dim, emb_dim=emb_dim,
                                     C=cfg["channels"], T=cfg["time_len"]).to(device)
    if "de" in cfg["feature_types"]:
        encoders["de"] = glfnet_mlp(out_dim=emb_dim, emb_dim=emb_dim, input_dim=310).to(device)
    if "psd" in cfg["feature_types"]:
        encoders["psd"] = glfnet_mlp(out_dim=emb_dim, emb_dim=emb_dim, input_dim=310).to(device)

    fusion = AttentionFusion(emb_dim, cfg["num_classes"]).to(device)
    return encoders, fusion


# ==========================================
# Dry Run Validation
# ==========================================
def simulate_dummy_features(cfg):
    """Create synthetic EEG tensors for dry-run validation."""
    C, T = cfg["channels"], cfg["time_len"]
    device = cfg["device"]
    dummy = {}
    for ft in cfg["feature_types"]:
        if ft == "segment":
            flat = torch.randn(1400, 1, C, T)
        else:
            flat = torch.randn(1400, C, 5)
        flat = (flat - flat.mean()) / (flat.std() + 1e-6)
        dummy[ft] = flat.to(device)
    return dummy


def dry_run_fusion(cfg):
    print("\n=== Fusion Model Dry Run ===")
    encoders, fusion = build_fusion_model(cfg)
    dummy_inputs = simulate_dummy_features(cfg)
    features = []
    for ft, encoder in encoders.items():
        x = dummy_inputs[ft]
        with torch.no_grad():
            out = encoder(x)
        print(f"{ft:8s} encoder → output {list(out.shape)}")
        features.append(out)
    tokens = torch.stack(features, dim=1)
    with torch.no_grad():
        y_hat = fusion(tokens)
    print(f"Fusion output: {list(y_hat.shape)}\n=== Dry Run Complete ===\n")


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


def train_and_eval(encoders, fusion, train_iter, test_iter, cfg):
    device = cfg["device"]
    params = list(fusion.parameters())
    for e in encoders.values():
        params += list(e.parameters())

    optimiser = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in tqdm(range(1, cfg["num_epochs"] + 1)):
        total_loss, total_acc, total_count = 0.0, 0.0, 0
        fusion.train()
        for e in encoders.values():
            e.train()

        for batch in train_iter:
            *Xs, y = batch
            y = y.to(device)
            features = []
            for i, ft in enumerate(cfg["feature_types"]):
                X = Xs[i].to(device)
                if ft == "segment":
                    X = X.unsqueeze(1)
                features.append(encoders[ft](X))
            tokens = torch.stack(features, dim=1)
            y_hat = fusion(tokens)

            loss = loss_fn(y_hat, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item() * y.size(0)
            total_acc += (y_hat.argmax(1) == y).sum().item()
            total_count += y.size(0)

        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] Loss={total_loss/total_count:.4f} | Acc={total_acc/total_count:.4f}")

    fusion.eval()
    for e in encoders.values():
        e.eval()

    top1_all, top5_all = [], []
    with torch.no_grad():
        for batch in test_iter:
            *Xs, y = batch
            y = y.to(device)
            features = []
            for i, ft in enumerate(cfg["feature_types"]):
                X = Xs[i].to(device)
                if ft == "segment":
                    X = X.unsqueeze(1)
                features.append(encoders[ft](X))
            tokens = torch.stack(features, dim=1)
            y_hat = fusion(tokens)
            top1, top5 = topk_accuracy(y_hat, y)
            top1_all.append(top1)
            top5_all.append(top5)

    return np.mean(top1_all), np.mean(top5_all)


# ==========================================
# Save Results
# ==========================================
def save_results(cfg, subjects, all_top1, all_top5):
    mean_top1, std_top1 = np.mean(all_top1), np.std(all_top1)
    mean_top5, std_top5 = np.mean(all_top5), np.std(all_top5)
    exp_dir = cfg["result_root"]
    os.makedirs(exp_dir, exist_ok=True)

    fusion_name = "_".join(cfg["feature_types"])
    filename = (
        f"fusion_{fusion_name}_emb{cfg['emb_dim']}_"
        f"lr{cfg['lr']}_bs{cfg['batch_size']}_ep{cfg['num_epochs']}.txt"
    )
    save_path = os.path.join(exp_dir, filename)

    with open(save_path, "w") as f:
        f.write("EEG Multi-Feature Fusion Classification Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write("Configuration:\n")
        for k, v in cfg.items():
            f.write(f"{k}: {v}\n")

        f.write("\nFinal Metrics:\n")
        f.write(f"Mean Top-1: {mean_top1:.4f} ± {std_top1:.4f}\n")
        f.write(f"Mean Top-5: {mean_top5:.4f} ± {std_top5:.4f}\n\n")

        f.write("Per-Subject Results:\n")
        for s, t1, t5 in zip(subjects, all_top1, all_top5):
            f.write(f"{s:15s} | Top-1: {t1:.4f} | Top-5: {t5:.4f}\n")

    print(f"[Saved Results] {save_path}")


# ==========================================
# Cleanup Utilities
# ==========================================
def clean_old_results(cfg):
    exp_dir = cfg["result_root"]
    if not os.path.exists(exp_dir):
        return
    deleted = []
    for f in os.listdir(exp_dir):
        if f.endswith(".txt"):
            os.remove(os.path.join(exp_dir, f))
            deleted.append(f)
    if deleted:
        print(f"[Cleanup] Removed old result files: {deleted}")
    else:
        print("[Cleanup] No previous result files found.")


# ==========================================
# Main
# ==========================================
def main(cfg):
    if cfg["dry_run"]:
        dry_run_fusion(cfg)
        return

    clean_old_results(cfg)
    subjects = cfg["subjects_to_train"]
    print(f"\nTraining subjects: {subjects}\n")

    all_top1, all_top5 = [], []

    for sub in subjects:
        print(f"\n=== Subject: {sub} ===")
        feature_data = load_feature_data(sub, cfg)
        processed, labels_dict = preprocess_data(feature_data, cfg)

        train_iter = Get_Dataloader(
            {ft: processed[ft]["train"] for ft in cfg["feature_types"]},
            labels_dict["train"], True, cfg["batch_size"]
        )
        test_iter = Get_Dataloader(
            {ft: processed[ft]["test"] for ft in cfg["feature_types"]},
            labels_dict["test"], False, cfg["batch_size"]
        )

        encoders, fusion = build_fusion_model(cfg)
        top1, top5 = train_and_eval(encoders, fusion, train_iter, test_iter, cfg)
        print(f"[Test Block-7] Top-1={top1:.4f} | Top-5={top5:.4f}")
        all_top1.append(top1)
        all_top5.append(top5)

    print("\n=== Final Results ===")
    print(f"Mean Top-1: {np.mean(all_top1):.4f} ± {np.std(all_top1):.4f}")
    print(f"Mean Top-5: {np.mean(all_top5):.4f} ± {np.std(all_top5):.4f}")

    save_results(cfg, subjects, all_top1, all_top5)


if __name__ == "__main__":
    main(CONFIG)
