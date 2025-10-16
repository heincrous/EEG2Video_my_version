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
import importlib
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils import data
from einops import rearrange


# === Encoder Imports ===
from models.glfnet import glfnet
from models.glfnet_mlp import glfnet_mlp


# ==========================================
# 1. Configuration Table
# ==========================================
CONFIG = {
    # --- Core setup ---
    "feature_types": ["segment", "de", "psd"],   # choose any subset: ["segment"], ["de", "psd"], ["segment", "de"], etc.
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
    "dry_run": True,

    # --- Data parameters ---
    "num_classes": 40,
    "channels": 62,
    "time_len": 400,
    "num_blocks": 7,
    "clips_per_class": 5,

    # --- Paths ---
    "data_root": "/content/drive/MyDrive/EEG2Video_data/processed/",
    "segment_dir": "EEG_segments/",
    "de_dir": "EEG_DE_1per1s/",
    "psd_dir": "EEG_PSD_1per1s/",

    # --- Model parameters ---
    "emb_dim": 64,                # latent embedding size used by both GLFNet and GLFNet-MLP
    "fusion_type": "attention",   # "concat" or "attention"

    # --- Training parameters ---
    "batch_size": 512,
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
    """
    Preprocess multiple EEG feature types (segment, DE, PSD).
    Each feature is standardized independently but split identically.
    Returns dict of processed arrays and shared labels.
    """
    processed = {}
    labels_dict = None

    for ft, data in feature_data.items():
        if ft == "segment":
            b, c, d, f, g = data.shape  # (7, 40, 5, 62, 400)
        else:
            data = data.mean(axis=3)     # for DE/PSD -> (7, 40, 5, 62, 5)
            b, c, d, f, g = data.shape

        # reshape to (blocks, samples_per_block, channels, features)
        data = data.reshape(b, c * d, f, g)

        # global normalization
        scaler = StandardScaler()
        flat_all = data.reshape(-1, g)
        scaler.fit(flat_all)
        scaled = scaler.transform(flat_all).reshape(b, c * d, f, g)

        # split by blocks (5 train, 1 val, 1 test)
        train_data = scaled[:5]
        val_data   = scaled[5:6]
        test_data  = scaled[6:7]

        # create labels once
        if labels_dict is None:
            labels_dict = {
                "train": create_labels(cfg["num_classes"], cfg["clips_per_class"], 5),
                "val":   create_labels(cfg["num_classes"], cfg["clips_per_class"]),
                "test":  create_labels(cfg["num_classes"], cfg["clips_per_class"]),
            }

        # flatten blocks into individual samples
        processed[ft] = {
            "train": train_data.reshape(-1, f, g),
            "val":   val_data.reshape(-1, f, g),
            "test":  test_data.reshape(-1, f, g),
        }

    return processed, labels_dict


def Get_Dataloader(feature_dict, labels, istrain, batch_size):
    """
    Combines multiple feature types into one PyTorch dataloader.
    Each batch yields (x_segment, x_de, x_psd, labels)
    depending on which feature types are present.
    """
    tensors = [torch.tensor(v, dtype=torch.float32) for v in feature_dict.values()]
    labels = torch.tensor(labels, dtype=torch.long)
    dataset = data.TensorDataset(*tensors, labels)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=istrain)


# ==========================================
# 3. Model Selection Module
# ==========================================
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
        # features: [B, N, D]
        Q, K, V = self.query(features), self.key(features), self.value(features)
        attn = self.softmax(torch.bmm(Q, K.transpose(1, 2)) / (features.size(-1) ** 0.5))
        fused = torch.bmm(attn, V).mean(dim=1)  # [B, D]
        return self.classifier(fused)


def build_fusion_model(cfg):
    """
    Build modality-specific encoders and an attention-based fusion head.
    Returns dict of encoders and the fusion module.
    """
    emb_dim = cfg["emb_dim"]
    device = cfg["device"]

    encoders = {}

    if "segment" in cfg["feature_types"]:
        encoders["segment"] = glfnet(
            out_dim=emb_dim, emb_dim=emb_dim,
            C=cfg["channels"], T=cfg["time_len"]
        ).to(device)

    if "de" in cfg["feature_types"]:
        encoders["de"] = glfnet_mlp(
            out_dim=emb_dim, emb_dim=emb_dim, input_dim=310
        ).to(device)

    if "psd" in cfg["feature_types"]:
        encoders["psd"] = glfnet_mlp(
            out_dim=emb_dim, emb_dim=emb_dim, input_dim=310
        ).to(device)

    fusion = AttentionFusion(emb_dim, cfg["num_classes"]).to(device)
    return encoders, fusion


# ==========================================
# 4. Functional Dry Run Module (Fixed + Filtered)
# ==========================================
def simulate_dummy_features(cfg):
    """
    Create synthetic EEG tensors for each feature type in cfg["feature_types"].
    Shapes mimic preprocessed data for quick dry-run validation.
    Returns dict of dummy tensors keyed by feature name.
    """
    C, T = cfg["channels"], cfg["time_len"]
    device = cfg["device"]
    dummy = {}

    for ft in cfg["feature_types"]:
        if ft == "segment":
            # Simulated segment EEG: (samples, 1, 62, 400)
            flat = torch.randn(1400, 1, C, T)
        else:
            # Simulated DE/PSD EEG: (samples, 62, 5)
            flat = torch.randn(1400, C, 5)
        flat = (flat - flat.mean()) / (flat.std() + 1e-6)
        dummy[ft] = flat.to(device)

    return dummy


def dry_run_fusion(cfg):
    """
    Dry run: forward pass through all encoders and attention fusion head
    with synthetic data for each feature type.
    Confirms compatibility and output dimensions.
    """
    print("\n=== Running Fusion Dry Run (GLFNet + GLFNet-MLP + Attention) ===")
    device = cfg["device"]
    emb_dim = cfg["emb_dim"]
    num_classes = cfg["num_classes"]

    # build models
    encoders, fusion = build_fusion_model(cfg)
    dummy_inputs = simulate_dummy_features(cfg)

    features = []
    for ft, encoder in encoders.items():
        x = dummy_inputs[ft]
        with torch.no_grad():
            out = encoder(x)
        print(f"{ft:8s} encoder → output {list(out.shape)}")
        features.append(out)

    tokens = torch.stack(features, dim=1)  # [B, N, D]
    with torch.no_grad():
        y_hat = fusion(tokens)
    print(f"Fusion output shape: {list(y_hat.shape)} (expected [B, {num_classes}])")

    print("=== Fusion Dry Run Complete ===\n")


# ==========================================
# 5. Training and Evaluation for Fusion
# ==========================================
def cal_accuracy(y_hat, y):
    if y_hat.ndim > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, dim=1)
    return (y_hat == y).sum()


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
    """
    Train all encoders + fusion head jointly on multi-feature batches.
    Each dataloader batch yields (x_segment, x_de, x_psd, labels).
    """
    device = cfg["device"]
    params = list(fusion.parameters())
    for enc in encoders.values():
        params += list(enc.parameters())

    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0)
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = cfg["num_epochs"]

    for epoch in range(num_epochs):
        total_loss, total_acc, total_count = 0.0, 0.0, 0
        fusion.train()
        for e in encoders.values():
            e.train()

        for batch in train_iter:
            *Xs, y = batch
            y = y.to(device)
            features = []

            # forward through each encoder
            for i, ft in enumerate(cfg["feature_types"]):
                X = Xs[i].to(device)
                if ft == "segment":
                    X = X.unsqueeze(1)  # (B,1,62,400)
                features.append(encoders[ft](X))

            tokens = torch.stack(features, dim=1)  # [B, N, D]
            y_hat = fusion(tokens)

            loss = loss_fn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_acc += cal_accuracy(y_hat, y).item()
            total_count += y.size(0)

        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch+1}] Loss {total_loss/total_count:.4f} | Acc {total_acc/total_count:.4f}")

    # --- Evaluation ---
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
# 6. Main Execution
# ==========================================
def main(cfg):
    # --- Dry run ---
    if cfg["dry_run"]:
        dry_run_fusion(cfg)
        return

    # --- Subject list ---
    data_dir = os.path.join(cfg["data_root"], cfg["segment_dir"])  # base path (segment always exists)
    subjects = os.listdir(data_dir)
    if cfg["subjects_to_train"] != "all":
        subjects = [s for s in subjects if s in cfg["subjects_to_train"]]

    print(f"\nTraining subjects: {subjects}\n")
    all_top1, all_top5 = [], []

    # --- Loop over subjects ---
    for sub in subjects:
        print(f"\n=== Subject: {sub} ===")

        # Load all features (segment, de, psd)
        feature_data = load_feature_data(sub, cfg)

        # Preprocess and split into train/val/test
        processed, labels_dict = preprocess_data(feature_data, cfg)

        # Build dataloaders
        train_iter = Get_Dataloader({ft: processed[ft]["train"] for ft in cfg["feature_types"]},
                                    labels_dict["train"], True, cfg["batch_size"])
        test_iter  = Get_Dataloader({ft: processed[ft]["test"]  for ft in cfg["feature_types"]},
                                    labels_dict["test"],  False, cfg["batch_size"])

        # Build encoders + fusion model
        encoders, fusion = build_fusion_model(cfg)

        # Train and evaluate
        top1, top5 = train_and_eval(encoders, fusion, train_iter, test_iter, cfg)
        print(f"Test (Block 7): Top-1={top1:.4f}, Top-5={top5:.4f}")
        all_top1.append(top1)
        all_top5.append(top5)

    # --- Final Summary ---
    mean_top1, std_top1 = np.mean(all_top1), np.std(all_top1)
    mean_top5, std_top5 = np.mean(all_top5), np.std(all_top5)

    print("\n=== Final Results ===")
    print(f"Mean Top-1: {mean_top1:.4f} ± {std_top1:.4f}")
    print(f"Mean Top-5: {mean_top5:.4f} ± {std_top5:.4f}")

    # --- Save results ---
    base_dir = "/content/drive/MyDrive/EEG2Video_results/EEG_VP_benchmark"
    fusion_name = "_".join(cfg["feature_types"])
    feature_dir = os.path.join(base_dir, fusion_name)
    os.makedirs(feature_dir, exist_ok=True)

    filename = (
        f"fusion_{fusion_name}_"
        f"emb{cfg['emb_dim']}_"
        f"lr{cfg['lr']}_"
        f"bs{cfg['batch_size']}_"
        f"ep{cfg['num_epochs']}.txt"
    )
    save_path = os.path.join(feature_dir, filename)

    with open(save_path, "w") as f:
        f.write("EEG Fusion Classification Summary\n")
        f.write("==========================================\n\n")

        f.write("Configuration:\n")
        f.write(f"Features Used: {', '.join(cfg['feature_types'])}\n")
        f.write(f"Subjects: {', '.join(cfg['subjects_to_train'])}\n")
        f.write(f"Embedding Dim: {cfg['emb_dim']}\n")
        f.write(f"Learning Rate: {cfg['lr']}\n")
        f.write(f"Batch Size: {cfg['batch_size']}\n")
        f.write(f"Epochs: {cfg['num_epochs']}\n\n")

        f.write("Results:\n")
        f.write(f"Mean Top-1 Accuracy: {mean_top1:.4f} ± {std_top1:.4f}\n")
        f.write(f"Mean Top-5 Accuracy: {mean_top5:.4f} ± {std_top5:.4f}\n\n")

        f.write("Per-Subject:\n")
        for sub, t1, t5 in zip(subjects, all_top1, all_top5):
            f.write(f"{sub:15s} | Top-1: {t1:.4f} | Top-5: {t5:.4f}\n")

    print(f"\nSaved results to: {save_path}")


if __name__ == "__main__":
    main(CONFIG)
