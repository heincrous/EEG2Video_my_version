# ==========================================
# EEG → CLIP Semantic Predictor (DE-Only, MSE Loss)
# Modular Version
# ==========================================

# === Standard libraries ===
import os
import numpy as np

# === Third-party libraries ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from einops import rearrange
from tqdm import tqdm


# ==========================================
# Config Block
# ==========================================
"""
DEFAULT MODEL

Architecture:
Layer width: [10000, 10000, 10000, 10000]
Dropout: 0.0
Activation: ReLU
Normalization: None

Optimisation:
Optimiser: Adam
Weight decay: 0.0
Scheduler: cosine
Learning rate: 0.0005

Adam
No weight decay
No dropout
ReLU activation

Training:
Epochs: 200
Batch size: 32
Loss: MSE
*Optimise values based on default model
"""

EXPERIMENT_MODE = "architectural"  # or "optimisation"
EXPERIMENT_TYPE = "activation" if EXPERIMENT_MODE == "architectural" else "scheduler"

if EXPERIMENT_MODE == "architectural":
    RESULT_ROOT = "/content/drive/MyDrive/EEG2Video_results/semantic_predictor/architectural_fine-tuning"
else:
    RESULT_ROOT = "/content/drive/MyDrive/EEG2Video_results/semantic_predictor/optimisation"

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
    "batch_size": 32,
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
# Model Definition
# ==========================================
class CLIPSemanticMLP(nn.Module):
    def __init__(self, input_dim, cfg):
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


# ==========================================
# Dataset
# ==========================================
class EEGTextDataset(Dataset):
    def __init__(self, eeg, text):
        self.eeg = eeg
        self.text = text

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        return self.eeg[idx], self.text[idx]


# ==========================================
# Data Handling
# ==========================================
def load_de_data(cfg):
    eeg_path = os.path.join(cfg["eeg_root"], cfg["feature_type"], cfg["subject_name"])
    clip_path = cfg["clip_path"]

    eeg = np.load(eeg_path, allow_pickle=True)
    clip = np.load(clip_path, allow_pickle=True)

    # Expect 6D: [7, 40, 5, 2, 62, 5]
    if eeg.ndim != 6:
        raise ValueError(f"Expected 6D EEG array [7,40,5,2,62,5], got {eeg.shape}")

    # Average across trials (axis=3)
    eeg = eeg.mean(axis=3)  # -> [7, 40, 5, 62, 5]

    print(f"Loaded EEG {cfg['subject_name']} shape: {eeg.shape}")
    print(f"Loaded CLIP shape: {clip.shape}")
    return eeg, clip


def prepare_data(eeg, clip, cfg):
    # Apply subset of classes
    eeg, clip = eeg[:, cfg["class_subset"]], clip[:, cfg["class_subset"]]

    # Split into blocks: 5 train, 1 val, 1 test
    train_eeg, val_eeg, test_eeg = eeg[:5], eeg[5:6], eeg[6:]
    train_clip, val_clip, test_clip = clip[:5], clip[5:6], clip[6:]

    # Flatten EEG and CLIP correctly
    # EEG: [b, c, s, ch, f] -> [(b c s), (ch f)]
    # CLIP: [b, c, s, tok, dim] -> [(b c s), (tok dim)]
    flatten_eeg = lambda x: rearrange(x, "b c s ch f -> (b c s) (ch f)")
    flatten_clip = lambda x: rearrange(x, "b c s tok dim -> (b c s) (tok dim)")

    train_eeg, val_eeg, test_eeg = map(flatten_eeg, [train_eeg, val_eeg, test_eeg])
    train_clip, val_clip, test_clip = map(flatten_clip, [train_clip, val_clip, test_clip])

    # Fit scaler on all EEG before splitting
    scaler = StandardScaler()
    scaler.fit(eeg.reshape(-1, eeg.shape[-2] * eeg.shape[-1]))
    train_eeg = scaler.transform(train_eeg)
    val_eeg = scaler.transform(val_eeg)
    test_eeg = scaler.transform(test_eeg)

    print(f"[Scaler] mean={np.mean(train_eeg):.5f}, std={np.std(train_eeg):.5f}")
    print(f"Train EEG shape: {train_eeg.shape}, Train CLIP shape: {train_clip.shape}")
    return train_eeg, val_eeg, test_eeg, train_clip, val_clip, test_clip


# ==========================================
# Optimiser and Scheduler
# ==========================================
def build_optimizer(model, cfg):
    opt = cfg["optimizer"].lower()
    if opt == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if opt == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    raise ValueError("Unsupported optimizer type.")


def build_scheduler(optimizer, cfg, steps_per_epoch):
    sched = cfg["scheduler"].lower()
    if sched == "cosine":
        total_steps = cfg["epochs"] * steps_per_epoch
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    if sched == "constant":
        return None
    raise ValueError("Unsupported scheduler type.")


# ==========================================
# Evaluation
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

    # === Basic label mapping ===
    num_classes = len(cfg["class_subset"])
    samples_per_class = eeg_flat.shape[0] // num_classes
    labels = np.repeat(np.arange(num_classes), samples_per_class)
    labels = labels[:eeg_flat.shape[0]]
    # print("labels:", labels.shape, "unique:", np.unique(labels, return_counts=True))

    # === Normalize embeddings (L2 per vector) ===
    preds_norm = preds / (np.linalg.norm(preds, axis=1, keepdims=True) + 1e-8)
    gt_norm = gt / (np.linalg.norm(gt, axis=1, keepdims=True) + 1e-8)

    # === (1) MSE already computed ===
    # === (2) Average cosine(pred, gt) ===
    cosines = np.sum(preds_norm * gt_norm, axis=1)
    avg_cosine = float(np.mean(cosines))

    # === (3.1) Compute per-class mean embeddings ===
    class_means_pred = np.zeros((num_classes, preds_norm.shape[1]))
    for c in range(num_classes):
        class_means_pred[c] = preds_norm[labels == c].mean(axis=0)
        class_means_pred[c] /= np.linalg.norm(class_means_pred[c]) + 1e-8

    # === (3.2) Compute ground-truth class means ===
    class_means_gt = np.zeros((num_classes, gt_norm.shape[1]))
    for c in range(num_classes):
        class_means_gt[c] = gt_norm[labels == c].mean(axis=0)
        class_means_gt[c] /= np.linalg.norm(class_means_gt[c]) + 1e-8

    # === (4) Within-class similarity (sample → its class mean) ===
    within_scores = []
    for i, p in enumerate(preds_norm):
        c = labels[i]
        within_scores.append(np.dot(p, class_means_pred[c]))
    avg_within = float(np.mean(within_scores))

    # === (5) Between-class similarity (sample → other class means) ===
    between_scores = []
    for i, p in enumerate(preds_norm):
        c = labels[i]
        other_means = class_means_pred[np.arange(num_classes) != c]
        sims = np.dot(other_means, p)
        between_scores.extend(sims)
    avg_between = float(np.mean(between_scores))

    # === (6) Fisher-style separability ===
    global_mean = class_means_pred.mean(axis=0)
    sb = np.sum([np.sum((m - global_mean) ** 2) for m in class_means_pred])
    sw = np.sum([
        np.sum((preds_norm[labels == c] - class_means_pred[c]) ** 2)
        for c in range(num_classes)
    ])
    fisher = sb / (sw + 1e-8)

    # === (7) Classification accuracy (nearest class mean) ===
    sims_to_means = np.dot(preds_norm, class_means_gt.T)
    pred_class = np.argmax(sims_to_means, axis=1)
    acc = np.mean(pred_class == labels)

    # === Print summary ===
    print(
        f"MSE: {mse_loss:.6f} | Cos(pred,gt): {avg_cosine:.4f} | "
        f"Within: {avg_within:.4f} | Between: {avg_between:.4f} | "
        f"Δ={avg_within - avg_between:.4f} | Fisher={fisher:.4f} | "
        f"Semantic Acc={acc*100:.2f}%"
    )

    return mse_loss, avg_cosine, avg_within, avg_between, fisher, acc


# ==========================================
# Plot Training Metrics (Cosine & Accuracy)
# ==========================================
import os
import matplotlib.pyplot as plt

def save_training_plots(cosine_list, acc_list, cfg):
    plots_dir = os.path.join(cfg["result_root"], "plots")
    os.makedirs(plots_dir, exist_ok=True)

    epochs = list(range(1, len(cosine_list) + 1))
    batch_size = cfg["batch_size"]

    # Plot 1: Cosine Similarity
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, cosine_list, marker="o", color="blue")
    plt.title(f"Cosine Similarity vs Epoch (Batch {batch_size})")
    plt.xlabel("Epoch")
    plt.ylabel("Cosine Similarity")
    plt.grid(True)
    plt.tight_layout()
    cos_path = os.path.join(plots_dir, f"cosine_batch{batch_size}.png")
    plt.savefig(cos_path, dpi=300)
    plt.close()

    # Plot 2: Accuracy
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, [a * 100 for a in acc_list], marker="o", color="green")
    plt.title(f"Semantic Accuracy vs Epoch (Batch {batch_size})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.tight_layout()
    acc_path = os.path.join(plots_dir, f"accuracy_batch{batch_size}.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()

    print(f"[Plot Saved] → {cos_path}")
    print(f"[Plot Saved] → {acc_path}")


# ==========================================
# Training Loop
# ==========================================
def train_model(model, train_loader, val_eeg, val_clip, cfg):
    device = cfg["device"]
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    cosine_history, acc_history = [], []

    for epoch in tqdm(range(1, cfg["epochs"] + 1)):
        model.train()
        epoch_loss = 0.0

        for eeg, clip in train_loader:
            eeg, clip = eeg.float().to(device), clip.float().to(device)
            optimizer.zero_grad()
            pred = model(eeg)
            loss = F.mse_loss(pred, clip)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"\n[Epoch {epoch}/{cfg['epochs']}] AvgLoss={avg_loss:.6f}")
            _, avg_cosine, _, _, _, acc = evaluate_model(model, val_eeg, val_clip, cfg)
            cosine_history.append(avg_cosine)
            acc_history.append(acc)

    save_training_plots(cosine_history, acc_history, cfg)


# ==========================================
# Save Results
# ==========================================
def save_results(cfg, metrics, exp_type, exp_mode):
    mse, cos, within, between, fisher, acc = metrics

    # === Create results directory ===
    exp_dir = os.path.join(cfg["result_root"], exp_type)
    os.makedirs(exp_dir, exist_ok=True)

    # === File naming rules ===
    if exp_mode == "architectural":
        fname = (
            f"{exp_type}_lw{'-'.join(map(str, cfg['layer_widths']))}"
            f"_do{cfg['dropout']}_act{cfg['activation']}_reg{cfg['normalization']}.txt"
        )
    elif exp_mode == "optimisation":
        fname = (
            f"{exp_type}_opt{cfg['optimizer']}_wd{cfg['weight_decay']}"
            f"_sched{cfg['scheduler']}_lr{cfg['lr']}.txt"
        )
    else:
        raise ValueError("Unknown experiment mode: must be 'architectural' or 'optimisation'.")

    save_path = os.path.join(exp_dir, fname)

    # === Write configuration and metrics ===
    with open(save_path, "w") as f:
        f.write(f"EEG→CLIP Semantic Predictor ({exp_mode.capitalize()})\n")
        f.write("=" * 60 + "\n\n")
        f.write("Configuration Used:\n")
        for k, v in cfg.items():
            f.write(f"{k}: {v}\n")

        f.write("\nFinal Evaluation Metrics:\n")
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"Cosine(pred, gt): {cos:.4f}\n")
        f.write(f"Within-Class Cosine: {within:.4f}\n")
        f.write(f"Between-Class Cosine: {between:.4f}\n")
        f.write(f"Δ (Within−Between): {within - between:.4f}\n")
        f.write(f"Fisher Score: {fisher:.4f}\n")
        f.write(f"Semantic Accuracy: {acc * 100:.2f}%\n")

    print(f"Saved results to: {save_path}")


# ==========================================
# Inference Module
# ==========================================
def run_inference_and_save(model, eeg_flat, cfg):
    device = cfg["device"]
    model.eval()

    with torch.no_grad():
        eeg_tensor = torch.tensor(eeg_flat, dtype=torch.float32, device=device)
        preds_tensor = model(eeg_tensor)
        preds = preds_tensor.cpu().numpy()

    # === Reshape to [num_classes, 5, 77, 768] ===
    num_classes = len(cfg["class_subset"])
    preds = preds.reshape(num_classes, 5, 77, 768)

    # === Save predictions ===
    pred_dir = os.path.join(
        "/content/drive/MyDrive/EEG2Video_results/semantic_predictor/predictions"
    )
    os.makedirs(pred_dir, exist_ok=True)

    subset_name = "_".join(map(str, cfg["class_subset"]))
    save_path = os.path.join(pred_dir, f"{subset_name}.npy")
    np.save(save_path, preds)

    print(f"[Inference] Saved predictions → {save_path}")
    print(f"[Inference] Shape: {preds.shape}")
    return preds


# ==========================================
# Cleanup Module
# ==========================================
def clean_old_predictions(cfg):
    pred_dir = "/content/drive/MyDrive/EEG2Video_results/semantic_predictor/predictions"
    os.makedirs(pred_dir, exist_ok=True)

    subset_name = "_".join(map(str, cfg["class_subset"]))
    target_pattern = f"{subset_name}.npy"

    deleted_files = []
    for f in os.listdir(pred_dir):
        if f == target_pattern:
            file_path = os.path.join(pred_dir, f)
            os.remove(file_path)
            deleted_files.append(file_path)

    if deleted_files:
        print(f"[Cleanup] Removed old prediction file(s): {deleted_files}")
    else:
        print(f"[Cleanup] No existing prediction file for subset {subset_name}.")


# ==========================================
# Main
# ==========================================
def main():
    cfg = CONFIG
    clean_old_predictions(cfg)
    eeg, clip = load_de_data(cfg)
    tr_eeg, va_eeg, te_eeg, tr_clip, va_clip, te_clip = prepare_data(eeg, clip, cfg)
    dataset = EEGTextDataset(tr_eeg, tr_clip)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
    model = CLIPSemanticMLP(input_dim=tr_eeg.shape[1], cfg=cfg).to(cfg["device"])
    train_model(model, loader, va_eeg, va_clip, cfg)
    print("\nFinal Test Evaluation:")
    metrics = evaluate_model(model, te_eeg, te_clip, cfg)
    save_results(cfg, metrics, EXPERIMENT_TYPE, EXPERIMENT_MODE)
    run_inference_and_save(model, te_eeg, cfg)


if __name__ == "__main__":
    main()
