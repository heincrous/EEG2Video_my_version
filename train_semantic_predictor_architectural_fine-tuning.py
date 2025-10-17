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
    "batch_size": 128,
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
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=0.0)
    if opt == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    raise ValueError("Unsupported optimizer type.")


def build_scheduler(optimizer, cfg):
    sched = cfg["scheduler"].lower()
    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
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

    # === Label construction ===
    num_classes = len(cfg["class_subset"])
    num_blocks_in_split = eeg_flat.shape[0] // (num_classes * 5)
    samples_per_class = num_blocks_in_split * 5  # 5 clips per class per block
    labels = np.repeat(np.arange(num_classes), samples_per_class)

    # === Normalise embeddings ===
    preds_norm = preds / (np.linalg.norm(preds, axis=1, keepdims=True) + 1e-8)
    gt_norm = clip_flat / (np.linalg.norm(clip_flat, axis=1, keepdims=True) + 1e-8)

    # === (1) MSE already computed ===
    # === (2) Cosine(pred, gt) ===
    avg_cosine = np.mean(np.sum(preds_norm * gt_norm, axis=1))

    # === Compute class means ===
    class_means_gt = np.zeros((num_classes, gt_norm.shape[1]))
    class_means_pred = np.zeros((num_classes, preds_norm.shape[1]))
    for c in range(num_classes):
        class_means_gt[c] = gt_norm[labels == c].mean(axis=0)
        class_means_gt[c] /= np.linalg.norm(class_means_gt[c]) + 1e-8
        class_means_pred[c] = preds_norm[labels == c].mean(axis=0)
        class_means_pred[c] /= np.linalg.norm(class_means_pred[c]) + 1e-8

    # === (3) Within-class cosine ===
    within_class_scores = []
    for c in range(num_classes):
        class_preds = preds_norm[labels == c]
        if len(class_preds) > 1:
            cos_mat = np.dot(class_preds, class_preds.T)
            n = len(class_preds)
            mean_cos = (np.sum(cos_mat) - np.trace(cos_mat)) / (n * (n - 1))
            within_class_scores.append(mean_cos)
    avg_within = np.mean(within_class_scores)

    # === (4) Between-class cosine ===
    between_scores = []
    for c in range(num_classes):
        others = np.delete(np.arange(num_classes), c)
        mean_cos = np.mean(np.dot(class_means_pred[c], class_means_pred[others].T))
        between_scores.append(mean_cos)
    avg_between = np.mean(between_scores)

    # === (5) Fisher-style separability ===
    global_mean = np.mean(class_means_pred, axis=0)
    sb = np.sum([np.sum((class_means_pred[c] - global_mean) ** 2) for c in range(num_classes)])
    sw = np.sum([
        np.sum((preds_norm[labels == c] - class_means_pred[c]) ** 2)
        for c in range(num_classes)
    ])
    fisher = sb / (sw + 1e-8)

    # === (6) Accuracy: per-sample nearest ground-truth class mean ===
    # Compute similarities of each pred to all gt class means
    sims_to_gt = np.dot(preds_norm, class_means_gt.T)
    pred_class = np.argmax(sims_to_gt, axis=1)
    acc = np.mean(pred_class == labels)

    # === Print ===
    print(
        f"MSE: {mse_loss:.6f} | Cos(pred,gt): {avg_cosine:.4f} | "
        f"Within: {avg_within:.4f} | Between: {avg_between:.4f} | "
        f"Δ={avg_within - avg_between:.4f} | Fisher={fisher:.4f} | "
        f"Semantic Acc={acc*100:.2f}%"
    )

    return mse_loss, avg_cosine, avg_within, avg_between, fisher, acc


# ==========================================
# Training Loop
# ==========================================
def train_model(model, train_loader, val_eeg, val_clip, cfg):
    device = cfg["device"]
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    for epoch in tqdm(range(1, cfg["epochs"] + 1)):
        model.train()
        epoch_loss = 0
        for eeg, clip in train_loader:
            eeg, clip = eeg.float().to(device), clip.float().to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(eeg), clip)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0:
            print(f"\n[Epoch {epoch}/{cfg['epochs']}] AvgLoss={epoch_loss/len(train_loader):.6f}")
            evaluate_model(model, val_eeg, val_clip, cfg)


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
# Main
# ==========================================
def main():
    cfg = CONFIG
    eeg, clip = load_de_data(cfg)
    tr_eeg, va_eeg, te_eeg, tr_clip, va_clip, te_clip = prepare_data(eeg, clip, cfg)
    dataset = EEGTextDataset(tr_eeg, tr_clip)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
    model = CLIPSemanticMLP(input_dim=tr_eeg.shape[1], cfg=cfg).to(cfg["device"])
    train_model(model, loader, va_eeg, va_clip, cfg)
    print("\nFinal Test Evaluation:")
    metrics = evaluate_model(model, te_eeg, te_clip, cfg)
    save_results(cfg, metrics, EXPERIMENT_TYPE, EXPERIMENT_MODE)


if __name__ == "__main__":
    main()
