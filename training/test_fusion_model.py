# ==========================================
# End-to-end FusionNet training with fixed 5/1/1 split + feature-wise normalisation
# ==========================================

# === Standard libraries ===
import os

# === Third-party libraries ===
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from sklearn.preprocessing import StandardScaler

# === Repo imports ===
from core.fusion_model import FusionNet


# ==========================================
# Config table
# ==========================================
CONFIG = {
    "batch_size":     32,
    "num_epochs":     100,
    "lr":             0.0005,
    "weight_decay":   0.05,
    "optimizer":      "AdamW",
    "C":              62,       # EEG channels
    "T":              200,      # 200 samples per 1s window (200 Hz)
    "emb_dim":        128,
    "de_dim":         310,      # 62*5
    "psd_dim":        310,      # 62*5
    "out_dim":        40,       # 40 classes
    "device":         "cuda",

    # Encoders (configurable)
    "raw_model":      "glfnet",      # options: shallownet, deepnet, eegnet, tsconv, conformer, glfnet
    "de_model":       "glfnet_mlp",      # options: mlpnet, glfnet_mlp
    "psd_model":      "glfnet_mlp",      # options: mlpnet, glfnet_mlp

    # Data paths
    "segments_dir": "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments",
    "de_dir":       "/content/drive/MyDrive/EEG2Video_data/processed/EEG_DE_1per1s",
    "psd_dir":      "/content/drive/MyDrive/EEG2Video_data/processed/EEG_PSD_1per1s",
    "subj_id":      "sub3.npy"
}


# ==========================================
# Dataset wrapper
# ==========================================
class FusionDataset(data.Dataset):
    def __init__(self, raw, de, psd, labels):
        self.raw = raw
        self.de = de
        self.psd = psd
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        raw_sample = torch.tensor(self.raw[idx], dtype=torch.float32)
        de_sample  = torch.tensor(self.de[idx], dtype=torch.float32)
        psd_sample = torch.tensor(self.psd[idx], dtype=torch.float32)
        label      = torch.tensor(self.labels[idx], dtype=torch.long)
        return (raw_sample, de_sample, psd_sample), label


def get_dataloader(raw, de, psd, labels, is_train, batch_size):
    dataset = FusionDataset(raw, de, psd, labels)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)


# ==========================================
# Preprocessing
# ==========================================
def split_segments_to_windows(segments):
    # (7,40,5,62,400) -> (7,40,5,2,62,200)
    first = segments[..., :, :200]
    second = segments[..., :, 200:]
    return np.stack([first, second], axis=3)


def flatten_and_labels(raw, de, psd):
    # raw: (7,40,5,2,62,200), de/psd: (7,40,5,2,62,5)
    n_blocks, n_cls, n_clips, n_win = raw.shape[:4]
    N = n_blocks * n_cls * n_clips * n_win
    raw = raw.reshape(N, 1, CONFIG["C"], CONFIG["T"])
    de  = de.reshape(N, CONFIG["C"], 5)
    psd = psd.reshape(N, CONFIG["C"], 5)

    # Labels from block structure
    labels_block = np.repeat(np.arange(n_cls), n_clips * n_win)  # 40*10 = 400 per block
    labels = np.tile(labels_block, n_blocks)                     # 7*400 = 2800 total
    return raw, de, psd, labels


def scale_feature(train, val, test):
    scaler = StandardScaler()
    scaler.fit(train.reshape(len(train), -1))
    train = scaler.transform(train.reshape(len(train), -1)).reshape(train.shape)
    val   = scaler.transform(val.reshape(len(val), -1)).reshape(val.shape)
    test  = scaler.transform(test.reshape(len(test), -1)).reshape(test.shape)
    return train, val, test, scaler


# ==========================================
# Metrics
# ==========================================
def topk_accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / target.size(0)).item())
        return res


def evaluate(model, dataloader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    total_loss, total_samples = 0.0, 0
    top1_list, top5_list = [], []
    with torch.no_grad():
        for (raw, de, psd), y in dataloader:
            raw, de, psd, y = raw.to(device), de.to(device), psd.to(device), y.to(device)
            logits = model(raw, de, psd)
            loss   = loss_fn(logits, y)
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
            t1, t5 = topk_accuracy(logits, y, topk=(1, 5))
            top1_list.append(t1)
            top5_list.append(t5)
    return total_loss / total_samples, np.mean(top1_list), np.mean(top5_list)


# ==========================================
# Training loop
# ==========================================
def train_fusion(train_loader, val_loader, test_loader, cfg):
    device = cfg["device"]
    model = FusionNet(
        out_dim=cfg["out_dim"],
        emb_dim=cfg["emb_dim"],
        C=cfg["C"],
        T=cfg["T"],
        de_dim=cfg["de_dim"],
        psd_dim=cfg["psd_dim"],
        raw_model=cfg["raw_model"],
        de_model=cfg["de_model"],
        psd_model=cfg["psd_model"]
    ).to(device)

    if cfg["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                     weight_decay=cfg["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                      weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    best_val_top1 = 0.0
    best_state = None

    for epoch in range(cfg["num_epochs"]):
        # === Training ===
        model.train()
        total_loss, total_samples = 0.0, 0
        top1_list, top5_list = [], []
        for (raw, de, psd), y in train_loader:
            raw, de, psd, y = raw.to(device), de.to(device), psd.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(raw, de, psd)
            loss   = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
            t1, t5 = topk_accuracy(logits, y, topk=(1, 5))
            top1_list.append(t1)
            top5_list.append(t5)

        train_loss = total_loss / total_samples
        train_top1 = np.mean(top1_list)
        train_top5 = np.mean(top5_list)

        # === Validation + Test ===
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, device)
        test_loss, test_top1, test_top5 = evaluate(model, test_loader, device)

        # === Save best checkpoint ===
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch+1,
                "val_top1": val_top1,
                "val_top5": val_top5
            }

        print(f"[Epoch {epoch+1}] "
              f"Train loss: {train_loss:.4f} | Top1: {train_top1:.3f}, Top5: {train_top5:.3f} || "
              f"Val loss: {val_loss:.4f} | Top1: {val_top1:.3f}, Top5: {val_top5:.3f} || "
              f"Test loss: {test_loss:.4f} | Top1: {test_top1:.3f}, Top5: {test_top5:.3f}")

    # === Restore best checkpoint and re-test ===
    if best_state:
        model.load_state_dict(best_state["model"])
        test_loss, test_top1, test_top5 = evaluate(model, test_loader, device)
        print(f"\n>>> Restored best checkpoint (epoch {best_state['epoch']}) "
              f"| Val Top1: {best_state['val_top1']:.3f} | Val Top5: {best_state['val_top5']:.3f}")
        print(f">>> Final Test after restoring best: Top1={test_top1:.3f}, Top5={test_top5:.3f}")

    return model


# ==========================================
# Main: 1-fold split (first 5 train, 6th val, 7th test)
# ==========================================
if __name__ == "__main__":
    segments = np.load(os.path.join(CONFIG["segments_dir"], CONFIG["subj_id"]))
    de       = np.load(os.path.join(CONFIG["de_dir"], CONFIG["subj_id"]))
    psd      = np.load(os.path.join(CONFIG["psd_dir"], CONFIG["subj_id"]))

    raw = split_segments_to_windows(segments)
    raw, de, psd, labels = flatten_and_labels(raw, de, psd)

    n_blocks = 7
    samples_per_block = labels.size // n_blocks
    raw    = raw.reshape(n_blocks, samples_per_block, 1, CONFIG["C"], CONFIG["T"])
    de     = de.reshape(n_blocks, samples_per_block, CONFIG["C"], 5)
    psd    = psd.reshape(n_blocks, samples_per_block, CONFIG["C"], 5)
    labels = labels.reshape(n_blocks, samples_per_block)

    # --- 1-fold split ---
    train_blocks = list(range(0, 5))   # first 5
    val_block    = 5                   # 6th
    test_block   = 6                   # 7th

    raw_train = raw[train_blocks].reshape(-1, 1, CONFIG["C"], CONFIG["T"])
    de_train  = de[train_blocks].reshape(-1, CONFIG["C"], 5)
    psd_train = psd[train_blocks].reshape(-1, CONFIG["C"], 5)
    y_train   = labels[train_blocks].reshape(-1)

    raw_val = raw[val_block].reshape(-1, 1, CONFIG["C"], CONFIG["T"])
    de_val  = de[val_block].reshape(-1, CONFIG["C"], 5)
    psd_val = psd[val_block].reshape(-1, CONFIG["C"], 5)
    y_val   = labels[val_block].reshape(-1)

    raw_test = raw[test_block].reshape(-1, 1, CONFIG["C"], CONFIG["T"])
    de_test  = de[test_block].reshape(-1, CONFIG["C"], 5)
    psd_test = psd[test_block].reshape(-1, CONFIG["C"], 5)
    y_test   = labels[test_block].reshape(-1)

    # Feature-wise scaling
    raw_train, raw_val, raw_test, _ = scale_feature(raw_train, raw_val, raw_test)
    de_train, de_val, de_test, _    = scale_feature(de_train, de_val, de_test)
    psd_train, psd_val, psd_test, _ = scale_feature(psd_train, psd_val, psd_test)

    train_loader = get_dataloader(raw_train, de_train, psd_train, y_train, True, CONFIG["batch_size"])
    val_loader   = get_dataloader(raw_val,   de_val,   psd_val,   y_val,   False, CONFIG["batch_size"])
    test_loader  = get_dataloader(raw_test,  de_test,  psd_test,  y_test,  False, CONFIG["batch_size"])

    print("\n=== 1-Fold Split | Train blocks = [0–4], Val block = 5, Test block = 6 ===")
    train_fusion(train_loader, val_loader, test_loader, CONFIG)
