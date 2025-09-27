# ==========================================
# fusion_train_full.py
# End-to-end FusionNet training script
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
from core.models import glfnet, glfnet_mlp


# ==========================================
# Config table
# ==========================================
CONFIG = {
    "batch_size":     256,
    "num_epochs":     100,
    "lr":             0.001,
    "weight_decay":   0.0,
    "optimizer":      "Adam",   # "Adam" or "AdamW"
    "C":              62,       # EEG channels
    "T":              200,      # 200 samples per 1s window (200 Hz)
    "emb_dim":        64,
    "de_dim":         310,      # 62*5
    "psd_dim":        310,      # 62*5
    "out_dim":        40,       # 40 classes
    "device":         "cuda",
    "output_dir":     "./output_dir/",
    "network_name":   "FusionNet"
}


# ==========================================
# FusionNet definition
# ==========================================
class FusionNet(nn.Module):
    def __init__(self, out_dim, emb_dim, C, T, de_dim, psd_dim):
        super(FusionNet, self).__init__()
        self.raw_encoder = glfnet(out_dim=emb_dim, emb_dim=emb_dim, C=C, T=T)
        self.de_encoder  = glfnet_mlp(out_dim=emb_dim, emb_dim=emb_dim, input_dim=de_dim)
        self.psd_encoder = glfnet_mlp(out_dim=emb_dim, emb_dim=emb_dim, input_dim=psd_dim)
        self.classifier  = nn.Linear(emb_dim * 3, out_dim)

    def forward(self, raw, de, psd):
        raw_feat = self.raw_encoder(raw)
        de_feat  = self.de_encoder(de)
        psd_feat = self.psd_encoder(psd)
        fused = torch.cat([raw_feat, de_feat, psd_feat], dim=1)
        return self.classifier(fused)


# ==========================================
# Dataset
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
    # segments: (7,40,5,62,400) -> (7,40,5,2,62,200)
    first = segments[..., :, :200]
    second = segments[..., :, 200:]
    return np.stack([first, second], axis=3)  # insert window dim


def flatten_and_labels(raw, de, psd, gt_labels):
    # raw: (7,40,5,2,62,200), de/psd: (7,40,5,2,62,5), labels: (7,40)
    n_blocks, n_cls, n_clips, n_win = raw.shape[:4]
    N = n_blocks * n_cls * n_clips * n_win
    raw = raw.reshape(N, 1, CONFIG["C"], CONFIG["T"])
    de  = de.reshape(N, CONFIG["C"], 5)
    psd = psd.reshape(N, CONFIG["C"], 5)

    labels = []
    for b in range(n_blocks):
        for c in range(n_cls):
            for clip in range(n_clips):
                for w in range(n_win):
                    labels.append(gt_labels[b, c])
    labels = np.array(labels, dtype=np.int64)
    return raw, de, psd, labels


def apply_scaling(train, val, test):
    scaler = StandardScaler()
    scaler.fit(train.reshape(len(train), -1))
    train = scaler.transform(train.reshape(len(train), -1)).reshape(train.shape)
    val   = scaler.transform(val.reshape(len(val), -1)).reshape(val.shape)
    test  = scaler.transform(test.reshape(len(test), -1)).reshape(test.shape)
    return train, val, test


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
    loss_fn = nn.CrossEntropyLoss()
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
# Training
# ==========================================
def train_fusion(train_loader, val_loader, test_loader, cfg):
    device = cfg["device"]
    model = FusionNet(cfg["out_dim"], cfg["emb_dim"], cfg["C"], cfg["T"],
                      cfg["de_dim"], cfg["psd_dim"]).to(device)
    if cfg["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                     weight_decay=cfg["weight_decay"])
    elif cfg["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                      weight_decay=cfg["weight_decay"])
    else:
        raise ValueError(f"Unsupported optimizer {cfg['optimizer']}")
    loss_fn = nn.CrossEntropyLoss()

    best_val_top1, best_state = 0, None

    for epoch in range(cfg["num_epochs"]):
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
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, device)
        test_loss, test_top1, test_top5 = evaluate(model, test_loader, device)

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_state = model.state_dict()

        print(f"[Epoch {epoch+1}] "
              f"Train loss: {train_loss:.4f} | Top1: {train_top1:.3f}, Top5: {train_top5:.3f} || "
              f"Val loss: {val_loss:.4f} | Top1: {val_top1:.3f}, Top5: {val_top5:.3f} || "
              f"Test loss: {test_loss:.4f} | Top1: {test_top1:.3f}, Top5: {test_top5:.3f}")

    save_path = os.path.join(cfg["output_dir"], cfg["network_name"] + "_best.pth")
    if best_state:
        torch.save(best_state, save_path)
        print(f"Best model saved to {save_path}")
    else:
        print("No improvement during training.")
    return model


# ==========================================
# Main entry (example use)
# ==========================================
if __name__ == "__main__":
    # Example: load subject file (replace with real path)
    subj = np.load("EEG2Video_data/raw/EEG/sub3.npy", allow_pickle=True).item()
    segments, de, psd = subj["segments"], subj["de"], subj["psd"]

    # Split into windows and align
    raw = split_segments_to_windows(segments)
    raw, de, psd, labels = flatten_and_labels(raw, de, psd, subj["labels"])

    # Split dataset indices (here simple 70/15/15 split)
    N = len(labels)
    idx = np.arange(N)
    np.random.shuffle(idx)
    n_train, n_val = int(0.7*N), int(0.85*N)
    train_idx, val_idx, test_idx = idx[:n_train], idx[n_train:n_val], idx[n_val:]

    raw_train, raw_val, raw_test = raw[train_idx], raw[val_idx], raw[test_idx]
    de_train, de_val, de_test   = de[train_idx], de[val_idx], de[test_idx]
    psd_train, psd_val, psd_test = psd[train_idx], psd[val_idx], psd[test_idx]
    y_train, y_val, y_test      = labels[train_idx], labels[val_idx], labels[test_idx]

    # Apply scaling
    raw_train, raw_val, raw_test = apply_scaling(raw_train, raw_val, raw_test)
    de_train, de_val, de_test    = apply_scaling(de_train, de_val, de_test)
    psd_train, psd_val, psd_test = apply_scaling(psd_train, psd_val, psd_test)

    # Build loaders
    train_loader = get_dataloader(raw_train, de_train, psd_train, y_train, True, CONFIG["batch_size"])
    val_loader   = get_dataloader(raw_val,   de_val,   psd_val,   y_val,   False, CONFIG["batch_size"])
    test_loader  = get_dataloader(raw_test,  de_test,  psd_test,  y_test,  False, CONFIG["batch_size"])

    # Train
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    train_fusion(train_loader, val_loader, test_loader, CONFIG)
