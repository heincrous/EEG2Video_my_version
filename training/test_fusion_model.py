# ==========================================
# End-to-end FusionNet training (flexible modalities)
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
from core.fusion_model import FusionNet, build_encoder


# ==========================================
# Config table
# ==========================================
CONFIG = {
    "batch_size":     256,
    "num_epochs":     100,
    "lr":             0.001,
    "weight_decay":   0.05,
    "optimizer":      "AdamW",
    "C":              62,       # EEG channels
    "T":              200,      # samples per 1s window (200 Hz)
    "emb_dim":        128,
    "out_dim":        40,       # 40 classes
    "device":         "cuda",

    # Choose which modalities to use
    "use_raw": False,
    "use_de":  True,
    "use_psd": False,

    # Encoder choices
    "raw_model": "glfnet",       # options: shallownet, deepnet, eegnet, tsconv, conformer, glfnet
    "de_model":  "glfnet_mlp",   # options: mlpnet, glfnet_mlp
    "psd_model": "glfnet_mlp",   # options: mlpnet, glfnet_mlp

    # Feature dimensions
    "de_dim":  310,  # 62*5
    "psd_dim": 310,  # 62*5

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
    def __init__(self, inputs, labels):
        self.inputs = inputs  # dict of modality -> array
        self.labels = labels
        self.modalities = list(inputs.keys())

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        out = {}
        for m in self.modalities:
            out[m] = torch.tensor(self.inputs[m][idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return out, label


def get_dataloader(inputs, labels, is_train, batch_size):
    dataset = FusionDataset(inputs, labels)
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

    labels_block = np.repeat(np.arange(n_cls), n_clips * n_win)
    labels = np.tile(labels_block, n_blocks)
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
        for inputs, y in dataloader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            y = y.to(device)
            logits = model(inputs)
            loss   = loss_fn(logits, y)
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
            t1, t5 = topk_accuracy(logits, y, topk=(1, 5))
            top1_list.append(t1); top5_list.append(t5)
    return total_loss / total_samples, np.mean(top1_list), np.mean(top5_list)


# ==========================================
# Training loop
# ==========================================
def train_fusion(train_loader, val_loader, test_loader, cfg, encoder_cfgs):
    device = cfg["device"]
    model = FusionNet(encoder_cfgs, num_classes=cfg["out_dim"], emb_dim=cfg["emb_dim"]).to(device)

    if cfg["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    best_val_top1, best_state = 0.0, None

    for epoch in range(cfg["num_epochs"]):
        # training
        model.train()
        total_loss, total_samples = 0.0, 0
        top1_list, top5_list = [], []
        for inputs, y in train_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
            t1, t5 = topk_accuracy(logits, y, topk=(1, 5))
            top1_list.append(t1); top5_list.append(t5)

        train_loss = total_loss / total_samples
        train_top1 = np.mean(top1_list); train_top5 = np.mean(top5_list)

        # validation + test
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, device)
        test_loss, test_top1, test_top5 = evaluate(model, test_loader, device)

        # checkpoint
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

    if best_state:
        model.load_state_dict(best_state["model"])
        test_loss, test_top1, test_top5 = evaluate(model, test_loader, device)
        print(f"\n>>> Restored best checkpoint (epoch {best_state['epoch']}) "
              f"| Val Top1: {best_state['val_top1']:.3f} | Val Top5: {best_state['val_top5']:.3f}")
        print(f">>> Final Test after restoring best: Top1={test_top1:.3f}, Top5={test_top5:.3f}")

    return model


# ==========================================
# Main
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

    # split
    train_blocks = list(range(0, 5)); val_block = 5; test_block = 6

    inputs_train, inputs_val, inputs_test = {}, {}, {}
    if CONFIG["use_raw"]:
        inputs_train["raw"] = raw[train_blocks].reshape(-1, 1, CONFIG["C"], CONFIG["T"])
        inputs_val["raw"]   = raw[val_block].reshape(-1, 1, CONFIG["C"], CONFIG["T"])
        inputs_test["raw"]  = raw[test_block].reshape(-1, 1, CONFIG["C"], CONFIG["T"])
        inputs_train["raw"], inputs_val["raw"], inputs_test["raw"], _ = scale_feature(
            inputs_train["raw"], inputs_val["raw"], inputs_test["raw"]
        )
    if CONFIG["use_de"]:
        inputs_train["de"] = de[train_blocks].reshape(-1, CONFIG["C"], 5)
        inputs_val["de"]   = de[val_block].reshape(-1, CONFIG["C"], 5)
        inputs_test["de"]  = de[test_block].reshape(-1, CONFIG["C"], 5)
        inputs_train["de"], inputs_val["de"], inputs_test["de"], _ = scale_feature(
            inputs_train["de"], inputs_val["de"], inputs_test["de"]
        )
    if CONFIG["use_psd"]:
        inputs_train["psd"] = psd[train_blocks].reshape(-1, CONFIG["C"], 5)
        inputs_val["psd"]   = psd[val_block].reshape(-1, CONFIG["C"], 5)
        inputs_test["psd"]  = psd[test_block].reshape(-1, CONFIG["C"], 5)
        inputs_train["psd"], inputs_val["psd"], inputs_test["psd"], _ = scale_feature(
            inputs_train["psd"], inputs_val["psd"], inputs_test["psd"]
        )

    y_train = labels[train_blocks].reshape(-1)
    y_val   = labels[val_block].reshape(-1)
    y_test  = labels[test_block].reshape(-1)

    # dataloaders
    train_loader = get_dataloader(inputs_train, y_train, True, CONFIG["batch_size"])
    val_loader   = get_dataloader(inputs_val,   y_val,   False, CONFIG["batch_size"])
    test_loader  = get_dataloader(inputs_test,  y_test,  False, CONFIG["batch_size"])

    # encoder configs
    encoder_cfgs = {}
    if CONFIG["use_raw"]:
        encoder_cfgs["raw"] = (CONFIG["raw_model"], {"out_dim": CONFIG["emb_dim"], "emb_dim": CONFIG["emb_dim"], "C": CONFIG["C"], "T": CONFIG["T"]})
    if CONFIG["use_de"]:
        encoder_cfgs["de"] = (CONFIG["de_model"], {"out_dim": CONFIG["emb_dim"], "emb_dim": CONFIG["emb_dim"], "input_dim": CONFIG["de_dim"]})
    if CONFIG["use_psd"]:
        encoder_cfgs["psd"] = (CONFIG["psd_model"], {"out_dim": CONFIG["emb_dim"], "emb_dim": CONFIG["emb_dim"], "input_dim": CONFIG["psd_dim"]})

    print("\n=== 1-Fold Split | Train=blocks[0–4], Val=5, Test=6 ===")
    train_fusion(train_loader, val_loader, test_loader, CONFIG, encoder_cfgs)
