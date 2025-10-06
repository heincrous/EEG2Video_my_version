# ==========================================
# EEG → CLIP Semantic Predictor (Improved)
# ==========================================
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import preprocessing
from einops import rearrange
from tqdm import tqdm


# ==========================================
# Config
# ==========================================
FEATURE_TYPE  = "EEG_DE_1per2s"
SUBJECT_NAME  = "sub1.npy"
CLASS_SUBSET  = [0, 9, 11, 15, 18, 22, 24, 30, 33, 38]
SUBSET_ID     = "1"

EPOCHS        = 200
BATCH_SIZE    = 32
LR            = 2e-4
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

EEG_PATH_ROOT  = "/content/drive/MyDrive/EEG2Video_data/processed"
CLIP_PATH      = os.path.join(EEG_PATH_ROOT, "CLIP_embeddings", "CLIP_embeddings.npy")
CKPT_SAVE_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"
EMB_SAVE_PATH  = "/content/drive/MyDrive/EEG2Video_outputs/semantic_embeddings"

os.makedirs(CKPT_SAVE_PATH, exist_ok=True)
os.makedirs(EMB_SAVE_PATH, exist_ok=True)


# ==========================================
# Model
# ==========================================
class CLIPSemanticMLP(nn.Module):
    def __init__(self, input_dim, hidden=4096, p=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, 77 * 768)
        )
    def forward(self, eeg):
        return self.mlp(eeg)


class EEGTextDataset:
    def __init__(self, eeg, text):
        self.eeg  = eeg
        self.text = text
    def __len__(self): return len(self.eeg)
    def __getitem__(self, idx): return self.eeg[idx], self.text[idx]


# ==========================================
# Data Preparation
# ==========================================
def load_data():
    print(f"Loading EEG features from: {FEATURE_TYPE}/{SUBJECT_NAME}")
    eeg_path = os.path.join(EEG_PATH_ROOT, FEATURE_TYPE, SUBJECT_NAME)
    eeg_data = np.load(eeg_path, allow_pickle=True)
    clip_data = np.load(CLIP_PATH, allow_pickle=True)
    if eeg_data.ndim == 6 and eeg_data.shape[3] in [2, 3, 7]:
        eeg_data = eeg_data.mean(axis=3)
    elif eeg_data.ndim != 5:
        raise ValueError(f"Unexpected EEG shape: {eeg_data.shape}")
    return eeg_data, clip_data


def prepare_data(eeg_data, clip_data):
    train_eeg, test_eeg = eeg_data[:6], eeg_data[6:]
    train_clip, test_clip = clip_data[:6], clip_data[6:]

    train_eeg  = train_eeg[:, CLASS_SUBSET]
    test_eeg   = test_eeg[:, CLASS_SUBSET]
    train_clip = train_clip[:, CLASS_SUBSET]
    test_clip  = test_clip[:, CLASS_SUBSET]

    train_eeg_flat = rearrange(train_eeg, "b c s ch t -> (b c s) (ch t)")
    test_eeg_flat  = rearrange(test_eeg,  "b c s ch t -> (b c s) (ch t)")
    train_clip_flat = rearrange(train_clip, "b c s tok dim -> (b c s) (tok dim)")
    test_clip_flat  = rearrange(test_clip,  "b c s tok dim -> (b c s) (tok dim)")

    # Normalize EEG
    scaler = preprocessing.StandardScaler().fit(train_eeg_flat)
    train_eeg_flat = scaler.transform(train_eeg_flat)
    test_eeg_flat  = scaler.transform(test_eeg_flat)

    # Normalize CLIP (direction only)
    train_clip_flat = train_clip_flat / (np.linalg.norm(train_clip_flat, axis=1, keepdims=True) + 1e-8)
    test_clip_flat  = test_clip_flat  / (np.linalg.norm(test_clip_flat,  axis=1, keepdims=True) + 1e-8)

    return train_eeg_flat, test_eeg_flat, train_clip_flat, test_clip_flat


# ==========================================
# Loss Function (MSE + Cosine)
# ==========================================
def combined_loss(pred, target, alpha=0.1):
    # Normalize to remove scale effects
    pred = F.normalize(pred, dim=1)
    target = F.normalize(target, dim=1)
    mse = F.mse_loss(pred, target)
    cos = 1 - F.cosine_similarity(pred, target, dim=1).mean()
    return mse + alpha * cos


# ==========================================
# Evaluation Utility
# ==========================================
def evaluate_model(model, test_eeg_flat, test_clip_flat):
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.tensor(test_eeg_flat, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    preds = test_preds / (np.linalg.norm(test_preds, axis=1, keepdims=True) + 1e-8)
    gt = test_clip_flat
    avg_cos = np.mean(np.sum(preds * gt, axis=1))
    print(f"  Avg cosine(pred, gt): {avg_cos:.4f}")


# ==========================================
# Training
# ==========================================
def train_model(model, dataloader, optimizer, scheduler, test_eeg_flat, test_clip_flat):
    print(f"Starting training for {FEATURE_TYPE} on subset {SUBSET_ID}...")
    for epoch in tqdm(range(1, EPOCHS + 1)):
        model.train()
        epoch_loss = 0
        for eeg, clip in dataloader:
            eeg, clip = eeg.float().to(DEVICE), clip.float().to(DEVICE)
            optimizer.zero_grad()
            pred = model(eeg)
            loss = combined_loss(pred, clip)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"\n[Epoch {epoch:03d}/{EPOCHS}] Avg Loss: {avg_loss:.6f}")
            evaluate_model(model, test_eeg_flat, test_clip_flat)


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    eeg_data, clip_data = load_data()
    train_eeg_flat, test_eeg_flat, train_clip_flat, test_clip_flat = prepare_data(eeg_data, clip_data)

    input_dim = train_eeg_flat.shape[1]
    dataset = EEGTextDataset(train_eeg_flat, train_clip_flat)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CLIPSemanticMLP(input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(dataloader))

    train_model(model, dataloader, optimizer, scheduler, test_eeg_flat, test_clip_flat)
