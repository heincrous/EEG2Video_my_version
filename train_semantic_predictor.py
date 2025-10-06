# ==========================================
# EEG → CLIP Semantic Predictor
#
# Data shape reference:
# segments:        (7, 40, 5, 62, 400)
# 1 per 1 s:       (7, 40, 5, 2, 62, 5)
# 1 per 2 s:       (7, 40, 5, 62, 5)
# windows_100:     (7, 40, 5, 7, 62, 100)
# windows_200:     (7, 40, 5, 3, 62, 200)
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
FEATURE_TYPE  = "EEG_windows_100"
SUBJECT_NAME  = "sub1.npy"
CLASS_SUBSET  = [0, 9, 11, 15, 18, 22, 24, 30, 33, 38]
SUBSET_ID     = "1"

EPOCHS        = 50
BATCH_SIZE    = 32
LR            = 5e-4
DEVICE        = "cuda:0" if torch.cuda.is_available() else "cpu"

EEG_PATH_ROOT   = "/content/drive/MyDrive/EEG2Video_data/processed"
CLIP_PATH        = os.path.join(EEG_PATH_ROOT, "CLIP_embeddings", "CLIP_embeddings.npy")
CKPT_SAVE_PATH  = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"
EMB_SAVE_PATH   = "/content/drive/MyDrive/EEG2Video_outputs/semantic_embeddings"

os.makedirs(CKPT_SAVE_PATH, exist_ok=True)
os.makedirs(EMB_SAVE_PATH, exist_ok=True)


# ==========================================
# Model
# ==========================================
class CLIPSemanticMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 77 * 768)
        )

    def forward(self, eeg):
        return self.mlp(eeg)


class EEGTextDataset:
    def __init__(self, eeg, text):
        self.eeg  = eeg
        self.text = text

    def __len__(self):
        return self.eeg.shape[0]

    def __getitem__(self, idx):
        return self.eeg[idx], self.text[idx]


# ==========================================
# Utility Functions
# ==========================================
def cleanup_previous_run():
    prefix_ckpt = f"semantic_predictor_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}"
    prefix_emb  = f"pred_embeddings_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}"

    deleted = 0
    for root, _, files in os.walk(CKPT_SAVE_PATH):
        for f in files:
            if prefix_ckpt in f and FEATURE_TYPE in root:
                os.remove(os.path.join(root, f))
                deleted += 1
    for root, _, files in os.walk(EMB_SAVE_PATH):
        for f in files:
            if prefix_emb in f and FEATURE_TYPE in root:
                os.remove(os.path.join(root, f))
                deleted += 1
    print(f"🧹 Deleted {deleted} old file(s) for subset {SUBSET_ID} ({FEATURE_TYPE}).")


def load_data():
    print(f"Loading EEG features from: {FEATURE_TYPE}/{SUBJECT_NAME}")
    eeg_path = os.path.join(EEG_PATH_ROOT, FEATURE_TYPE, SUBJECT_NAME)
    eeg_data = np.load(eeg_path, allow_pickle=True)
    clip_data = np.load(CLIP_PATH, allow_pickle=True)
    print(f"Original EEG shape: {eeg_data.shape}")

    if eeg_data.ndim == 6 and eeg_data.shape[3] in [2, 3, 7]:
        eeg_data = eeg_data.mean(axis=3)
    elif eeg_data.ndim != 5:
        raise ValueError(f"Unexpected EEG shape: {eeg_data.shape}")

    print(f"Processed EEG shape: {eeg_data.shape}")
    print(f"CLIP shape: {clip_data.shape}")
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

    scaler = preprocessing.StandardScaler()
    scaler.fit(train_eeg_flat)
    train_eeg_flat = scaler.transform(train_eeg_flat)
    test_eeg_flat  = scaler.transform(test_eeg_flat)

    return train_eeg_flat, test_eeg_flat, train_clip_flat, test_clip_flat


def evaluate_model(model, test_eeg_flat, test_clip_flat):
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.tensor(test_eeg_flat, dtype=torch.float32, device=DEVICE)).cpu().numpy()
        gt = test_clip_flat

    # --- 1. Average cosine similarity (pred vs ground truth) ---
    cos_sim = np.sum(test_preds * gt, axis=1) / (
        np.linalg.norm(test_preds, axis=1) * np.linalg.norm(gt, axis=1) + 1e-8
    )
    avg_cosine = np.mean(cos_sim)

    # --- 2. Pairwise cosine similarities for within/between classes ---
    num_classes = len(CLASS_SUBSET)
    samples_per_class = 5
    labels = np.repeat(np.arange(num_classes), samples_per_class)

    # Normalize predictions
    norm_preds = test_preds / (np.linalg.norm(test_preds, axis=1, keepdims=True) + 1e-8)

    # Full cosine similarity matrix
    sim_matrix = np.dot(norm_preds, norm_preds.T)

    # Masks
    mask_self    = np.eye(sim_matrix.shape[0], dtype=bool)
    mask_within  = labels[:, None] == labels[None, :]
    mask_between = labels[:, None] != labels[None, :]

    # Extract valid entries
    within_sims  = sim_matrix[mask_within & ~mask_self]
    between_sims = sim_matrix[mask_between]

    avg_within_class  = np.mean(within_sims)
    avg_between_class = np.mean(between_sims)

    # --- 3. Fisher Score (class separability, normalized) ---
    preds_reshaped = test_preds.reshape(num_classes, samples_per_class, -1)
    preds_reshaped = preds_reshaped / (np.linalg.norm(preds_reshaped, axis=2, keepdims=True) + 1e-8)

    global_mean = preds_reshaped.reshape(-1, preds_reshaped.shape[-1]).mean(axis=0)
    numerator = 0.0
    denominator = 0.0
    for c in range(num_classes):
        class_mean = preds_reshaped[c].mean(axis=0)
        numerator += preds_reshaped[c].shape[0] * np.sum((class_mean - global_mean) ** 2)
        denominator += np.sum((preds_reshaped[c] - class_mean) ** 2)
    fisher_score = numerator / (denominator + 1e-8)

    print(f"[Eval] Avg cosine: {avg_cosine:.4f} | Within-class: {avg_within_class:.4f} | "
          f"Between-class: {avg_between_class:.4f} | Fisher Score: {fisher_score:.4f}")


def train_model(model, dataloader, optimizer, scheduler, test_eeg_flat, test_clip_flat):
    print("Starting training...")
    for epoch in tqdm(range(1, EPOCHS + 1)):
        model.train()
        epoch_loss = 0
        for eeg, clip in dataloader:
            eeg, clip = eeg.float().to(DEVICE), clip.float().to(DEVICE)
            optimizer.zero_grad()
            pred = model(eeg)
            loss = F.mse_loss(pred, clip)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] Avg Loss: {epoch_loss:.6f}")
            evaluate_model(model, test_eeg_flat, test_clip_flat)


def save_outputs(model, test_eeg_flat):
    with torch.no_grad():
        preds = model(torch.tensor(test_eeg_flat, dtype=torch.float32, device=DEVICE)).cpu().numpy()

    ckpt_name = f"semantic_predictor_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}.pt"
    emb_name  = f"pred_embeddings_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}.npy"

    torch.save({'state_dict': model.state_dict()}, os.path.join(CKPT_SAVE_PATH, ckpt_name))
    np.save(os.path.join(EMB_SAVE_PATH, emb_name), preds)

    print(f"Saved → {ckpt_name}")
    print(f"Saved → {emb_name}")


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    cleanup_previous_run()
    eeg_data, clip_data = load_data()
    train_eeg_flat, test_eeg_flat, train_clip_flat, test_clip_flat = prepare_data(eeg_data, clip_data)

    input_dim = train_eeg_flat.shape[1]
    dataset = EEGTextDataset(train_eeg_flat, train_clip_flat)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CLIPSemanticMLP(input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(dataloader)
    )

    train_model(model, dataloader, optimizer, scheduler, test_eeg_flat, test_clip_flat)
    save_outputs(model, test_eeg_flat)
