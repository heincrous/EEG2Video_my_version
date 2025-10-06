# ==========================================
# EEG → CLIP Semantic Predictor
# Procrustes Alignment (Train Only, Token-Averaged)
# ==========================================
import os
import torch
import numpy as np
from einops import rearrange
from sklearn import preprocessing
from torch.utils.data import DataLoader, TensorDataset
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm

# ==========================================
# Config
# ==========================================
FEATURE_TYPE  = "EEG_DE_1per2s"
SUBJECT_NAME  = "sub1.npy"
SUBSET_ID     = "1"
CLASS_SUBSET  = [0, 9, 11, 15, 18, 22, 24, 30, 33, 38]

EEG_PATH_ROOT = "/content/drive/MyDrive/EEG2Video_data/processed"
CLIP_PATH     = os.path.join(EEG_PATH_ROOT, "CLIP_embeddings", "CLIP_embeddings.npy")
CKPT_PATH     = f"/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints/semantic_predictor_{FEATURE_TYPE}_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}.pt"
SAVE_ROOT     = "/content/drive/MyDrive/EEG2Video_outputs/semantic_embeddings"
R_SAVE_PATH   = f"/content/drive/MyDrive/EEG2Video_checkpoints/procrustes_R_{FEATURE_TYPE}_subset{SUBSET_ID}.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
os.makedirs(SAVE_ROOT, exist_ok=True)

# ==========================================
# Model Definition (same as training)
# ==========================================
class CLIPSemanticMLP(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 10000),
            torch.nn.ReLU(),
            torch.nn.Linear(10000, 10000),
            torch.nn.ReLU(),
            torch.nn.Linear(10000, 10000),
            torch.nn.ReLU(),
            torch.nn.Linear(10000, 10000),
            torch.nn.ReLU(),
            torch.nn.Linear(10000, 77 * 768)
        )
    def forward(self, eeg): return self.mlp(eeg)

# ==========================================
# Load Data
# ==========================================
print(f"Loading EEG + CLIP data for {SUBJECT_NAME}...")
eeg_data = np.load(os.path.join(EEG_PATH_ROOT, FEATURE_TYPE, SUBJECT_NAME), allow_pickle=True)
clip_data = np.load(CLIP_PATH, allow_pickle=True)

train_eeg = eeg_data[:6]
train_clip = clip_data[:6]

train_eeg = train_eeg[:, CLASS_SUBSET]
train_clip = train_clip[:, CLASS_SUBSET]

train_eeg_flat  = rearrange(train_eeg, "b c s ch t -> (b c s) (ch t)")
train_clip_flat = rearrange(train_clip, "b c s tok dim -> (b c s) (tok dim)")

scaler = preprocessing.StandardScaler().fit(train_eeg_flat)
train_eeg_flat = scaler.transform(train_eeg_flat)

print(f"Train EEG shape: {train_eeg_flat.shape}, Train CLIP shape: {train_clip_flat.shape}")

# ==========================================
# Load Trained Model
# ==========================================
input_dim = train_eeg_flat.shape[1]
model = CLIPSemanticMLP(input_dim).to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# ==========================================
# Run Forward Pass (Train Only)
# ==========================================
def run_inference(model, eeg_flat):
    preds = []
    loader = DataLoader(
        TensorDataset(torch.tensor(eeg_flat, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )
    with torch.no_grad():
        for (batch,) in tqdm(loader, desc="Running train inference"):
            preds.append(model(batch.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds, axis=0)

train_preds = run_inference(model, train_eeg_flat)
print(f"✅ Done. Shape of train_preds: {train_preds.shape}")

# ==========================================
# Token-Averaging to 768-D
# ==========================================
print("Averaging token dimension for faster alignment...")
train_preds_mean = train_preds.reshape(-1, 77, 768).mean(axis=1)
train_clip_mean  = train_clip_flat.reshape(-1, 77, 768).mean(axis=1)

print(f"Reduced shape: {train_preds_mean.shape}")

# ==========================================
# Compute Procrustes Rotation
# ==========================================
print("Computing Procrustes alignment on averaged 768-D embeddings...")
train_preds_centered = train_preds_mean - train_preds_mean.mean(0, keepdims=True)
train_clip_centered  = train_clip_mean  - train_clip_mean.mean(0, keepdims=True)

R, _ = orthogonal_procrustes(train_preds_centered, train_clip_centered)
np.save(R_SAVE_PATH, R)
print(f"✅ Saved Procrustes rotation matrix → {R_SAVE_PATH}")

# ==========================================
# Evaluate Improvement (Train Only)
# ==========================================
def cosine(a, b):
    return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)

before = np.mean(cosine(train_preds_mean, train_clip_mean))
aligned_train = (train_preds_centered @ R)
after  = np.mean(cosine(aligned_train, train_clip_mean))

print(f"\nAverage TRAIN cosine before alignment: {before:.4f}")
print(f"Average TRAIN cosine after  alignment: {after:.4f}")
print("\n✅ Fast Procrustes alignment complete.")
