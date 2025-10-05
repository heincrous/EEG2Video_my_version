# ==========================================
# EEG → CLIP Semantic Predictor
# (Unified all-feature trainer + 6/1 block split + cosine evaluation)
# ==========================================
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import preprocessing
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from einops import rearrange


# ==========================================
# Config
# ==========================================
FEATURE_TYPE  = "EEG_DE_1per2s"     # EEG_DE_1per2s, EEG_DE_1per1s, EEG_PSD_1per2s, EEG_PSD_1per1s, EEG_segments, EEG_windows_100, EEG_windows_200
SUBJECT_NAME  = "sub1.npy"
CLASS_SUBSET  = [0, 2, 4, 10, 11, 12, 22, 26, 29, 37]
SUBSET_ID     = "1"

EPOCHS        = 200
BATCH_SIZE    = 32
LR            = 5e-4
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

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


# ==========================================
# Dataset
# ==========================================
class EEGTextDataset:
    def __init__(self, eeg, text):
        self.eeg  = eeg
        self.text = text
        self.len  = eeg.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.eeg[idx], self.text[idx]


# ==========================================
# Load data
# ==========================================
print(f"Loading EEG features from: {FEATURE_TYPE}/{SUBJECT_NAME}")
eeg_path = os.path.join(EEG_PATH_ROOT, FEATURE_TYPE, SUBJECT_NAME)
eeg_data = np.load(eeg_path, allow_pickle=True)
clip_data = np.load(CLIP_PATH, allow_pickle=True)

print(f"Loaded EEG shape: {eeg_data.shape}, CLIP shape: {clip_data.shape}")


# ==========================================
# Preprocess EEG (average temporal axes)
# ==========================================
if FEATURE_TYPE in ["EEG_DE_1per1s", "EEG_PSD_1per1s"]:
    eeg_data = eeg_data.mean(axis=2)  # average across 2 frequency bands
elif FEATURE_TYPE == "EEG_windows_100":
    eeg_data = eeg_data.mean(axis=2)  # average across 7 windows
elif FEATURE_TYPE == "EEG_windows_200":
    eeg_data = eeg_data.mean(axis=2)  # average across 3 windows

eeg_data = np.squeeze(eeg_data)
print(f"EEG after averaging: {eeg_data.shape}")


# ==========================================
# Train/test split: first 6 blocks → train, last block → test
# ==========================================
train_eeg, test_eeg = eeg_data[:, :6], eeg_data[:, 6:]
train_clip, test_clip = clip_data[:, :6], clip_data[:, 6:]
print(f"Train EEG: {train_eeg.shape}, Test EEG: {test_eeg.shape}")


# ==========================================
# Subset selection
# ==========================================
train_eeg  = train_eeg[CLASS_SUBSET]
test_eeg   = test_eeg[CLASS_SUBSET]
train_clip = train_clip[CLASS_SUBSET]
test_clip  = test_clip[CLASS_SUBSET]


# ==========================================
# Flatten
# ==========================================
train_eeg_flat = rearrange(train_eeg, "c b ... -> (c b) (...)")
test_eeg_flat  = rearrange(test_eeg, "c b ... -> (c b) (...)")
train_clip_flat = rearrange(train_clip, "c b s d -> (c b) (s d)")
test_clip_flat  = rearrange(test_clip, "c b s d -> (c b) (s d)")

input_dim = train_eeg_flat.shape[1]
print(f"Flattened input dimension: {input_dim}")


# ==========================================
# Normalization (fit only on training EEG)
# ==========================================
scaler = preprocessing.StandardScaler()
scaler.fit(train_eeg_flat)
train_eeg_flat = scaler.transform(train_eeg_flat)
test_eeg_flat  = scaler.transform(test_eeg_flat)


# ==========================================
# Training setup
# ==========================================
dataset    = EEGTextDataset(train_eeg_flat, train_clip_flat)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = CLIPSemanticMLP(input_dim=input_dim).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(dataloader))


# ==========================================
# Train
# ==========================================
print("Starting training...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0

    for eeg, clip in dataloader:
        eeg  = eeg.float().to(DEVICE)
        clip = clip.float().to(DEVICE)

        optimizer.zero_grad()
        pred = model(eeg)
        loss = F.mse_loss(pred, clip)
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()

    if epoch % 10 == 0:
        print(f"[Epoch {epoch:03d}] Training Loss: {epoch_loss:.6f}")


# ==========================================
# Evaluation (on last block)
# ==========================================
print("\nEvaluating on test set...")
model.eval()
with torch.no_grad():
    preds = model(torch.tensor(test_eeg_flat, dtype=torch.float32, device=DEVICE)).cpu().numpy()

# Mean alignment cosine between predicted and GT CLIP
alignment = np.mean([cosine_similarity([preds[i]], [test_clip_flat[i]])[0][0] for i in range(len(preds))])

# Intra- and inter-class cosine similarities
num_classes = len(CLASS_SUBSET)
samples_per_class = test_eeg.shape[1]

intra_sims, inter_sims = [], []

for i in range(num_classes):
    start_i = i * samples_per_class
    end_i   = start_i + samples_per_class
    preds_i = preds[start_i:end_i]

    # Intra-class: mean cosine among this class's predictions
    if len(preds_i) > 1:
        sim_matrix = cosine_similarity(preds_i)
        upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        intra_sims.append(np.mean(upper_tri))

    # Inter-class: cosine vs other classes
    for j in range(i + 1, num_classes):
        start_j = j * samples_per_class
        end_j   = start_j + samples_per_class
        preds_j = preds[start_j:end_j]
        inter_val = cosine_similarity(preds_i.mean(axis=0).reshape(1, -1),
                                      preds_j.mean(axis=0).reshape(1, -1))[0][0]
        inter_sims.append(inter_val)

intra_mean = np.mean(intra_sims)
inter_mean = np.mean(inter_sims)

print(f"Alignment cosine (Pred↔Real): {alignment:.4f}")
print(f"Intra-class cosine (same class): {intra_mean:.4f}")
print(f"Inter-class cosine (different classes): {inter_mean:.4f}")


# ==========================================
# Save checkpoint and predicted embeddings
# ==========================================
ckpt_name = f"semantic_predictor_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}.pt"
emb_name  = f"pred_embeddings_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}.npy"

torch.save({'state_dict': model.state_dict()}, os.path.join(CKPT_SAVE_PATH, ckpt_name))
np.save(os.path.join(EMB_SAVE_PATH, emb_name), preds)

print(f"\nSaved checkpoint → {ckpt_name}")
print(f"Saved predicted embeddings → {emb_name}")
print("Training and evaluation complete.")
