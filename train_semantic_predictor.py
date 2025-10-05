# ==========================================
# EEG → CLIP Semantic Predictor
# (block–class–clip order + cosine metrics every 10 epochs + verified reshape + correct LR step)
# ==========================================
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from einops import rearrange


# ==========================================
# Config
# ==========================================
FEATURE_TYPE  = "EEG_windows_100"
SUBJECT_NAME  = "sub1.npy"
CLASS_SUBSET  = [0, 9, 11, 15, 18, 22, 24, 30, 33, 38]
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
# Load EEG and CLIP data
# ==========================================
print(f"Loading EEG features from: {FEATURE_TYPE}/{SUBJECT_NAME}")
eeg_path = os.path.join(EEG_PATH_ROOT, FEATURE_TYPE, SUBJECT_NAME)
eeg_data = np.load(eeg_path, allow_pickle=True)
clip_data = np.load(CLIP_PATH, allow_pickle=True)
print(f"Original EEG shape: {eeg_data.shape}")

# Average feature dimension if needed
if eeg_data.ndim == 5:
    pass
elif eeg_data.ndim == 6 and eeg_data.shape[3] in [2, 3, 7]:
    eeg_data = eeg_data.mean(axis=3)
else:
    raise ValueError(f"Unexpected EEG shape: {eeg_data.shape}")

print(f"Processed EEG shape after averaging: {eeg_data.shape}")
print(f"CLIP shape: {clip_data.shape}")


# ==========================================
# Split: first 6 blocks train, last block test
# ==========================================
train_eeg, test_eeg = eeg_data[:6], eeg_data[6:]
train_clip, test_clip = clip_data[:6], clip_data[6:]


# ==========================================
# Apply class subset
# ==========================================
train_eeg  = train_eeg[:, CLASS_SUBSET]
test_eeg   = test_eeg[:, CLASS_SUBSET]
train_clip = train_clip[:, CLASS_SUBSET]
test_clip  = test_clip[:, CLASS_SUBSET]


# ==========================================
# Flatten while preserving (block, class, clip)
# ==========================================
train_eeg_flat = rearrange(train_eeg, "b c s ch t -> (b c s) (ch t)")
test_eeg_flat  = rearrange(test_eeg,  "b c s ch t -> (b c s) (ch t)")
train_clip_flat = rearrange(train_clip, "b c s tok dim -> (b c s) (tok dim)")
test_clip_flat  = rearrange(test_clip,  "b c s tok dim -> (b c s) (tok dim)")

input_dim = train_eeg_flat.shape[1]


# ==========================================
# Normalize
# ==========================================
scaler = preprocessing.StandardScaler()
scaler.fit(train_eeg_flat)
train_eeg_flat = scaler.transform(train_eeg_flat)
test_eeg_flat  = scaler.transform(test_eeg_flat)


# ==========================================
# Train
# ==========================================
dataset = EEGTextDataset(train_eeg_flat, train_clip_flat)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = CLIPSemanticMLP(input_dim=input_dim).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


def evaluate_cosine(model, test_eeg_flat, test_clip_flat, num_classes, clips_per_class, num_blocks=1):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(test_eeg_flat, dtype=torch.float32, device=DEVICE)).cpu().numpy()

    # alignment
    alignment = np.mean([
        cosine_similarity([preds[i]], [test_clip_flat[i]])[0][0]
        for i in range(len(preds))
    ])

    # check reshape validity
    assert num_blocks * num_classes * clips_per_class == preds.shape[0], \
        "Reshape mismatch: check flattening order."

    preds = preds.reshape(num_blocks, num_classes, clips_per_class, -1)

    intra_sims, inter_sims = [], []
    for c in range(num_classes):
        preds_c = preds[:, c].reshape(num_blocks * clips_per_class, -1)
        sim_matrix = cosine_similarity(preds_c)
        intra_sims.append(np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)]))
        for c2 in range(c + 1, num_classes):
            preds_c2 = preds[:, c2].reshape(num_blocks * clips_per_class, -1)
            inter_val = cosine_similarity(
                preds_c.mean(0).reshape(1, -1),
                preds_c2.mean(0).reshape(1, -1)
            )[0][0]
            inter_sims.append(inter_val)

    return alignment, np.mean(intra_sims), np.mean(inter_sims)


print("Starting training...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for eeg, clip in dataloader:
        eeg, clip = eeg.float().to(DEVICE), clip.float().to(DEVICE)
        optimizer.zero_grad()
        pred = model(eeg)
        loss = F.mse_loss(pred, clip)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()  # moved to end of epoch

    if epoch % 10 == 0:
        alignment, intra, inter = evaluate_cosine(
            model, test_eeg_flat, test_clip_flat,
            len(CLASS_SUBSET), test_eeg.shape[2]
        )
        print(f"[Epoch {epoch}] Loss: {total_loss:.6f} | Align: {alignment:.4f} | Intra: {intra:.4f} | Inter: {inter:.4f}")


# ==========================================
# Final evaluation + save
# ==========================================
alignment, intra, inter = evaluate_cosine(
    model, test_eeg_flat, test_clip_flat,
    len(CLASS_SUBSET), test_eeg.shape[2]
)
print(f"\nFinal Metrics → Align: {alignment:.4f} | Intra: {intra:.4f} | Inter: {inter:.4f}")

with torch.no_grad():
    preds = model(torch.tensor(test_eeg_flat, dtype=torch.float32, device=DEVICE)).cpu().numpy()

ckpt_name = f"semantic_predictor_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}.pt"
emb_name  = f"pred_embeddings_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}.npy"
torch.save({'state_dict': model.state_dict()}, os.path.join(CKPT_SAVE_PATH, ckpt_name))
np.save(os.path.join(EMB_SAVE_PATH, emb_name), preds)

print(f"Saved → {ckpt_name}")
print(f"Saved → {emb_name}")
