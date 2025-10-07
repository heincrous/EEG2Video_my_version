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
from sklearn.preprocessing import StandardScaler
from einops import rearrange
from tqdm import tqdm


# ==========================================
# Config
# ==========================================
FEATURE_TYPE   = "EEG_DE_1per2s"  # single feature if fusion is empty
FEATURE_FUSION = []  # e.g. ["EEG_DE_1per2s", "EEG_PSD_1per2s", "EEG_windows_100"]
SUBJECT_NAME  = "sub1.npy"
CLASS_SUBSET  = [0, 11, 24, 30, 33] # [0, 9, 11, 15, 18, 22, 24, 30, 33, 38]
SUBSET_ID     = "1"

EPOCHS        = 200
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
# Clean-up Utility
# ==========================================
def cleanup_previous_run():
    tag = "_".join(FEATURE_FUSION) if FEATURE_FUSION else FEATURE_TYPE
    prefix_ckpt = f"semantic_predictor_{tag}_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}"
    prefix_emb  = f"pred_embeddings_{tag}_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}"

    deleted = 0
    for root, _, files in os.walk(CKPT_SAVE_PATH):
        for f in files:
            if f.startswith(prefix_ckpt):
                os.remove(os.path.join(root, f))
                deleted += 1
    for root, _, files in os.walk(EMB_SAVE_PATH):
        for f in files:
            if f.startswith(prefix_emb):
                os.remove(os.path.join(root, f))
                deleted += 1
    print(f"🧹 Deleted {deleted} old file(s) for subset {SUBSET_ID} ({tag}).")


# ==========================================
# Data Loading Utility
# ==========================================
def load_data():
    # ----- Single feature -----
    if not FEATURE_FUSION:
        print(f"Loading EEG features from: {FEATURE_TYPE}/{SUBJECT_NAME}")
        eeg_path = os.path.join(EEG_PATH_ROOT, FEATURE_TYPE, SUBJECT_NAME)
        eeg_data = np.load(eeg_path, allow_pickle=True)
        clip_data = np.load(CLIP_PATH, allow_pickle=True)

        if eeg_data.ndim == 6 and eeg_data.shape[3] in [2, 3, 7]:
            eeg_data = eeg_data.mean(axis=3)
        elif eeg_data.ndim != 5:
            raise ValueError(f"Unexpected EEG shape: {eeg_data.shape}")

        print(f"Loaded EEG shape: {eeg_data.shape}")
        print(f"CLIP shape: {clip_data.shape}")
        return eeg_data, clip_data

    # ----- Multi-feature fusion -----
    eeg_list = []
    print(f"Loading EEG features for fusion: {FEATURE_FUSION}")
    for ftype in FEATURE_FUSION:
        path = os.path.join(EEG_PATH_ROOT, ftype, SUBJECT_NAME)
        eeg = np.load(path, allow_pickle=True)

        if eeg.ndim == 6 and eeg.shape[3] in [2, 3, 7]:
            eeg = eeg.mean(axis=3)
        elif eeg.ndim == 7:
            eeg = eeg.mean(axis=3)
        elif eeg.ndim != 5:
            raise ValueError(f"Unexpected EEG shape for {ftype}: {eeg.shape}")

        eeg_list.append(eeg)
        print(f"  {ftype}: shape {eeg.shape}")

    # Align temporal dimension (pad to longest)
    max_t = max(eeg.shape[-1] for eeg in eeg_list)
    padded_list = []
    for eeg in eeg_list:
        t = eeg.shape[-1]
        if t < max_t:
            pad_width = [(0,0)] * eeg.ndim
            pad_width[-1] = (0, max_t - t)
            eeg = np.pad(eeg, pad_width, mode='constant')
        padded_list.append(eeg)
    eeg_list = padded_list
    print(f"Padded feature lengths: {[e.shape[-1] for e in eeg_list]}")

    # Concatenate along last axis
    eeg_data = np.concatenate(eeg_list, axis=-1)
    clip_data = np.load(CLIP_PATH, allow_pickle=True)

    print(f"Fused EEG shape: {eeg_data.shape}")
    print(f"CLIP shape: {clip_data.shape}")
    return eeg_data, clip_data


# ==========================================
# Data Shaping Utility
# ==========================================
def prepare_data(eeg_data, clip_data):
    # Split 6 train, 1 test blocks
    train_eeg, test_eeg = eeg_data[:6], eeg_data[6:]
    train_clip, test_clip = clip_data[:6], clip_data[6:]

    # Apply subset BEFORE scaling
    train_eeg  = train_eeg[:, CLASS_SUBSET]
    test_eeg   = test_eeg[:, CLASS_SUBSET]
    train_clip = train_clip[:, CLASS_SUBSET]
    test_clip  = test_clip[:, CLASS_SUBSET]

    # train_clip = np.repeat(train_clip[:, :, :1, :, :], 5, axis=2)
    # test_clip  = np.repeat(test_clip[:, :, :1, :, :], 5, axis=2)
    # print("Applied authors' scheme: repeated first clip embedding 5× per class.")

    # Flatten EEG & CLIP
    train_eeg_flat  = rearrange(train_eeg,  "b c s ch t -> (b c s) (ch t)")
    test_eeg_flat   = rearrange(test_eeg,   "b c s ch t -> (b c s) (ch t)")
    train_clip_flat = rearrange(train_clip, "b c s tok dim -> (b c s) (tok dim)")
    test_clip_flat  = rearrange(test_clip,  "b c s tok dim -> (b c s) (tok dim)")

    # Scaling
    if not FEATURE_FUSION:
        print("Scaling EEG features (single-feature mode)...")
        scaler = StandardScaler()
        scaler.fit(train_eeg_flat)
        train_eeg_flat = scaler.transform(train_eeg_flat)
        test_eeg_flat  = scaler.transform(test_eeg_flat)

    else:
        print("Scaling fused features separately using training EEG only...")
        n_feats = len(FEATURE_FUSION)
        total_len = train_eeg_flat.shape[1]
        chunk_size = total_len // n_feats

        scaled_train_parts, scaled_test_parts = [], []
        for i in range(n_feats):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < n_feats - 1 else total_len
            scaler = StandardScaler()
            scaler.fit(train_eeg_flat[:, start:end])  # fit only on training EEG
            scaled_train_parts.append(scaler.transform(train_eeg_flat[:, start:end]))
            scaled_test_parts.append(scaler.transform(test_eeg_flat[:, start:end]))

        train_eeg_flat = np.concatenate(scaled_train_parts, axis=1)
        test_eeg_flat  = np.concatenate(scaled_test_parts, axis=1)

    # ===== Scaling sanity checks =====
    print(f"[EEG scaler] train mean={np.mean(train_eeg_flat):.5f}, std={np.std(train_eeg_flat):.5f}")
    print(f"[EEG scaler] test  mean={np.mean(test_eeg_flat):.5f}, std={np.std(test_eeg_flat):.5f}")
    print(f"[CLIP check]  train mean={np.mean(train_clip_flat):.5f}, std={np.std(train_clip_flat):.5f}")

    return train_eeg_flat, test_eeg_flat, train_clip_flat, test_clip_flat


# ==========================================
# Evaluation Utility
# ==========================================
def evaluate_model(model, test_eeg_flat, test_clip_flat):
    model.eval()
    with torch.no_grad():
        test_preds = model(
            torch.tensor(test_eeg_flat, dtype=torch.float32, device=DEVICE)
        ).cpu().numpy()
        gt = test_clip_flat

    # === Normalize predicted and GT embeddings ===
    preds = test_preds / (np.linalg.norm(test_preds, axis=1, keepdims=True) + 1e-8)
    gt_norm = gt / (np.linalg.norm(gt, axis=1, keepdims=True) + 1e-8)

    num_classes = len(CLASS_SUBSET)
    samples_per_class = 5
    labels = np.repeat(np.arange(num_classes), samples_per_class)

    # === 1. Average cosine similarity (pred vs ground truth) ===
    avg_cosine = np.mean(np.sum(preds * gt_norm, axis=1))

    # === 2. Compute per-class means ===
    class_means = np.zeros((num_classes, preds.shape[1]))
    for c in range(num_classes):
        class_means[c] = preds[labels == c].mean(axis=0)
        class_means[c] /= np.linalg.norm(class_means[c]) + 1e-8

    # === 3. Within-class similarity ===
    within_scores = []
    for i, p in enumerate(preds):
        c = labels[i]
        sim = np.dot(p, class_means[c])
        within_scores.append(sim)
    avg_within = np.mean(within_scores)

    # === 4. Between-class similarity ===
    between_scores = []
    for i, p in enumerate(preds):
        c = labels[i]
        sims = np.dot(class_means[np.arange(num_classes) != c], p)
        between_scores.extend(sims)
    avg_between = np.mean(between_scores)

    # === 5. Fisher-style class separability (optional) ===
    global_mean = class_means.mean(axis=0)
    numerator = np.sum([np.sum((m - global_mean) ** 2) for m in class_means])
    denominator = np.sum([
        np.sum((preds[labels == c] - class_means[c]) ** 2)
        for c in range(num_classes)
    ])
    fisher_score = numerator / (denominator + 1e-8)

    print(
        f"  Avg cosine(pred,gt): {avg_cosine:.4f}\n"
        f"  Within-class cosine: {avg_within:.4f}\n"
        f"  Between-class cosine: {avg_between:.4f}\n"
        f"  Fisher Score: {fisher_score:.4f}\n"
        f"  Δ (Within−Between): {avg_within - avg_between:.4f}"
    )


# ==========================================
# Training Utility
# ==========================================
def train_model(model, dataloader, optimizer, scheduler, test_eeg_flat, test_clip_flat):
    tag = "_".join(FEATURE_FUSION) if FEATURE_FUSION else FEATURE_TYPE
    print(f"Starting training for {tag} on subset {SUBSET_ID}...")
    for epoch in tqdm(range(1, EPOCHS + 1)):
        model.train()
        epoch_loss = 0
        for eeg, clip in dataloader:
            eeg, clip = eeg.float().to(DEVICE), clip.float().to(DEVICE)
            optimizer.zero_grad()
            pred = model(eeg)
            
            # ==========================================
            # Weighted composite loss (toggleable)
            # ==========================================
            # Toggle between raw-space and normalized-space MSE
            USE_NORMALIZED = True  # False = pure unnormalized MSE, True = combine with cosine

            # Adjustable coefficients
            L_MSE    = 1.0   # weight for MSE loss
            L_COSINE = 5.0   # weight for cosine alignment
            L_MAG    = 0.1   # weight for magnitude consistency

            # === Choose MSE mode ===
            if USE_NORMALIZED:
                # When mixing with cosine, keep MSE on normalized vectors for consistency
                pred_for_mse = F.normalize(pred, p=2, dim=1)
                clip_for_mse = F.normalize(clip, p=2, dim=1)
            else:
                # Pure MSE mode (raw, unnormalized space)
                pred_for_mse = pred
                clip_for_mse = clip

            # (1) MSE loss
            mse_loss = F.mse_loss(pred_for_mse, clip_for_mse) if L_MSE > 0 else torch.tensor(0.0, device=DEVICE)

            # (2) Cosine loss (always on normalized embeddings)
            pred_norm = F.normalize(pred, p=2, dim=1)
            clip_norm = F.normalize(clip, p=2, dim=1)
            cosine_loss = (1 - torch.mean(torch.sum(pred_norm * clip_norm, dim=1))) if L_COSINE > 0 else torch.tensor(0.0, device=DEVICE)

            # (3) Magnitude (norm) loss (always unnormalized)
            pred_mag = torch.norm(pred, dim=1)
            clip_mag = torch.norm(clip, dim=1)
            mag_loss = F.mse_loss(pred_mag, clip_mag) if L_MAG > 0 else torch.tensor(0.0, device=DEVICE)

            # === Combined total loss ===
            loss = (L_MSE * mse_loss) + (L_COSINE * cosine_loss) + (L_MAG * mag_loss)

            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print("\n" + "="*65)
            print(f"[Epoch {epoch:03d}/{EPOCHS}]  Avg Loss: {avg_loss:.6f}")
            print("-"*65)
            evaluate_model(model, test_eeg_flat, test_clip_flat)
            print("="*65 + "\n")


# ==========================================
# Saving Utility
# ==========================================
def save_outputs(model, test_eeg_flat):
    with torch.no_grad():
        preds = model(torch.tensor(test_eeg_flat, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    preds = preds.reshape(-1, 77, 768)  # store true token×embedding structure

    tag = "_".join(FEATURE_FUSION) if FEATURE_FUSION else FEATURE_TYPE
    ckpt_name = f"semantic_predictor_{tag}_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}.pt"
    emb_name  = f"pred_embeddings_{tag}_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}.npy"

    torch.save({'state_dict': model.state_dict()}, os.path.join(CKPT_SAVE_PATH, ckpt_name))
    np.save(os.path.join(EMB_SAVE_PATH, emb_name), preds)

    print(f"Saved → {ckpt_name}")
    print(f"Saved → {emb_name} (shape: {preds.shape})")


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
