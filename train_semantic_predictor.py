# ==========================================
# EEG → CLIP Semantic Predictor
# (Improved Alignment + Separation)
# ==========================================
import os, torch, numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F


# ==========================================
# Config
# ==========================================
batch_size    = 32
num_epochs    = 200
lr            = 5e-4
run_device    = "cuda"

FEATURE_TYPES = ["DE"]
SUBJECT_NAME  = "sub1.npy"
CLASS_SUBSET  = [0, 2, 4, 10, 11, 12, 22, 26, 29, 37]

FEATURE_PATHS = {
    "segments":    "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments",
    "DE":          "/content/drive/MyDrive/EEG2Video_data/processed/EEG_DE_1per2s",
    "PSD":         "/content/drive/MyDrive/EEG2Video_data/processed/EEG_PSD_1per2s",
    "windows_100": "/content/drive/MyDrive/EEG2Video_data/processed/EEG_windows_100",
    "windows_200": "/content/drive/MyDrive/EEG2Video_data/processed/EEG_windows_200"
}
CLIP_EMB_PATH     = "/content/drive/MyDrive/EEG2Video_data/processed/CLIP_embeddings/CLIP_embeddings.npy"
SEMANTIC_CKPT_DIR = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"
OUTPUT_DIR        = "/content/drive/MyDrive/EEG2Video_outputs/semantic_embeddings"

TEMPORAL_MODE = "mean"


# ==========================================
# Model (dropout added)
# ==========================================
class SemanticMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 10000), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(10000, 10000), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(10000, 10000), nn.ReLU(),
            nn.Linear(10000, 10000), nn.ReLU(),
            nn.Linear(10000, 77 * 768)
        )
    def forward(self, x):
        return self.net(x)


# ==========================================
# Dataset
# ==========================================
class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, eeg, clip):
        self.eeg, self.clip = eeg, clip
    def __len__(self): return len(self.eeg)
    def __getitem__(self, i):
        return torch.tensor(self.eeg[i], dtype=torch.float32), \
               torch.tensor(self.clip[i], dtype=torch.float32)


# ==========================================
# Feature loader
# ==========================================
def load_features(subname, types):
    feats = []
    for t in types:
        arr = np.load(os.path.join(FEATURE_PATHS[t], subname))

        if t in ["DE", "PSD"]:
            arr = arr.reshape(7, 40, 5, 62 * 5)
        elif t == "segments":
            arr = arr.reshape(7, 40, 5, 62 * 400)
        elif t == "windows_100":
            if TEMPORAL_MODE == "mean":
                arr = arr.mean(axis=3).reshape(7, 40, 5, 62 * 100)
            else:
                arr = arr.reshape(7, 40, 5, 7 * 62 * 100)
        elif t == "windows_200":
            if TEMPORAL_MODE == "mean":
                arr = arr.mean(axis=3).reshape(7, 40, 5, 62 * 200)
            else:
                arr = arr.reshape(7, 40, 5, 3 * 62 * 200)
        arr = arr.reshape(-1, arr.shape[-1])
        feats.append(arr)

    return np.concatenate(feats, axis=1)


# ==========================================
# Evaluation
# ==========================================
def evaluate(model, loader, device):
    model.eval()
    cos = torch.nn.CosineSimilarity(dim=-1)
    total_loss, total_cos, count = 0, 0, 0
    with torch.no_grad():
        for eeg, clip in loader:
            eeg, clip = eeg.to(device), clip.to(device)
            pred = model(eeg)
            pred_n = F.normalize(pred, dim=-1)
            clip_n = F.normalize(clip, dim=-1)
            loss = F.cosine_embedding_loss(pred_n, clip_n, torch.ones(pred.size(0), device=device))
            total_loss += loss.item()
            total_cos  += cos(pred_n, clip_n).mean().item()
            count += 1
    return total_loss / count, total_cos / count


# ==========================================
# Training (improved loss)
# ==========================================
def train(model, train_loader, test_loader, device, clip_norm_mean):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_cos, count = 0, 0, 0
        for eeg, clip in train_loader:
            eeg, clip = eeg.to(device), clip.to(device)
            opt.zero_grad()

            pred = model(eeg)
            pred_n = F.normalize(pred, dim=-1)
            clip_n = F.normalize(clip, dim=-1)

            # cosine alignment
            cos_loss = F.cosine_embedding_loss(pred_n, clip_n, torch.ones(pred.size(0), device=device))

            # distribution match
            mean_loss = F.mse_loss(pred_n.mean(0), clip_n.mean(0))
            var_loss  = F.mse_loss(pred_n.var(0),  clip_n.var(0))

            # contrastive spread
            sim = pred_n @ pred_n.T
            mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
            contrast_loss = sim[mask].mean()

            loss = cos_loss + 0.05 * (mean_loss + var_loss) + 0.1 * contrast_loss
            loss.backward()
            opt.step()

            with torch.no_grad():
                total_loss += loss.item()
                total_cos  += (pred_n * clip_n).sum(-1).mean().item()
                count += 1

        avg_loss = total_loss / count
        avg_cos  = total_cos / count
        test_loss, test_cos = evaluate(model, test_loader, device)
        print(f"[{epoch+1}] train_loss={avg_loss:.4f} | train_cos={avg_cos:.4f} "
              f"| test_loss={test_loss:.4f} | test_cos={test_cos:.4f}")
        sched.step()

    os.makedirs(SEMANTIC_CKPT_DIR, exist_ok=True)
    name = f"semantic_predictor_{'_'.join(FEATURE_TYPES)}_{SUBJECT_NAME.replace('.npy','')}.pt"
    torch.save({'state_dict': model.state_dict()}, os.path.join(SEMANTIC_CKPT_DIR, name))
    print("Checkpoint saved:", name)


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    print("Loading EEG and CLIP embeddings...")
    eeg = load_features(SUBJECT_NAME, FEATURE_TYPES)
    clip = np.load(CLIP_EMB_PATH).reshape(-1, 77 * 768)
    labels = np.tile(np.repeat(np.arange(40), 5), 7)

    if CLASS_SUBSET:
        mask = np.isin(labels, CLASS_SUBSET)
        eeg, clip, labels = eeg[mask], clip[mask], labels[mask]

    samples_per_block = (len(CLASS_SUBSET) if CLASS_SUBSET else 40) * 5
    train_idx = np.arange(0, 6 * samples_per_block)
    test_idx  = np.arange(6 * samples_per_block, 7 * samples_per_block)

    X_train_raw, X_test_raw = eeg[train_idx], eeg[test_idx]
    Y_train, Y_test = clip[train_idx], clip[test_idx]

    print("Averaging CLIP embeddings per class for training targets...")
    unique_classes = np.unique(labels[train_idx])
    averaged_targets = {c: clip[train_idx][labels[train_idx] == c].mean(axis=0) for c in unique_classes}
    Y_train = np.array([averaged_targets[c] for c in labels[train_idx]])

    # calculate CLIP norm mean from training set only
    clip_norm_mean = np.linalg.norm(Y_train, axis=1).mean()
    print(f"Mean CLIP norm (train set): {clip_norm_mean:.3f}")

    print("Normalizing EEG features (fit on training data only)...")
    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    train_loader = DataLoader(EEGDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(EEGDataset(X_test,  Y_test),  batch_size=batch_size, shuffle=False)

    print(f"Training {SUBJECT_NAME} with {FEATURE_TYPES} (improved loss + normalization)...")
    model = SemanticMLP(eeg.shape[1])
    train(model, train_loader, test_loader, run_device, clip_norm_mean)

    print("Generating semantic embeddings for test set only...")
    model.eval()
    with torch.no_grad():
        X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(run_device)
        preds = model(X_test_torch)
        preds = F.normalize(preds, dim=-1) * clip_norm_mean
        preds = preds.cpu().numpy()

    preds = preds.reshape(preds.shape[0], 77, 768)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    embed_path = os.path.join(OUTPUT_DIR,
        f"embeddings_{'_'.join(FEATURE_TYPES)}_{SUBJECT_NAME.replace('.npy','')}.npy")
    np.save(embed_path, preds)
    print(f"Test-set embeddings saved: {embed_path} | Shape: {preds.shape}")
