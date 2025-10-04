# ==========================================
# EEG → CLIP Semantic Predictor
# (Author Scheme + Subset + Fusion + Cosine + Cleanup)
# ==========================================
import os, torch, numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F


# ==========================================
# Config (author defaults)
# ==========================================
batch_size    = 32
num_epochs    = 50
lr            = 5e-4
run_device    = "cuda"

FEATURE_TYPES = ["DE"]    # choose: ["segments"], ["DE"], ["PSD"], ["windows_100"], ["windows_200"], or fusion list
SUBJECT_NAME  = "sub1.npy"
CLASS_SUBSET  = [0, 2, 4, 10, 11, 12, 22, 26, 29, 37]  # set None for all 40

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

TEMPORAL_MODE = "mean"   # options: "mean" or "concat"

# ==========================================
# Model (same as authors)
# ==========================================
class SemanticMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 10000), nn.ReLU(),
            nn.Linear(10000, 10000), nn.ReLU(),
            nn.Linear(10000, 10000), nn.ReLU(),
            nn.Linear(10000, 10000), nn.ReLU(),
            nn.Linear(10000, 77 * 768)
        )
    def forward(self, x): return self.net(x)


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
# Feature loader (handles fusion)
# ==========================================
def load_features(subname, types):
    feats = []
    for t in types:
        arr = np.load(os.path.join(FEATURE_PATHS[t], subname))

        if t in ["DE", "PSD"]:
            # (7,40,5,62,5)
            arr = arr.reshape(7, 40, 5, 62 * 5)

        elif t == "segments":
            # (7,40,5,62,400)
            arr = arr.reshape(7, 40, 5, 62 * 400)

        elif t == "windows_100":
            # (7,40,5,7,62,100)
            if TEMPORAL_MODE == "mean":
                arr = arr.mean(axis=3).reshape(7, 40, 5, 62 * 100)
            elif TEMPORAL_MODE == "concat":
                arr = arr.reshape(7, 40, 5, 7 * 62 * 100)
            else:
                raise ValueError("TEMPORAL_MODE must be 'mean' or 'concat'.")

        elif t == "windows_200":
            # (7,40,5,3,62,200)
            if TEMPORAL_MODE == "mean":
                arr = arr.mean(axis=3).reshape(7, 40, 5, 62 * 200)
            elif TEMPORAL_MODE == "concat":
                arr = arr.reshape(7, 40, 5, 3 * 62 * 200)
            else:
                raise ValueError("TEMPORAL_MODE must be 'mean' or 'concat'.")

        # flatten (7 blocks × 40 classes × 5 clips)
        arr = arr.reshape(-1, arr.shape[-1])
        feats.append(arr)

    return np.concatenate(feats, axis=1)


# ==========================================
# Evaluation (MSE + Cosine metrics)
# ==========================================
def evaluate(model, loader, device):
    model.eval()
    cos = torch.nn.CosineSimilarity(dim=-1)
    total_loss, total_cos, count = 0, 0, 0
    with torch.no_grad():
        for eeg, clip in loader:
            eeg, clip = eeg.to(device), clip.to(device)
            pred = model(eeg)
            loss = F.mse_loss(pred, clip)
            total_loss += loss.item()
            total_cos  += cos(pred, clip).mean().item()
            count += 1
    return total_loss / count, total_cos / count


# ==========================================
# Training (author scheme + cosine metrics)
# ==========================================
def train(model, train_loader, test_loader, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs * len(train_loader))
    cos = torch.nn.CosineSimilarity(dim=-1)

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_cos, count = 0, 0, 0
        for eeg, clip in train_loader:
            eeg, clip = eeg.to(device), clip.to(device)
            opt.zero_grad()
            pred = model(eeg)
            loss = F.mse_loss(pred, clip)
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item()
            total_cos  += cos(pred, clip).mean().item()
            count += 1

        avg_loss = total_loss / count
        avg_cos  = total_cos / count
        test_loss, test_cos = evaluate(model, test_loader, device)
        print(f"[{epoch+1}] train_loss={avg_loss:.4f} | train_cos={avg_cos:.4f} "
              f"| test_loss={test_loss:.4f} | test_cos={test_cos:.4f}")

    os.makedirs(SEMANTIC_CKPT_DIR, exist_ok=True)
    subset_tag = "_subset" + "-".join(map(str, CLASS_SUBSET)) if CLASS_SUBSET else ""
    name = f"semantic_predictor_{'_'.join(FEATURE_TYPES)}_{SUBJECT_NAME.replace('.npy','')}{subset_tag}.pt"
    torch.save({'state_dict': model.state_dict()}, os.path.join(SEMANTIC_CKPT_DIR, name))
    print("Checkpoint saved:", name)


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    print("Loading EEG and CLIP embeddings...")
    eeg = load_features(SUBJECT_NAME, FEATURE_TYPES)
    clip = np.load(CLIP_EMB_PATH).reshape(-1, 77 * 768)
    labels = np.tile(np.repeat(np.arange(40), 5), 7)  # 7 blocks × 40 classes × 5 clips

    # subset selection
    if CLASS_SUBSET:
        mask = np.isin(labels, CLASS_SUBSET)
        eeg, clip, labels = eeg[mask], clip[mask], labels[mask]

    # cleanup old checkpoints/embeddings
    subset_tag = "_subset" + "-".join(map(str, CLASS_SUBSET)) if CLASS_SUBSET else ""
    ckpt_name  = f"semantic_predictor_{'_'.join(FEATURE_TYPES)}_{SUBJECT_NAME.replace('.npy','')}{subset_tag}.pt"
    embed_name = f"embeddings_{'_'.join(FEATURE_TYPES)}_{SUBJECT_NAME.replace('.npy','')}{subset_tag}.npy"
    ckpt_path  = os.path.join(SEMANTIC_CKPT_DIR, ckpt_name)
    embed_path = os.path.join(OUTPUT_DIR, embed_name)

    print("\nChecking for existing files to delete...")
    found = False
    for path in [ckpt_path, embed_path]:
        if os.path.exists(path):
            os.remove(path)
            print(f"🧹 Deleted existing file: {path}")
            found = True
        else:
            print(f"— No existing file found: {path}")
    if not found:
        print("✅ No old checkpoints or embeddings found. Fresh run will begin.")
    else:
        print("✅ All old matching files deleted. Fresh run will begin.\n")

    print("Normalizing EEG features...")
    scaler = StandardScaler().fit(eeg)
    eeg = scaler.transform(eeg)

    # 6-train / 1-test block split
    samples_per_block = (len(CLASS_SUBSET) if CLASS_SUBSET else 40) * 5
    train_idx = np.arange(0, 6 * samples_per_block)
    test_idx  = np.arange(6 * samples_per_block, 7 * samples_per_block)

    X_train, X_test = eeg[train_idx], eeg[test_idx]
    Y_train, Y_test = clip[train_idx], clip[test_idx]

    train_loader = DataLoader(EEGDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(EEGDataset(X_test,  Y_test),  batch_size=batch_size, shuffle=False)

    print(f"Training {SUBJECT_NAME} with {FEATURE_TYPES} (author scheme)...")
    model = SemanticMLP(eeg.shape[1])
    train(model, train_loader, test_loader, run_device)

    print("Generating semantic embeddings for test set only...")
    model.eval()
    with torch.no_grad():
        X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(run_device)
        preds = model(X_test_torch).cpu().numpy()

    # reshape each embedding from (59136,) → (77, 768)
    preds = preds.reshape(preds.shape[0], 77, 768)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(embed_path, preds)
    print(f"Test-set embeddings saved: {embed_path} | Shape: {preds.shape}")


