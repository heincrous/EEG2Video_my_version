# ==========================================
# EEG → CLIP semantic predictor (multi-feature fusion, authors' MLP logic)
# With per-feature scaling + saving scalers
# ==========================================
import os
import numpy as np
import torch
from torch import nn
from torch.utils import data
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from einops import rearrange
import joblib

# ==========================================
# Config
# ==========================================
batch_size    = 256
num_epochs    = 200
lr            = 0.0005
run_device    = "cuda"

# Choose: ["segments"], ["DE"], ["PSD"], ["segments","DE"], ["DE","PSD"], ["segments","DE","PSD"]
FEATURE_TYPES    = ["DE"]
USE_ALL_SUBJECTS = False

# Loss type: "mse", "cosine", "mse+cosine", "contrastive"
LOSS_TYPE        = "mse+cosine"

USE_VAR_REG = True
VAR_LAMBDA  = 0.01

FEATURE_PATHS = {
    "segments": "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments",
    "DE":       "/content/drive/MyDrive/EEG2Video_data/processed/EEG_DE_1per1s",
    "PSD":      "/content/drive/MyDrive/EEG2Video_data/processed/EEG_PSD_1per1s",
}

CLIP_EMB_PATH     = "/content/drive/MyDrive/EEG2Video_data/processed/CLIP_embeddings/CLIP_embeddings.npy"
SEMANTIC_CKPT_DIR = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"

# ==========================================
# Semantic MLP (authors' structure)
# ==========================================
class SemanticMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 10000), nn.ReLU(),
            nn.Linear(10000, 10000), nn.ReLU(),
            nn.Linear(10000, 10000), nn.ReLU(),
            nn.Linear(10000, 10000), nn.ReLU(),
            nn.Linear(10000, 77*768)
        )
    def forward(self, x):
        return self.mlp(x)

# ==========================================
# Fusion wrapper
# ==========================================
class SemanticPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.head = SemanticMLP(input_dim)
    def forward(self, x):
        return self.head(x)

# ==========================================
# Dataset
# ==========================================
class FusionDataset(data.Dataset):
    def __init__(self, features, clip_targets, class_labels):
        self.features = features
        self.targets  = clip_targets
        self.labels   = class_labels
    def __len__(self): return len(self.targets)
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), \
               torch.tensor(self.targets[idx], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.long)

def Get_Dataloader(features, targets, labels, istrain, batch_size):
    return data.DataLoader(FusionDataset(features, targets, labels), batch_size, shuffle=istrain)

# ==========================================
# Contrastive loss
# ==========================================
def contrastive_loss_fn(y_hat, y, margin=1.0):
    y_hat = F.normalize(y_hat, dim=-1)
    y     = F.normalize(y, dim=-1)
    sim_matrix = torch.matmul(y_hat, y.t())
    pos = torch.diag(sim_matrix)
    return torch.mean(F.relu(margin - pos[:, None] + sim_matrix))

# ==========================================
# Training loop
# ==========================================
def train(net, train_iter, val_iter, test_iter, num_epochs, lr, device,
          subname="subject", scalers=None):
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    mse_loss  = nn.MSELoss()
    cos_loss  = nn.CosineEmbeddingLoss()

    best_val, best_state = 1e12, None
    for epoch in range(num_epochs):
        net.train()
        total_loss = 0
        for X, y, _ in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)

            if LOSS_TYPE == "mse":
                loss = mse_loss(y_hat, y)
            elif LOSS_TYPE == "cosine":
                target = torch.ones(y_hat.size(0), device=device)
                loss = cos_loss(y_hat, y, target)
            elif LOSS_TYPE in ["mse+cosine", "cosine+mse"]:
                target = torch.ones(y_hat.size(0), device=device)
                loss = mse_loss(y_hat, y) + cos_loss(y_hat, y, target)
            elif LOSS_TYPE == "contrastive":
                loss = contrastive_loss_fn(y_hat, y)
            else:
                raise ValueError(f"Unknown LOSS_TYPE {LOSS_TYPE}")

            if USE_VAR_REG:
                loss -= VAR_LAMBDA * torch.var(y_hat, dim=0).mean()

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)

        val_mse = evaluate_mse(net, val_iter, device)
        if val_mse < best_val:
            best_val, best_state = val_mse, net.state_dict()

        if epoch % 3 == 0:
            test_mse, test_cos, fisher_score = evaluate(net, test_iter, device)
            print(f"[{epoch+1}] train_loss={total_loss/len(train_iter.dataset):.4f}, "
                  f"val_mse={val_mse:.4f}, test_mse={test_mse:.4f}, "
                  f"test_cos={test_cos:.4f}, fisher_score={fisher_score:.4f}")

    if best_state:
        net.load_state_dict(best_state)
        os.makedirs(SEMANTIC_CKPT_DIR, exist_ok=True)
        fname = f"semantic_predictor_{'_'.join(FEATURE_TYPES)}_{subname.replace('.npy','')}"
        # save model
        torch.save({"state_dict": net.state_dict()}, os.path.join(SEMANTIC_CKPT_DIR, fname + ".pt"))
        # save scalers per feature
        for ft, sc in scalers.items():
            joblib.dump(sc, os.path.join(SEMANTIC_CKPT_DIR, fname + f"_{ft}_scaler.pkl"))
    return net

# ==========================================
# Evaluation
# ==========================================
def evaluate_mse(net, data_iter, device):
    net.eval()
    mse_loss = nn.MSELoss(reduction="sum")
    total, count, dim = 0, 0, None
    with torch.no_grad():
        for X, y, _ in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            total += mse_loss(y_hat, y).item()
            count += y.size(0)
            dim = y.size(1)
    return total / (count * dim)

def evaluate(net, data_iter, device):
    net.eval()
    cos = nn.CosineSimilarity(dim=-1)
    total_mse, total_cos, count, dim = 0, 0, 0, None
    preds_all, targets_all = [], []
    with torch.no_grad():
        for X, y, _ in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            total_mse += F.mse_loss(y_hat, y, reduction="sum").item()
            total_cos += cos(y_hat, y).mean().item() * y.size(0)
            preds_all.append(y_hat.cpu().numpy())
            targets_all.append(y.cpu().numpy())
            count += y.size(0)
            dim = y.size(1)
    preds_all = np.concatenate(preds_all, axis=0)
    try:
        preds_all = preds_all.reshape(40, 5, 2, -1)
        class_samples = preds_all.reshape(40, -1, preds_all.shape[-1])
        class_means = class_samples.mean(axis=1)
        overall_mean = class_means.mean(axis=0)
        between = np.sum([len(c) * np.sum((m - overall_mean) ** 2)
                          for m, c in zip(class_means, class_samples)])
        within = np.sum([np.sum((c - m) ** 2) for m, c in zip(class_means, class_samples)])
        fisher_score = between / (within + 1e-8)
    except Exception:
        fisher_score = 0.0
    return total_mse / (count * dim), total_cos / count, fisher_score

# ==========================================
# Helpers
# ==========================================
def load_subject_data(subname, feature_types):
    feats, scalers = [], {}
    for ft in feature_types:
        path = os.path.join(FEATURE_PATHS[ft], subname)
        arr = np.load(path)
        if ft in ["DE","PSD"]:
            arr = arr.reshape(-1, 62*5)
        elif ft == "segments":
            arr = rearrange(arr, "a b c d (w t) -> (a b c w) (d t)", w=2, t=200)

        scaler = StandardScaler().fit(arr)
        arr = scaler.transform(arr)
        scalers[ft] = scaler
        feats.append(arr)
    return np.concatenate(feats, axis=1), scalers

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    clip_embeddings = np.load(CLIP_EMB_PATH)               # [7,40,5,77,768]
    clip_embeddings = clip_embeddings.reshape(-1, 77*768)  # [1400, 77*768]
    clip_embeddings = np.repeat(clip_embeddings, 2, axis=0)  # [2800, 77*768]

    labels_block = np.repeat(np.arange(40), 5*2)
    labels_all   = np.tile(labels_block, 7)

    sub_list = os.listdir(FEATURE_PATHS[FEATURE_TYPES[0]]) if USE_ALL_SUBJECTS else ["sub1.npy"]

    for subname in sub_list:
        print(f"\n=== Training subject {subname} with {FEATURE_TYPES} ===")

        # --- Load and scale features (per-feature) ---
        features, scalers = load_subject_data(subname, FEATURE_TYPES)

        # valid length
        samples_per_block = 400
        valid_len = samples_per_block * 7
        features = features[:valid_len]
        Y = clip_embeddings[:valid_len]
        L = labels_all[:valid_len]

        # Splits (5 train, 1 val, 1 test)
        train_idx = np.arange(0, 5*samples_per_block)
        val_idx   = np.arange(5*samples_per_block, 6*samples_per_block)
        test_idx  = np.arange(6*samples_per_block, 7*samples_per_block)

        X_train, X_val, X_test = features[train_idx], features[val_idx], features[test_idx]
        Y_train, Y_val, Y_test = Y[train_idx], Y[val_idx], Y[test_idx]
        L_train, L_val, L_test = L[train_idx], L[val_idx], L[test_idx]

        train_iter = Get_Dataloader(X_train, Y_train, L_train, True,  batch_size)
        val_iter   = Get_Dataloader(X_val,   Y_val,   L_val,   False, batch_size)
        test_iter  = Get_Dataloader(X_test,  Y_test,  L_test,  False, batch_size)

        # --- Train ---
        input_dim = features.shape[1]
        modelnet = SemanticPredictor(input_dim)
        modelnet = train(modelnet, train_iter, val_iter, test_iter,
                        num_epochs, lr, run_device, subname=subname, scalers=scalers)
