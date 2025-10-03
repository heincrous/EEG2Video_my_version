# ==========================================
# EEG → CLIP semantic predictor
# ==========================================
import os
import numpy as np
import torch
from torch import nn
from torch.utils import data
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from einops import rearrange


# ==========================================
# Config
# ==========================================
batch_size    = 32
num_epochs    = 50
lr            = 5e-4
run_device    = "cuda"

# optimizer: "adam" or "adamw"; set WEIGHT_DECAY=0 for Adam
WEIGHT_DECAY   = 0

# scheduler: "cosine" or "constant"
SCHEDULER_TYPE = "cosine"

FEATURE_TYPES  = ["DE"]   # options: ["segments"], ["DE"], ["PSD"], ["windows_100"], ["windows_200"],
                          # or fusion: ["segments","DE"], ["segments","PSD"], ["DE","PSD"], ["segments","DE","PSD"]

# default is subject 1 only; set to True to use all subjects in folder
USE_ALL_SUBJECTS = False
subject_name     = "sub1.npy"

# restrict to certain classes (0–39); set to None for all
CLASS_SUBSET     = [0, 2, 4, 10, 11, 12, 22, 26, 29, 37]

# loss type: "mse", "cosine", "mse+cosine", "contrastive"
LOSS_TYPE        = "mse"

USE_VAR_REG = False
VAR_LAMBDA  = 0.01

P = 0 # dropout prob

# checkpoint metric: "val_mse", "val_cos", "val_fisher"
CHECKPOINT_METRIC = "val_cos"

FEATURE_PATHS = {
    "segments":    "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments",
    "DE":          "/content/drive/MyDrive/EEG2Video_data/processed/EEG_DE_1per2s",
    "PSD":         "/content/drive/MyDrive/EEG2Video_data/processed/EEG_PSD_1per2s",
    "windows_100": "/content/drive/MyDrive/EEG2Video_data/processed/EEG_windows_100",  # (7,40,5,7,62,100)
    "windows_200": "/content/drive/MyDrive/EEG2Video_data/processed/EEG_windows_200",  # (7,40,5,3,62,200)
}

CLIP_EMB_PATH     = "/content/drive/MyDrive/EEG2Video_data/processed/CLIP_embeddings/CLIP_embeddings.npy"
SEMANTIC_CKPT_DIR = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"


# ==========================================
# Semantic MLP
# ==========================================
class SemanticMLP(nn.Module):
    def __init__(self, input_dim, p=P):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 10000), nn.ReLU(), nn.Dropout(p),
            nn.Linear(10000, 10000), nn.ReLU(), nn.Dropout(p),
            nn.Linear(10000, 10000), nn.ReLU(), nn.Dropout(p),
            nn.Linear(10000, 10000), nn.ReLU(), nn.Dropout(p),
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
# Training loop (no validation)
# ==========================================
def train(net, train_iter, test_iter, num_epochs, lr, device, subname="subject"):
    net.to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    if SCHEDULER_TYPE.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs * len(train_iter)
        )
    else:
        scheduler = None

    mse_loss  = nn.MSELoss()
    cos_loss  = nn.CosineEmbeddingLoss()

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
            if scheduler:
                scheduler.step()

            total_loss += loss.item() * y.size(0)

        # evaluate on test set
        test_mse, test_cos, fisher_score = evaluate(net, test_iter, device)
        print(f"[{epoch+1}] train_loss={total_loss/len(train_iter.dataset):.4f}, "
              f"test_mse={test_mse:.4f}, test_cos={test_cos:.4f}, fisher_score={fisher_score:.4f}")

    # save after all epochs
    os.makedirs(SEMANTIC_CKPT_DIR, exist_ok=True)
    subset_tag = ""
    if CLASS_SUBSET is not None:
        subset_tag = "_subset" + "-".join(str(c) for c in CLASS_SUBSET)

    model_name = f"semantic_predictor_{'_'.join(FEATURE_TYPES)}_{subname.replace('.npy','')}{subset_tag}.pt"
    torch.save({"state_dict": net.state_dict()},
               os.path.join(SEMANTIC_CKPT_DIR, model_name))
    print(f"Saved checkpoint: {model_name}")
    return net


# ==========================================
# Evaluation
# ==========================================
def evaluate(net, data_iter, device):
    net.eval()
    cos = nn.CosineSimilarity(dim=-1)
    total_mse, total_cos, count, dim = 0, 0, 0, None
    preds_all, labels_all = [], []
    with torch.no_grad():
        for X, y, lbl in data_iter:
            X, y, lbl = X.to(device), y.to(device), lbl.to(device)
            y_hat = net(X)
            total_mse += F.mse_loss(y_hat, y, reduction="sum").item()
            total_cos += cos(y_hat, y).mean().item() * y.size(0)
            preds_all.append(y_hat.cpu().numpy())
            labels_all.append(lbl.cpu().numpy())
            count += y.size(0)
            dim = y.size(1)
    preds_all  = np.concatenate(preds_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)

    # compute fisher score dynamically
    try:
        classes = np.unique(labels_all)
        class_samples = [preds_all[labels_all == c] for c in classes]
        class_means   = [c.mean(axis=0) for c in class_samples]
        overall_mean  = np.mean(class_means, axis=0)
        between = sum(len(c) * np.sum((m - overall_mean) ** 2)
                      for c, m in zip(class_samples, class_means))
        within  = sum(np.sum((c - m) ** 2) for c, m in zip(class_samples, class_means))
        fisher_score = between / (within + 1e-8)
    except Exception:
        fisher_score = 0.0

    return total_mse / (count * dim), total_cos / count, fisher_score


# ==========================================
# Helpers
# ==========================================
def load_subject_data(subname, feature_types):
    # prevent windows in fusion mode
    if len(feature_types) > 1 and any(ft.startswith("windows") for ft in feature_types):
        raise ValueError("Windows can only be used in single-feature mode, not fusion.")

    feats = []
    for ft in feature_types:
        path = os.path.join(FEATURE_PATHS[ft], subname)
        arr  = np.load(path)

        if ft in ["DE","PSD"]:
            # DE/PSD 1per2s: (7,40,5,62,5) -> [N,310]
            arr = arr.reshape(-1, 62*5)

        elif ft == "segments":
            # full 2s segment: (7,40,5,62,400) -> [N,24800]
            arr = arr.reshape(-1, 62*400)

        elif ft == "windows_200":  
            # (7,40,5,3,62,200) -> [N,12400]
            arr = arr.reshape(-1, 62*200)

        elif ft == "windows_100":  
            # (7,40,5,7,62,100) -> [N,6200]
            arr = arr.reshape(-1, 62*100)

        feats.append(arr)

    return np.concatenate(feats, axis=1)


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    clip_embeddings = np.load(CLIP_EMB_PATH)                 # [7,40,5,77,768]
    clip_embeddings = clip_embeddings.reshape(-1, 77*768)    # [1400, 77*768]

    labels_block = np.repeat(np.arange(40), 5)  # 200 per block (not doubled)
    labels_all   = np.tile(labels_block, 7)     # 1400 labels

    sub_list = os.listdir(FEATURE_PATHS[FEATURE_TYPES[0]]) if USE_ALL_SUBJECTS else [subject_name]

    for subname in sub_list:
        print(f"\n=== Training subject {subname} with {FEATURE_TYPES} ===")

        features = load_subject_data(subname, FEATURE_TYPES)

        samples_per_block = 40 * 5
        valid_len = samples_per_block * 7
        features = features[:valid_len]
        Y = clip_embeddings[:valid_len]
        L = labels_all[:valid_len]

        if CLASS_SUBSET is not None:
            mask = np.isin(L, CLASS_SUBSET)
            features, Y, L = features[mask], Y[mask], L[mask]

        # block structure: 7 blocks → first 6 train, last 1 test
        samples_per_block = (len(CLASS_SUBSET) if CLASS_SUBSET else 40) * 5

        train_idx = np.arange(0, 6*samples_per_block)
        test_idx  = np.arange(6*samples_per_block, 7*samples_per_block)

        X_train, X_test = features[train_idx], features[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        L_train, L_test = L[train_idx], L[test_idx]

        # global scaler (fit on all features, then split)
        scaler = StandardScaler().fit(features)
        features = scaler.transform(features)

        X_train, X_test = features[train_idx], features[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        L_train, L_test = L[train_idx], L[test_idx]


        train_iter = Get_Dataloader(X_train, Y_train, L_train, True,  batch_size)
        test_iter  = Get_Dataloader(X_test,  Y_test,  L_test,  False, batch_size)

        input_dim = features.shape[1]
        modelnet = SemanticPredictor(input_dim)
        modelnet = train(modelnet, train_iter, test_iter,
                         num_epochs, lr, run_device, subname=subname)
