# ==========================================
# EEG → CLIP semantic predictor
# ==========================================
import os
import numpy as np
import torch
from torch import nn
from torch.utils import data
from sklearn.preprocessing import StandardScaler
from einops import rearrange
import torch.nn.functional as F
import models

# ==========================================
# Config
# ==========================================
batch_size   = 128
num_epochs   = 50
lr           = 0.0005
C            = 62
T            = 5
run_device   = "cuda"

emb_dim_segments = 128
emb_dim_DE       = 64
emb_dim_PSD      = 64

# Choose: "segments", "DE", "PSD", or "fusion"
FEATURE_TYPE     = "DE"
USE_ALL_SUBJECTS = False
LOSS_TYPE        = "mse"   # "mse", "cosine", "mse+cosine"

USE_VAR_REG = False
VAR_LAMBDA  = 0.01

FEATURE_PATHS = {
    "segments": "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments",
    "DE":       "/content/drive/MyDrive/EEG2Video_data/processed/EEG_DE_1per1s",
    "PSD":      "/content/drive/MyDrive/EEG2Video_data/processed/EEG_PSD_1per1s",
}

CLIP_EMB_PATH = "/content/drive/MyDrive/EEG2Video_data/processed/CLIP_embeddings/CLIP_embeddings.npy"

MODEL_MAP = {
    "segments": lambda: models.glfnet(out_dim=emb_dim_segments, emb_dim=emb_dim_segments, C=62, T=200),
    "DE":       lambda: models.glfnet_mlp(out_dim=emb_dim_DE, emb_dim=emb_dim_DE, input_dim=62*5),
    "PSD":      lambda: models.glfnet_mlp(out_dim=emb_dim_PSD, emb_dim=emb_dim_PSD, input_dim=62*5),
}

# ==========================================
# Fusion model wrapper
# ==========================================
class FusionNet(nn.Module):
    def __init__(self, encoders):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.total_dim = sum([e.out_dim for e in encoders.values()])
    def forward(self, inputs):
        feats = []
        for name, enc in self.encoders.items():
            feats.append(enc(inputs[name]))
        return torch.cat(feats, dim=-1)

# ==========================================
# Semantic MLP (exact user architecture)
# ==========================================
class SemanticMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 10000), nn.ReLU(),
            nn.Linear(10000, 10000), nn.ReLU(),
            nn.Linear(10000, 10000), nn.ReLU(),
            nn.Linear(10000, 10000), nn.ReLU(),
            nn.Linear(10000, 77*768)   # final mapping to CLIP space
        )
    def forward(self, x):
        return self.mlp(x)

# ==========================================
# Combined model
# ==========================================
class SemanticPredictor(nn.Module):
    def __init__(self, encoder, input_dim):
        super().__init__()
        self.encoder = encoder
        self.head = SemanticMLP(input_dim)
    def forward(self, x):
        feats = self.encoder(x)
        return self.head(feats)

# ==========================================
# Dataset
# ==========================================
class FusionDataset(data.Dataset):
    def __init__(self, features_dict, clip_targets):
        self.features = features_dict
        self.targets  = clip_targets
    def __len__(self): return len(self.targets)
    def __getitem__(self, idx):
        return {ft: torch.tensor(self.features[ft][idx], dtype=torch.float32) for ft in self.features}, \
               torch.tensor(self.targets[idx], dtype=torch.float32)

def Get_Dataloader(features, targets, istrain, batch_size, fusion=False):
    if fusion:
        return data.DataLoader(FusionDataset(features, targets), batch_size, shuffle=istrain)
    else:
        features = torch.tensor(features, dtype=torch.float32)
        targets  = torch.tensor(targets, dtype=torch.float32)
        return data.DataLoader(data.TensorDataset(features, targets), batch_size, shuffle=istrain)

# ==========================================
# Training loop
# ==========================================
def train(net, train_iter, val_iter, test_iter, num_epochs, lr, device, fusion=False,
          subname="subject"):
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    mse_loss  = nn.MSELoss()
    cos_loss  = nn.CosineEmbeddingLoss()

    best_val, best_state = 1e12, None
    for epoch in range(num_epochs):
        net.train()
        total_loss = 0
        for X, y in train_iter:
            if fusion: X = {ft: X[ft].to(device) for ft in X}
            else: X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = net(X)

            if LOSS_TYPE == "mse":
                loss = mse_loss(y_hat, y)
            elif LOSS_TYPE == "cosine":
                target = torch.ones(y_hat.size(0), device=device)
                loss = cos_loss(y_hat, y, target)
            elif LOSS_TYPE == "mse+cosine":
                target = torch.ones(y_hat.size(0), device=device)
                loss = mse_loss(y_hat, y) + cos_loss(y_hat, y, target)

            if USE_VAR_REG:
                loss -= VAR_LAMBDA * torch.var(y_hat, dim=0).mean()

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)

        # validation
        val_mse = evaluate_mse(net, val_iter, device, fusion)
        if val_mse < best_val:
            best_val, best_state = val_mse, net.state_dict()

        if epoch % 3 == 0:
            test_mse, test_cos, class_var = evaluate(net, test_iter, device, fusion)
            print(f"[{epoch+1}] train_loss={total_loss/len(train_iter.dataset):.4f}, "
                  f"val_mse={val_mse:.4f}, test_mse={test_mse:.4f}, "
                  f"test_cos={test_cos:.4f}, class_var={class_var:.4f}")

    if best_state:
        net.load_state_dict(best_state)
        save_dir = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            "state_dict": net.state_dict(),
            "feature_type": FEATURE_TYPE,
            "input_dim": net.head.mlp[0].in_features,
        }, os.path.join(save_dir, f"semantic_predictor_{subname.replace('.npy','')}.pt"))
    return net

# ==========================================
# Evaluation
# ==========================================
def evaluate_mse(net, data_iter, device, fusion=False):
    net.eval()
    mse_loss = nn.MSELoss(reduction="sum")
    total, count, dim = 0, 0, None
    with torch.no_grad():
        for X, y in data_iter:
            if fusion: X = {ft: X[ft].to(device) for ft in X}
            else: X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            total += mse_loss(y_hat, y).item()
            count += y.size(0)
            dim = y.size(1)
    return total / (count * dim)

def evaluate(net, data_iter, device, fusion=False):
    net.eval()
    cos = nn.CosineSimilarity(dim=-1)
    total_mse, total_cos, count, dim = 0, 0, 0, None
    preds_all, targets_all = [], []
    with torch.no_grad():
        for X, y in data_iter:
            if fusion: X = {ft: X[ft].to(device) for ft in X}
            else: X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            total_mse += F.mse_loss(y_hat, y, reduction="sum").item()
            total_cos += cos(y_hat, y).mean().item() * y.size(0)
            preds_all.append(y_hat.cpu().numpy())
            targets_all.append(y.cpu().numpy())
            count += y.size(0)
            dim = y.size(1)
    preds_all = np.concatenate(preds_all, axis=0)

    # --- Between-class variance ---
    try:
        preds_all = preds_all.reshape(40, 5, 2, -1)  # 40 classes × 5 clips × 2 windows × emb_dim
        preds_all = preds_all.mean(axis=(1, 2))      # average across clips+windows → [40, emb_dim]
        class_means = preds_all
        overall_mean = class_means.mean(axis=0)
        # variance across class means
        class_var = np.mean(np.sum((class_means - overall_mean)**2, axis=1))
    except:
        class_var = 0.0

    return total_mse / (count * dim), total_cos / count, class_var

# ==========================================
# Main
# ==========================================
clip_embeddings = np.load(CLIP_EMB_PATH)   # [7,40,5,77,768]
clip_embeddings = np.expand_dims(clip_embeddings, axis=3)   # [7,40,5,1,77,768]
clip_embeddings = np.repeat(clip_embeddings, 2, axis=3)     # [7,40,5,2,77,768]
clip_embeddings = clip_embeddings.reshape(-1, 77*768)

samples_per_block = clip_embeddings.shape[0] // 7

if FEATURE_TYPE == "fusion":
    sub_list = os.listdir(FEATURE_PATHS["DE"]) if USE_ALL_SUBJECTS else ["sub1.npy"]
else:
    sub_list = os.listdir(FEATURE_PATHS[FEATURE_TYPE]) if USE_ALL_SUBJECTS else ["sub1.npy"]

for subname in sub_list:
    print(f"\n=== Training subject {subname} ===")
    if FEATURE_TYPE == "fusion":
        raw_data = {ft: np.load(os.path.join(FEATURE_PATHS[ft], subname)) for ft in ["segments", "DE", "PSD"]}
    else:
        raw_data = np.load(os.path.join(FEATURE_PATHS[FEATURE_TYPE], subname))

    def reshape(ft, arr):
        if ft in ["DE", "PSD"]:
            return rearrange(arr, "a b c d e f -> a (b c d) e f")
        elif ft == "segments":
            return rearrange(arr, "a b c d (w t) -> a (b c w) d t", w=2)
        else:
            raise ValueError

    All_train = {ft: reshape(ft, raw_data[ft]) for ft in raw_data} if FEATURE_TYPE=="fusion" else reshape(FEATURE_TYPE, raw_data)

    # --- One-fold split: 5 train, 1 val, 1 test ---
    train_blocks = [0,1,2,3,4]
    val_block    = 5
    test_block   = 6

    if FEATURE_TYPE == "fusion":
        train_data, val_data, test_data = {}, {}, {}
        for ft in All_train:
            train_data[ft] = np.concatenate([All_train[ft][i] for i in train_blocks])
            val_data[ft]   = All_train[ft][val_block]
            test_data[ft]  = All_train[ft][test_block]

        train_targets = np.concatenate([clip_embeddings[i*samples_per_block:(i+1)*samples_per_block] for i in train_blocks])
        val_targets   = clip_embeddings[val_block*samples_per_block:(val_block+1)*samples_per_block]
        test_targets  = clip_embeddings[test_block*samples_per_block:(test_block+1)*samples_per_block]

        for ft in train_data:
            tr = train_data[ft].reshape(train_data[ft].shape[0], -1)
            va = val_data[ft].reshape(val_data[ft].shape[0], -1)
            te = test_data[ft].reshape(test_data[ft].shape[0], -1)
            scaler = StandardScaler()
            tr = scaler.fit_transform(tr); va = scaler.transform(va); te = scaler.transform(te)
            if ft == "segments":
                train_data[ft] = tr.reshape(-1, 1, C, 200)
                val_data[ft]   = va.reshape(-1, 1, C, 200)
                test_data[ft]  = te.reshape(-1, 1, C, 200)
            else:
                train_data[ft] = tr.reshape(-1, C, T)
                val_data[ft]   = va.reshape(-1, C, T)
                test_data[ft]  = te.reshape(-1, C, T)

        train_iter = Get_Dataloader(train_data, train_targets, True, batch_size, fusion=True)
        val_iter   = Get_Dataloader(val_data,   val_targets,   False, batch_size, fusion=True)
        test_iter  = Get_Dataloader(test_data,  test_targets,  False, batch_size, fusion=True)

        encoders = {ft: MODEL_MAP[ft]() for ft in All_train}
        encoder  = FusionNet(encoders)
        input_dim = encoder.total_dim

    else:
        train_data = np.concatenate([All_train[i] for i in train_blocks])
        val_data   = All_train[val_block]
        test_data  = All_train[test_block]

        train_targets = np.concatenate([clip_embeddings[i*samples_per_block:(i+1)*samples_per_block] for i in train_blocks])
        val_targets   = clip_embeddings[val_block*samples_per_block:(val_block+1)*samples_per_block]
        test_targets  = clip_embeddings[test_block*samples_per_block:(test_block+1)*samples_per_block]

        tr = train_data.reshape(train_data.shape[0], -1)
        va = val_data.reshape(val_data.shape[0], -1)
        te = test_data.reshape(test_data.shape[0], -1)
        scaler = StandardScaler()
        tr = scaler.fit_transform(tr); va = scaler.transform(va); te = scaler.transform(te)

        if FEATURE_TYPE == "segments":
            train_data = tr.reshape(-1, 1, C, 200)
            val_data   = va.reshape(-1, 1, C, 200)
            test_data  = te.reshape(-1, 1, C, 200)
        else:
            train_data = tr.reshape(-1, C, T)
            val_data   = va.reshape(-1, C, T)
            test_data  = te.reshape(-1, C, T)

        train_iter = Get_Dataloader(train_data, train_targets, True, batch_size)
        val_iter   = Get_Dataloader(val_data,   val_targets,   False, batch_size)
        test_iter  = Get_Dataloader(test_data,  test_targets,  False, batch_size)

        encoder   = MODEL_MAP[FEATURE_TYPE]()
        input_dim = emb_dim_DE if FEATURE_TYPE=="DE" else emb_dim_PSD if FEATURE_TYPE=="PSD" else emb_dim_segments

    modelnet = SemanticPredictor(encoder, input_dim)
    modelnet = train(modelnet, train_iter, val_iter, test_iter, num_epochs, lr,
                     run_device, fusion=(FEATURE_TYPE=="fusion"), subname=subname)
