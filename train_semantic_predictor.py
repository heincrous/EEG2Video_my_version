# ==========================================
# EEG → CLIP semantic predictor (with frozen encoders or fusion)
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

# Loss type: "mse", "cosine", "mse+cosine", "contrastive"
LOSS_TYPE        = "mse"

USE_VAR_REG = True
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
# Semantic MLP
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
# Dataset wrappers
# ==========================================
class FusionDataset(data.Dataset):
    def __init__(self, features_dict, clip_targets, class_labels):
        self.features = features_dict
        self.targets  = clip_targets
        self.labels   = class_labels
    def __len__(self): return len(self.targets)
    def __getitem__(self, idx):
        return {ft: torch.tensor(self.features[ft][idx], dtype=torch.float32) for ft in self.features}, \
               torch.tensor(self.targets[idx], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.long)

class SimpleDataset(data.Dataset):
    def __init__(self, features, clip_targets, class_labels):
        self.features = features
        self.targets  = clip_targets
        self.labels   = class_labels
    def __len__(self): return len(self.targets)
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), \
               torch.tensor(self.targets[idx], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.long)

def Get_Dataloader(features, targets, labels, istrain, batch_size, fusion=False):
    if fusion:
        return data.DataLoader(FusionDataset(features, targets, labels), batch_size, shuffle=istrain)
    else:
        return data.DataLoader(SimpleDataset(features, targets, labels), batch_size, shuffle=istrain)

# ==========================================
# Contrastive loss helper
# ==========================================
def contrastive_loss_fn(y_hat, y, margin=1.0):
    y_hat = F.normalize(y_hat, dim=-1)
    y     = F.normalize(y, dim=-1)
    sim_matrix = torch.matmul(y_hat, y.t())
    pos = torch.diag(sim_matrix)
    loss = torch.mean(F.relu(margin - pos[:, None] + sim_matrix))
    return loss

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
        for batch in train_iter:
            if fusion:
                X, y, y_class = batch
                X = {ft: X[ft].to(device) for ft in X}
            else:
                X, y, y_class = batch
                X = X.to(device)
            y       = y.to(device)
            y_class = y_class.to(device)

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
                pred_vars, target_vars = [], []
                for cls in torch.unique(y_class):
                    mask = (y_class == cls)
                    if mask.sum() > 1:
                        pred_vars.append(torch.var(y_hat[mask], dim=0).mean())
                        target_vars.append(torch.var(y[mask], dim=0).mean())
                if pred_vars:
                    pred_var   = torch.stack(pred_vars).mean()
                    target_var = torch.stack(target_vars).mean()
                    var_ratio  = pred_var / (target_var + 1e-8)
                    loss -= VAR_LAMBDA * var_ratio

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)

        val_mse = evaluate_mse(net, val_iter, device, fusion)
        if val_mse < best_val:
            best_val, best_state = val_mse, net.state_dict()

        if epoch % 3 == 0:
            test_mse, test_cos, fisher_score = evaluate(net, test_iter, device, fusion)
            print(f"[{epoch+1}] train_loss={total_loss/len(train_iter.dataset):.4f}, "
                  f"val_mse={val_mse:.4f}, test_mse={test_mse:.4f}, "
                  f"test_cos={test_cos:.4f}, fisher_score={fisher_score:.4f}")

    if best_state:
        net.load_state_dict(best_state)
        save_dir = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_predictor"
        os.makedirs(save_dir, exist_ok=True)
        fname = f"semantic_predictor_{FEATURE_TYPE}_{subname.replace('.npy','')}.pt"
        torch.save({
            "state_dict": net.state_dict(),
            "feature_type": FEATURE_TYPE,
            "input_dim": net.head.mlp[0].in_features,
        }, os.path.join(save_dir, fname))
    return net

# ==========================================
# Evaluation
# ==========================================
def evaluate_mse(net, data_iter, device, fusion=False):
    net.eval()
    mse_loss = nn.MSELoss(reduction="sum")
    total, count, dim = 0, 0, None
    with torch.no_grad():
        for batch in data_iter:
            if fusion:
                X, y, _ = batch
                X = {ft: X[ft].to(device) for ft in X}
            else:
                X, y, _ = batch
                X = X.to(device)
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
        for batch in data_iter:
            if fusion:
                X, y, _ = batch
                X = {ft: X[ft].to(device) for ft in X}
            else:
                X, y, _ = batch
                X = X.to(device)
            y = y.to(device)
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
# Main
# ==========================================
clip_embeddings = np.load(CLIP_EMB_PATH)
clip_embeddings = np.expand_dims(clip_embeddings, axis=3)
clip_embeddings = np.repeat(clip_embeddings, 2, axis=3)
clip_embeddings = clip_embeddings.reshape(-1, 77*768)

labels_block = np.repeat(np.arange(40), 5*2)
labels_all   = np.tile(labels_block, 7)
samples_per_block = clip_embeddings.shape[0] // 7

if FEATURE_TYPE == "fusion":
    sub_list = os.listdir(FEATURE_PATHS["DE"]) if USE_ALL_SUBJECTS else ["sub1.npy"]
else:
    sub_list = os.listdir(FEATURE_PATHS[FEATURE_TYPE]) if USE_ALL_SUBJECTS else ["sub1.npy"]

for subname in sub_list:
    print(f"\n=== Training subject {subname} ===")
    if FEATURE_TYPE == "fusion":
        # --- Fusion checkpoint naming ---
        fusion_name = f"classifier_fusion_{'_'.join(sorted(MODEL_MAP.keys()))}_{subname.replace('.npy','')}.pt"
        fusion_ckpt = os.path.join("/content/drive/MyDrive/EEG2Video_checkpoints/classifier_checkpoints", fusion_name)

        if os.path.exists(fusion_ckpt):
            print(f"Loading whole fusion encoder: {fusion_ckpt}")
            encoder = FusionNet({
                "DE": models.glfnet_mlp(out_dim=emb_dim_DE, emb_dim=emb_dim_DE, input_dim=62*5),
                "PSD": models.glfnet_mlp(out_dim=emb_dim_PSD, emb_dim=emb_dim_PSD, input_dim=62*5),
                "segments": models.glfnet(out_dim=emb_dim_segments, emb_dim=emb_dim_segments, C=62, T=200)
            })
            enc_state = torch.load(fusion_ckpt, map_location=run_device)
            encoder.load_state_dict(enc_state["state_dict"] if "state_dict" in enc_state else enc_state, strict=False)
            for p in encoder.parameters(): 
                p.requires_grad = False
            input_dim = encoder.total_dim

        else:
            # fallback: load individual encoders
            encoders = {ft: MODEL_MAP[ft]() for ft in ["DE", "PSD", "segments"]}
            for ft, enc in encoders.items():
                ckpt_name = f"classifier_{ft}_{subname.replace('.npy','')}.pt"
                ckpt_path = os.path.join("/content/drive/MyDrive/EEG2Video_checkpoints/classifier_checkpoints", ckpt_name)
                if os.path.exists(ckpt_path):
                    print(f"Loading checkpoint for {ft}: {ckpt_path}")
                    enc_state = torch.load(ckpt_path, map_location=run_device)
                    enc.load_state_dict(enc_state["state_dict"] if "state_dict" in enc_state else enc_state, strict=False)
                for p in enc.parameters():
                    p.requires_grad = False
            encoder = FusionNet(encoders)
            input_dim = encoder.total_dim

    else:
        encoder = MODEL_MAP[FEATURE_TYPE]()
        ckpt_name = f"classifier_{FEATURE_TYPE}_{subname.replace('.npy','')}.pt"
        ckpt_path = os.path.join("/content/drive/MyDrive/EEG2Video_checkpoints/classifier_checkpoints", ckpt_name)

        if os.path.exists(ckpt_path):
            print(f"Loading encoder checkpoint: {ckpt_path}")
            enc_state = torch.load(ckpt_path, map_location=run_device)
            encoder.load_state_dict(enc_state["state_dict"] if "state_dict" in enc_state else enc_state, strict=False)
        for p in encoder.parameters():
            p.requires_grad = False

        input_dim = emb_dim_DE if FEATURE_TYPE=="DE" else emb_dim_PSD if FEATURE_TYPE=="PSD" else emb_dim_segments
