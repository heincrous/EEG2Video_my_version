# ==========================================
# EEG → Dynamic Predictor (clean single vs fusion)
# ==========================================
import os, numpy as np, torch, joblib
from torch import nn
from torch.utils import data
from sklearn.preprocessing import StandardScaler
from einops import rearrange
import torch.nn.functional as F
import models

# ==========================================
# Config
# ==========================================
batch_size    = 64
num_epochs    = 100
lr            = 0.001
run_device    = "cuda"

# EEG dimensions
C, T = 62, 5
emb_dim_segments = 512
emb_dim_DE       = 256
emb_dim_PSD      = 256

# optimizer: "adam" or "adamw"; set WEIGHT_DECAY=0 for Adam
WEIGHT_DECAY   = 0

# scheduler: "cosine" or "constant"
SCHEDULER_TYPE = "cosine"

# choose: ["segments"], ["DE"], ["PSD"], ["segments","DE"], ["DE","PSD"], ["segments","DE","PSD"]
FEATURE_TYPES    = ["DE"]

USE_ALL_SUBJECTS = False
subject_name     = "sub1.npy"

# restrict to certain classes (0–39); set to None for all
CLASS_SUBSET     = [1, 10, 12, 16, 19, 23, 25, 31, 34, 39]

# loss type: "mse", "cosine", "mse+cosine", "contrastive", "crossentropy"
LOSS_TYPE        = "mse"

USE_VAR_REG = False
VAR_LAMBDA  = 0.01

FEATURE_PATHS = {
    "segments": "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments",
    "DE":       "/content/drive/MyDrive/EEG2Video_data/processed/EEG_DE_1per1s",
    "PSD":      "/content/drive/MyDrive/EEG2Video_data/processed/EEG_PSD_1per1s",
}
OFS_PATH         = "/content/drive/MyDrive/EEG2Video_data/processed/meta-info/All_video_optical_flow_score_byclass.npy"
DYNPRED_CKPT_DIR = "/content/drive/MyDrive/EEG2Video_checkpoints/dynamic_checkpoints"


# ==========================================
# Encoders
# ==========================================
def make_encoder(ft, return_logits=False):
    """Build encoder. If return_logits=True, attach classifier head."""
    if ft == "segments":
        base = models.glfnet(out_dim=emb_dim_segments, emb_dim=emb_dim_segments, C=C, T=200)
        emb_dim = emb_dim_segments
    elif ft == "DE":
        base = models.glfnet_mlp(out_dim=emb_dim_DE, emb_dim=emb_dim_DE, input_dim=C*T)
        emb_dim = emb_dim_DE
    elif ft == "PSD":
        base = models.glfnet_mlp(out_dim=emb_dim_PSD, emb_dim=emb_dim_PSD, input_dim=C*T)
        emb_dim = emb_dim_PSD
    else:
        raise ValueError(f"Unknown feature type {ft}")

    if return_logits:
        return nn.Sequential(base, nn.Linear(emb_dim, 2))  # encoder + classifier
    else:
        return base, emb_dim  # just embeddings


# ==========================================
# Fusion classifier (embedding-level)
# ==========================================
class FusionNet(nn.Module):
    def __init__(self, encoders, emb_dims):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        total_dim     = sum(emb_dims.values())
        self.classifier = nn.Linear(total_dim, 2)  # 2 logits
    def forward(self, inputs):
        feats = []
        for name, enc in self.encoders.items():
            feats.append(enc(inputs[name]))
        fused = torch.cat(feats, dim=-1)
        return self.classifier(fused)


# ==========================================
# Dataset
# ==========================================
class FusionDataset(data.Dataset):
    def __init__(self, features_dict, targets, labels):
        self.features = features_dict
        self.targets  = targets
        self.labels   = labels
    def __len__(self): return len(self.targets)
    def __getitem__(self, idx):
        X = {ft: torch.tensor(self.features[ft][idx], dtype=torch.float32)
             for ft in self.features}
        return X, torch.tensor(self.targets[idx], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.long)

def Get_Dataloader(features, targets, labels, istrain, batch_size, multi=False):
    if multi:
        return data.DataLoader(FusionDataset(features, targets, labels), batch_size, shuffle=istrain)
    else:
        feats = torch.tensor(features, dtype=torch.float32)
        tgts  = torch.tensor(targets, dtype=torch.float32)
        lbls  = torch.tensor(labels, dtype=torch.long)
        return data.DataLoader(data.TensorDataset(feats, tgts, lbls), batch_size, shuffle=istrain)


# ==========================================
# Losses
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
          subname="subject", scalers=None, threshold=None):
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    
    # scheduler (per batch, like authors)
    if SCHEDULER_TYPE.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs * len(train_iter)
        )
    else:
        scheduler = None

    mse_loss  = nn.MSELoss()
    cos_loss  = nn.CosineEmbeddingLoss()
    ce_loss   = nn.CrossEntropyLoss()

    best_val, best_state = 1e12, None
    for epoch in range(num_epochs):
        net.train()
        total_loss = 0
        for X, y, _ in train_iter:
            if isinstance(X, dict):
                X = {ft: X[ft].to(device) for ft in X}
            else:
                X = X.to(device)
            y = y.to(device).view(-1,1)

            optimizer.zero_grad()
            y_hat = net(X)   # (batch,2) logits

            if LOSS_TYPE == "mse":
                probs = torch.softmax(y_hat, dim=-1)[:,1].view(-1,1)
                loss = mse_loss(probs, y)
            elif LOSS_TYPE == "cosine":
                probs = torch.softmax(y_hat, dim=-1)[:,1].view(-1,1)
                target = torch.ones(probs.size(0), device=device)
                loss = cos_loss(probs, y, target)
            elif LOSS_TYPE in ["mse+cosine", "cosine+mse"]:
                probs = torch.softmax(y_hat, dim=-1)[:,1].view(-1,1)
                target = torch.ones(probs.size(0), device=device)
                loss = mse_loss(probs, y) + cos_loss(probs, y, target)
            elif LOSS_TYPE == "contrastive":
                probs = torch.softmax(y_hat, dim=-1)[:,1].view(-1,1)
                loss = contrastive_loss_fn(probs, y)
            elif LOSS_TYPE == "crossentropy":
                y_cls = (y > threshold).long().view(-1)  # 0/1 labels
                loss  = ce_loss(y_hat, y_cls)
            else:
                raise ValueError(f"Unknown LOSS_TYPE {LOSS_TYPE}")

            if USE_VAR_REG:
                loss -= VAR_LAMBDA * torch.var(y_hat, dim=0).mean()

            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            total_loss += loss.item() * y.size(0)

        val_mse, val_acc = evaluate_with_classification(net, val_iter, device, threshold)
        if val_mse < best_val:
            best_val, best_state = val_mse, net.state_dict()

        if epoch % 3 == 0:
            test_mse, test_acc = evaluate_with_classification(net, test_iter, device, threshold)
            print(f"[{epoch+1}] train_loss={total_loss/len(train_iter.dataset):.4f}, "
                  f"val_mse={val_mse:.4f}, val_acc={val_acc:.3f}, "
                  f"test_mse={test_mse:.4f}, test_acc={test_acc:.3f}")

    if best_state:
        net.load_state_dict(best_state)
        os.makedirs(DYNPRED_CKPT_DIR, exist_ok=True)
        subset_tag = "" if CLASS_SUBSET is None else "_subset" + "-".join(str(c) for c in CLASS_SUBSET)
        model_name = f"dynpredictor_{'_'.join(FEATURE_TYPES)}_{subname.replace('.npy','')}{subset_tag}.pt"
        torch.save({"state_dict": net.state_dict()},
                os.path.join(DYNPRED_CKPT_DIR, model_name))
        print(f"Saved checkpoint: {model_name}")

        for ft, sc in scalers.items():
            scaler_name = f"scaler_{ft}_{subname.replace('.npy','')}{subset_tag}.pkl"
            joblib.dump(sc, os.path.join(DYNPRED_CKPT_DIR, scaler_name))
            print(f"Saved scaler: {scaler_name}")
    return net


# ==========================================
# Evaluation
# ==========================================
def evaluate_with_classification(net, data_iter, device, threshold):
    net.eval()
    mse_loss = nn.MSELoss(reduction="sum")
    total_mse, count, correct = 0, 0, 0
    with torch.no_grad():
        for X, y, _ in data_iter:
            if isinstance(X, dict):
                X = {ft: X[ft].to(device) for ft in X}
            else:
                X = X.to(device)
            y = y.to(device).view(-1,1)
            y_hat = net(X)   # (batch,2)
            probs = torch.softmax(y_hat, dim=-1)[:,1].view(-1,1)
            total_mse += mse_loss(probs, y).item()
            count += y.size(0)
            y_cls = (y > threshold).long().view(-1)
            pred_cls = torch.argmax(y_hat, dim=-1)
            correct += (pred_cls.cpu() == y_cls.cpu()).sum().item()
    return total_mse / count, correct / count


# ==========================================
# Helpers
# ==========================================
def load_subject_data(subname, feature_types):
    feats, scalers = {}, {}
    for ft in feature_types:
        path = os.path.join(FEATURE_PATHS[ft], subname)
        arr = np.load(path)
        if ft in ["DE","PSD"]: arr = arr.reshape(-1, C*T)
        elif ft == "segments": arr = rearrange(arr, "a b c d (w t) -> (a b c w) (d t)", w=2, t=200)
        scaler = StandardScaler().fit(arr)
        arr = scaler.transform(arr)
        scalers[ft] = scaler
        feats[ft] = arr
    return feats, scalers


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    ofs_all = np.load(OFS_PATH).reshape(-1,1)    # (1400,1)
    labels_all = np.tile(np.arange(40).repeat(5), 7)  # (1400,)
    threshold = np.median(ofs_all)

    # Duplicate to match 2 EEG windows per clip → 2800 samples
    ofs_all    = np.repeat(ofs_all, 2, axis=0)     # (2800,1)
    labels_all = np.repeat(labels_all, 2, axis=0)  # (2800,)

    sub_list = os.listdir(FEATURE_PATHS[FEATURE_TYPES[0]]) if USE_ALL_SUBJECTS else [subject_name]
    for subname in sub_list:
        print(f"\n=== Training subject {subname} with {FEATURE_TYPES} ===")
        features, scalers = load_subject_data(subname, FEATURE_TYPES)

        if CLASS_SUBSET is not None:
            mask = np.isin(labels_all, CLASS_SUBSET)
            for ft in features: features[ft] = features[ft][mask]
            ofs_all, labels_all = ofs_all[mask], labels_all[mask]

        samples_per_block = (len(CLASS_SUBSET) if CLASS_SUBSET else 40) * 5 * 2
        train_idx = np.arange(0, 5*samples_per_block)
        val_idx   = np.arange(5*samples_per_block, 6*samples_per_block)
        test_idx  = np.arange(6*samples_per_block, 7*samples_per_block)

        if len(FEATURE_TYPES) > 1:
            # multi-feature fusion: encoders output embeddings only
            encoders = {}
            emb_dims = {}
            for ft in FEATURE_TYPES:
                enc, dim = make_encoder(ft, return_logits=False)
                encoders[ft] = enc
                emb_dims[ft] = dim
            modelnet = FusionNet(encoders, emb_dims)

            train_iter = Get_Dataloader({ft: features[ft][train_idx] for ft in FEATURE_TYPES},
                                        ofs_all[train_idx], labels_all[train_idx], True, batch_size, multi=True)
            val_iter   = Get_Dataloader({ft: features[ft][val_idx] for ft in FEATURE_TYPES},
                                        ofs_all[val_idx], labels_all[val_idx], False, batch_size, multi=True)
            test_iter  = Get_Dataloader({ft: features[ft][test_idx] for ft in FEATURE_TYPES},
                                        ofs_all[test_idx], labels_all[test_idx], False, batch_size, multi=True)
        else:
            # single feature: encoder already outputs logits
            ft = FEATURE_TYPES[0]
            modelnet = make_encoder(ft, return_logits=True)

            train_iter = Get_Dataloader(features[ft][train_idx], ofs_all[train_idx], labels_all[train_idx], True, batch_size)
            val_iter   = Get_Dataloader(features[ft][val_idx],   ofs_all[val_idx],   labels_all[val_idx],   False, batch_size)
            test_iter  = Get_Dataloader(features[ft][test_idx],  ofs_all[test_idx],  labels_all[test_idx],  False, batch_size)

        modelnet = train(modelnet, train_iter, val_iter, test_iter,
                         num_epochs, lr, run_device,
                         subname=subname, scalers=scalers, threshold=threshold)
