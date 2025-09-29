# ==========================================
# EEG → Dynamic Predictor (crossentropy only)
# ==========================================
import os, numpy as np, torch, joblib
from torch import nn
from torch.utils import data
from sklearn.preprocessing import StandardScaler
from einops import rearrange
import models

# ==========================================
# Config
# ==========================================
batch_size    = 256
num_epochs    = 100
lr            = 0.0005
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
FEATURE_TYPES    = ["DE", "PSD", "segments"]

# default is subject 1 only; set to True to use all subjects in folder
USE_ALL_SUBJECTS = False
subject_name     = "sub1.npy"

# restrict to certain classes (0–39); set to None for all
CLASS_SUBSET     = [1, 10, 12, 16, 19, 23, 25, 31, 34, 39]

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
    if ft == "segments":
        base = models.glfnet(out_dim=2 if return_logits else emb_dim_segments,
                             emb_dim=emb_dim_segments, C=C, T=200)
        return base if return_logits else (base, emb_dim_segments)
    elif ft == "DE":
        base = models.glfnet_mlp(out_dim=2 if return_logits else emb_dim_DE,
                                 emb_dim=emb_dim_DE, input_dim=C*T)
        return base if return_logits else (base, emb_dim_DE)
    elif ft == "PSD":
        base = models.glfnet_mlp(out_dim=2 if return_logits else emb_dim_PSD,
                                 emb_dim=emb_dim_PSD, input_dim=C*T)
        return base if return_logits else (base, emb_dim_PSD)
    else:
        raise ValueError(f"Unknown feature type {ft}")


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
    def __init__(self, features_dict, labels):
        self.features = features_dict
        self.labels   = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        X = {ft: torch.tensor(self.features[ft][idx], dtype=torch.float32)
             for ft in self.features}
        return X, torch.tensor(self.labels[idx], dtype=torch.long)

def Get_Dataloader(features, labels, istrain, batch_size, multi=False):
    if multi:
        return data.DataLoader(FusionDataset(features, labels), batch_size, shuffle=istrain)
    else:
        feats = torch.tensor(features, dtype=torch.float32)
        lbls  = torch.tensor(labels, dtype=torch.long)
        return data.DataLoader(data.TensorDataset(feats, lbls), batch_size, shuffle=istrain)


# ==========================================
# Training loop
# ==========================================
def train(net, train_iter, val_iter, test_iter, num_epochs, lr, device,
          subname="subject", scalers=None):
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    
    # scheduler (per batch, like authors)
    if SCHEDULER_TYPE.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs * len(train_iter)
        )
    else:
        scheduler = None

    ce_loss   = nn.CrossEntropyLoss()

    best_val_acc, best_state = 0.0, None
    for epoch in range(num_epochs):
        net.train()
        total_loss = 0
        for X, y in train_iter:
            if isinstance(X, dict):
                X = {ft: X[ft].to(device) for ft in X}
            else:
                X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = net(X)   # (batch,2) logits
            loss  = ce_loss(y_hat, y)

            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            total_loss += loss.item() * y.size(0)

        val_loss, val_acc = evaluate(net, val_iter, device)
        if val_acc > best_val_acc:
            best_val_acc, best_state = val_acc, net.state_dict()

        if epoch % 3 == 0:
            test_loss, test_acc = evaluate(net, test_iter, device)
            print(f"[{epoch+1}] train_loss={total_loss/len(train_iter.dataset):.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}, "
                  f"test_loss={test_loss:.4f}, test_acc={test_acc:.3f}")

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
def evaluate(net, data_iter, device):
    net.eval()
    ce_loss = nn.CrossEntropyLoss(reduction="sum")
    total_loss, count, correct = 0, 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, dict):
                X = {ft: X[ft].to(device) for ft in X}
            else:
                X = X.to(device)
            y = y.to(device)
            y_hat = net(X)   # (batch,2)
            total_loss += ce_loss(y_hat, y).item()
            pred_cls = torch.argmax(y_hat, dim=-1)
            correct += (pred_cls.cpu() == y.cpu()).sum().item()
            count += y.size(0)
    return total_loss / count, correct / count


# ==========================================
# Helpers
# ==========================================
def load_subject_data(subname, feature_types):
    feats, scalers = {}, {}
    for ft in feature_types:
        path = os.path.join(FEATURE_PATHS[ft], subname)
        arr = np.load(path)

        if ft in ["DE","PSD"]:
            # keep (N, C, T) for the model
            arr = arr.reshape(-1, C, T)
            flat = arr.reshape(arr.shape[0], -1)              # (N, C*T)
            scaler = StandardScaler().fit(flat)               # fit on 2D
            arr = scaler.transform(flat).reshape(-1, C, T)    # scale, then restore 3D

        elif ft == "segments":
            # expect (blocks, classes, trials, C, 400)
            arr = rearrange(arr, "a b c d (w t) -> (a b c w) d t", w=2, t=200)  # (N*2, 62, 200)
            flat = arr.reshape(arr.shape[0], -1)                                 # (N*2, 12400)
            scaler = StandardScaler().fit(flat)
            arr = scaler.transform(flat).reshape(-1, 1, C, 200)                  # (N*2, 1, 62, 200)

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

    # Binarize OFS into 0/1 class labels
    y_cls = (ofs_all > threshold).astype(int).flatten()

    sub_list = os.listdir(FEATURE_PATHS[FEATURE_TYPES[0]]) if USE_ALL_SUBJECTS else [subject_name]
    for subname in sub_list:
        print(f"\n=== Training subject {subname} with {FEATURE_TYPES} ===")
        features, scalers = load_subject_data(subname, FEATURE_TYPES)

        if CLASS_SUBSET is not None:
            mask = np.isin(labels_all, CLASS_SUBSET)
            for ft in features: features[ft] = features[ft][mask]
            y_cls, labels_all = y_cls[mask], labels_all[mask]

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
                                        y_cls[train_idx], True, batch_size, multi=True)
            val_iter   = Get_Dataloader({ft: features[ft][val_idx] for ft in FEATURE_TYPES},
                                        y_cls[val_idx], False, batch_size, multi=True)
            test_iter  = Get_Dataloader({ft: features[ft][test_idx] for ft in FEATURE_TYPES},
                                        y_cls[test_idx], False, batch_size, multi=True)
        else:
            # single feature: encoder already outputs logits
            ft = FEATURE_TYPES[0]
            modelnet = make_encoder(ft, return_logits=True)

            train_iter = Get_Dataloader(features[ft][train_idx], y_cls[train_idx], True, batch_size)
            val_iter   = Get_Dataloader(features[ft][val_idx],   y_cls[val_idx],   False, batch_size)
            test_iter  = Get_Dataloader(features[ft][test_idx],  y_cls[test_idx],  False, batch_size)

        modelnet = train(modelnet, train_iter, val_iter, test_iter,
                         num_epochs, lr, run_device,
                         subname=subname, scalers=scalers)
