# ==========================================
# EEG → Dynamic Predictor (crossentropy only, clip-level eval added)
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
num_epochs    = 200
lr            = 1e-3
run_device    = "cuda"

# EEG DE and PSD dimensions
C, T = 62, 5
emb_dim_segments = 512
emb_dim_windows  = 512 # cannot be used in fusion with other features
emb_dim_DE       = 256
emb_dim_PSD      = 256

# optimizer: "adam" or "adamw"; set WEIGHT_DECAY=0 for Adam
WEIGHT_DECAY   = 0.5

# scheduler: "cosine" or "constant"
SCHEDULER_TYPE = "cosine"

# choose: ["segments"], ["DE"], ["PSD"], ["segments","DE"], ["DE","PSD"], ["segments","DE","PSD"]
FEATURE_TYPES    = ["windows"]

# default is subject 1 only; set to True to use all subjects in folder
USE_ALL_SUBJECTS = False
subject_name     = "sub1.npy"

P = 0.5 # dropout prob

FEATURE_PATHS = {
    "segments": "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments",
    "DE":       "/content/drive/MyDrive/EEG2Video_data/processed/EEG_DE_1per1s",
    "PSD":      "/content/drive/MyDrive/EEG2Video_data/processed/EEG_PSD_1per1s",
    "windows":  "/content/drive/MyDrive/EEG2Video_data/processed/EEG_windows_200"
}
OFS_PATH         = "/content/drive/MyDrive/EEG2Video_data/processed/meta-info/All_video_optical_flow_score_byclass.npy"
DYNPRED_CKPT_DIR = "/content/drive/MyDrive/EEG2Video_checkpoints/dynamic_checkpoints"


# ==========================================
# Encoders
# ==========================================
def make_encoder(ft, return_logits=False, use_dropout=True, p=P):
    if ft == "segments":
        base = models.glfnet(out_dim=2 if return_logits else emb_dim_segments,
                             emb_dim=emb_dim_segments, C=C, T=200)
        return base if return_logits else (base, emb_dim_segments)
    
    elif ft == "windows":
        base = models.glfnet(out_dim=2 if return_logits else emb_dim_windows,
                            emb_dim=emb_dim_windows, C=C, T=200)
        return base if return_logits else (base, emb_dim_segments)

    elif ft in ["DE", "PSD"]:
        emb_dim = emb_dim_DE if ft == "DE" else emb_dim_PSD
        base = models.glfnet_mlp(out_dim=2 if return_logits else emb_dim,
                                 emb_dim=emb_dim, input_dim=C*T)
        if not return_logits and use_dropout:
            base = nn.Sequential(base, nn.Dropout(p))
        return base if return_logits else (base, emb_dim)

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
        self.classifier = nn.Linear(total_dim, 2)
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
# Evaluation (window + clip level)
# ==========================================
def evaluate(net, data_iter, device, clip_level=True):
    net.eval()
    ce_loss = nn.CrossEntropyLoss(reduction="sum")
    total_loss, count, correct = 0, 0, 0
    clip_correct, clip_count = 0, 0

    all_logits, all_labels = [], []
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, dict):
                X = {ft: X[ft].to(device) for ft in X}
            else:
                X = X.to(device)
            y = y.to(device)

            y_hat = net(X)  # (batch,2)
            total_loss += ce_loss(y_hat, y).item()

            # window-level
            pred_cls = torch.argmax(y_hat, dim=-1)
            correct += (pred_cls.cpu() == y.cpu()).sum().item()
            count += y.size(0)

            all_logits.append(y_hat.cpu())
            all_labels.append(y.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if clip_level:
        num_clips = all_logits.shape[0] // 2
        logits_clips = all_logits.view(num_clips, 2, -1).mean(dim=1)
        labels_clips = all_labels.view(num_clips, 2)[:, 0]
        pred_clips   = torch.argmax(logits_clips, dim=-1)
        clip_correct = (pred_clips == labels_clips).sum().item()
        clip_count   = num_clips

    window_loss = total_loss / count
    window_acc  = correct / count
    if clip_level:
        clip_acc = clip_correct / clip_count
        return window_loss, window_acc, clip_acc
    else:
        return window_loss, window_acc


# ==========================================
# Training loop
# ==========================================
def train(net, train_iter, val_iter, test_iter, num_epochs, lr, device,
          subname="subject", scalers=None):
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    if SCHEDULER_TYPE.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs * len(train_iter)
        )
    else:
        scheduler = None

    ce_loss   = nn.CrossEntropyLoss()

    best_val_clip_acc, best_state = 0.0, None
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
            y_hat = net(X)
            loss  = ce_loss(y_hat, y)

            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            total_loss += loss.item() * y.size(0)

        val_loss, val_acc, val_clip_acc = evaluate(net, val_iter, device, clip_level=True)
        if val_clip_acc > best_val_clip_acc:
            best_val_clip_acc, best_state = val_clip_acc, net.state_dict()

        if epoch % 3 == 0:
            test_loss, test_acc, test_clip_acc = evaluate(net, test_iter, device, clip_level=True)
            print(f"[{epoch+1}] train_loss={total_loss/len(train_iter.dataset):.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}, val_clip_acc={val_clip_acc:.3f}, "
                  f"test_loss={test_loss:.4f}, test_acc={test_acc:.3f}, test_clip_acc={test_clip_acc:.3f}")

    if best_state:
        net.load_state_dict(best_state)
        os.makedirs(DYNPRED_CKPT_DIR, exist_ok=True)
        subset_tag = ""
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
# Helpers
# ==========================================
# def prepare_features_with_scaler(subname, feature_types, train_idx, val_idx, test_idx):
#     feats, scalers = {}, {}
#     for ft in feature_types:
#         path = os.path.join(FEATURE_PATHS[ft], subname)
#         arr = np.load(path)

#         if ft in ["DE","PSD"]:
#             arr = arr.reshape(-1, C, T)
#             flat = arr.reshape(arr.shape[0], -1)
#             scaler = StandardScaler().fit(flat[train_idx])
#             arr = scaler.transform(flat).reshape(-1, C, T)

#         elif ft == "segments":
#             arr = rearrange(arr, "a b c d (w t) -> (a b c w) d t", w=2, t=200)
#             flat = arr.reshape(arr.shape[0], -1)
#             scaler = StandardScaler().fit(flat[train_idx])
#             arr = scaler.transform(flat).reshape(-1, 1, C, 200)

#         scalers[ft] = scaler
#         feats[ft] = arr
#     return feats, scalers

def prepare_features_with_scaler(subname, feature_types, train_idx, val_idx, test_idx):
    feats, scalers = {}, {}
    for ft in feature_types:
        path = os.path.join(FEATURE_PATHS[ft], subname)
        arr = np.load(path)

        if ft in ["DE","PSD"]:
            arr = arr.reshape(-1, C, T)
            flat = arr.reshape(arr.shape[0], -1)
            scaler = StandardScaler().fit(flat)   # fit on all data
            arr = scaler.transform(flat).reshape(-1, C, T)

        elif ft == "segments":
            arr = rearrange(arr, "a b c d (w t) -> (a b c w) d t", w=2, t=200)
            flat = arr.reshape(arr.shape[0], -1)
            scaler = StandardScaler().fit(flat)   # fit on all data
            arr = scaler.transform(flat).reshape(-1, 1, C, 200)
        
        elif ft == "windows":
            # shape: (7,40,5,3,62,200) → (N,62,200)
            arr = rearrange(arr, "a b c w d t -> (a b c w) d t")
            flat = arr.reshape(arr.shape[0], -1)
            scaler = StandardScaler().fit(flat)  # or fit(flat[train_idx]) if you want train-only
            arr = scaler.transform(flat).reshape(-1, 1, C, 200)

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

    ofs_all    = np.repeat(ofs_all, 2, axis=0)     # (2800,1)
    labels_all = np.repeat(labels_all, 2, axis=0)  # (2800,)
    y_cls = (ofs_all > threshold).astype(int).flatten()

    sub_list = os.listdir(FEATURE_PATHS[FEATURE_TYPES[0]]) if USE_ALL_SUBJECTS else [subject_name]
    for subname in sub_list:
        print(f"\n=== Training subject {subname} with {FEATURE_TYPES} ===")

        samples_per_block = 40 * 5 * 2
        train_idx = np.arange(0, 5*samples_per_block)
        val_idx   = np.arange(5*samples_per_block, 6*samples_per_block)
        test_idx  = np.arange(6*samples_per_block, 7*samples_per_block)

        features, scalers = prepare_features_with_scaler(subname, FEATURE_TYPES, train_idx, val_idx, test_idx)

        y_cls_sub    = y_cls
        labels_sub   = labels_all

        if len(FEATURE_TYPES) > 1:
            encoders = {}
            emb_dims = {}
            for ft in FEATURE_TYPES:
                enc, dim = make_encoder(ft, return_logits=False)
                encoders[ft] = enc
                emb_dims[ft] = dim
            modelnet = FusionNet(encoders, emb_dims)

            train_iter = Get_Dataloader({ft: features[ft][train_idx] for ft in FEATURE_TYPES},
                                        y_cls_sub[train_idx], True, batch_size, multi=True)
            val_iter   = Get_Dataloader({ft: features[ft][val_idx] for ft in FEATURE_TYPES},
                                        y_cls_sub[val_idx], False, batch_size, multi=True)
            test_iter  = Get_Dataloader({ft: features[ft][test_idx] for ft in FEATURE_TYPES},
                                        y_cls_sub[test_idx], False, batch_size, multi=True)
        else:
            ft = FEATURE_TYPES[0]
            modelnet = make_encoder(ft, return_logits=True)

            train_iter = Get_Dataloader(features[ft][train_idx], y_cls_sub[train_idx], True, batch_size)
            val_iter   = Get_Dataloader(features[ft][val_idx],   y_cls_sub[val_idx],   False, batch_size)
            test_iter  = Get_Dataloader(features[ft][test_idx],  y_cls_sub[test_idx],  False, batch_size)

        modelnet = train(modelnet, train_iter, val_iter, test_iter,
                         num_epochs, lr, run_device,
                         subname=subname, scalers=scalers)
