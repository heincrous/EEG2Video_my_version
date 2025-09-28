# ==========================================
# EEG classification
# ==========================================
import os, random
import numpy as np
import torch
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
lr            = 0.001
C             = 62
T             = 5
run_device    = "cuda"

emb_dim_segments = 512
emb_dim_DE       = 64
emb_dim_PSD      = 256

# optimizer: "adam" or "adamw"; set WEIGHT_DECAY=0 for Adam
WEIGHT_DECAY     = 0

# ==========================================
# Select which features and encoders to use
# ==========================================
# For SEGMENTS (raw EEG windows, shape [B,1,62,200]):
#   FEATURE_TYPES = ["segments_shallow"]
#   FEATURE_TYPES = ["segments_deep"]
#   FEATURE_TYPES = ["segments_eegnet"]
#   FEATURE_TYPES = ["segments_tsconv"]
#   FEATURE_TYPES = ["segments_glf"]
#   FEATURE_TYPES = ["segments_conformer"]
#
# For DE or PSD (frequency features, shape [B,62,5]):
#   FEATURE_TYPES = ["DE_glf"]         # glfnet_mlp
#   FEATURE_TYPES = ["DE_mlp"]         # plain mlpnet
#   FEATURE_TYPES = ["PSD_glf"]
#   FEATURE_TYPES = ["PSD_mlp"]
#
# For multi-feature fusion, just combine:
#   FEATURE_TYPES = ["DE_glf", "PSD_glf"]
#   FEATURE_TYPES = ["segments_glf", "DE_mlp"]
#   FEATURE_TYPES = ["segments_conformer", "DE_glf", "PSD_mlp"]
# ==========================================
FEATURE_TYPES = ["segments_conformer", "PSD_mlp"]

# default is subject 1 only; set to True to use all subjects in folder
USE_ALL_SUBJECTS = False
subject_name     = "sub1.npy"

# loss type: "mse", "cosine", "mse+cosine", "contrastive", "crossentropy"
LOSS_TYPE        = "crossentropy"

USE_VAR_REG = False
VAR_LAMBDA  = 0.01

FEATURE_PATHS = {
    "segments_shallow": "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments",
    "segments_deep":    "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments",
    "segments_eegnet":  "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments",
    "segments_tsconv":  "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments",
    "segments_glf":     "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments",
    "segments_conformer": "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments",

    "DE_glf":  "/content/drive/MyDrive/EEG2Video_data/processed/EEG_DE_1per1s",
    "DE_mlp":  "/content/drive/MyDrive/EEG2Video_data/processed/EEG_DE_1per1s",

    "PSD_glf": "/content/drive/MyDrive/EEG2Video_data/processed/EEG_PSD_1per1s",
    "PSD_mlp": "/content/drive/MyDrive/EEG2Video_data/processed/EEG_PSD_1per1s",
}


# ==========================================
# Encoders → embeddings
# ==========================================
MODEL_MAP = {
    # Segments-based models (CNN-style)
    "segments_shallow":   lambda: models.shallownet(out_dim=emb_dim_segments, C=62, T=200),
    "segments_deep":      lambda: models.deepnet(out_dim=emb_dim_segments, C=62, T=200),
    "segments_eegnet":    lambda: models.eegnet(out_dim=emb_dim_segments, C=62, T=200),
    "segments_tsconv":    lambda: models.tsconv(out_dim=emb_dim_segments, C=62, T=200),
    "segments_glf":       lambda: models.glfnet(out_dim=emb_dim_segments, emb_dim=emb_dim_segments, C=62, T=200),
    "segments_conformer": lambda: models.conformer(emb_size=40, depth=3, out_dim=emb_dim_segments),

    # DE/PSD-based models (MLP-style)
    "DE_glf":   lambda: models.glfnet_mlp(out_dim=emb_dim_DE, emb_dim=emb_dim_DE, input_dim=62*5),
    "DE_mlp":   lambda: models.mlpnet(out_dim=emb_dim_DE, input_dim=62*5),
    "PSD_glf":  lambda: models.glfnet_mlp(out_dim=emb_dim_PSD, emb_dim=emb_dim_PSD, input_dim=62*5),
    "PSD_mlp":  lambda: models.mlpnet(out_dim=emb_dim_PSD, input_dim=62*5),
}

EMB_DIMS = {
    # Segments
    "segments_shallow":   emb_dim_segments,
    "segments_deep":      emb_dim_segments,
    "segments_eegnet":    emb_dim_segments,
    "segments_tsconv":    emb_dim_segments,
    "segments_glf":       emb_dim_segments,
    "segments_conformer": emb_dim_segments,

    # DE/PSD
    "DE_glf":   emb_dim_DE,
    "DE_mlp":   emb_dim_DE,
    "PSD_glf":  emb_dim_PSD,
    "PSD_mlp":  emb_dim_PSD,
}


# ==========================================
# Models
# ==========================================
class FusionNet(nn.Module):
    def __init__(self, encoders, emb_dims, num_classes=40):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        total_dim     = sum(emb_dims.values())
        self.classifier = nn.Linear(total_dim, num_classes)
    def forward(self, inputs):
        feats = [enc(inputs[name]) for name, enc in self.encoders.items()]
        fused = torch.cat(feats, dim=-1)
        return self.classifier(fused)  # logits

class SingleNet(nn.Module):
    def __init__(self, encoder, emb_dim, num_classes=40):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(emb_dim, num_classes)
    def forward(self, x):
        emb = self.encoder(x)
        return self.classifier(emb)  # logits


# ==========================================
# Dataset wrappers
# ==========================================
class FusionDataset(data.Dataset):
    def __init__(self, features_dict, labels):
        self.features = features_dict
        self.labels   = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {ft: torch.tensor(self.features[ft][idx], dtype=torch.float32) for ft in self.features}, \
               torch.tensor(self.labels[idx], dtype=torch.long)

def Get_Dataloader(features, labels, istrain, batch_size, multi=False):
    if multi:
        return data.DataLoader(FusionDataset(features, labels), batch_size, shuffle=istrain)
    else:
        features = torch.tensor(features, dtype=torch.float32)
        labels   = torch.tensor(labels, dtype=torch.long)
        return data.DataLoader(data.TensorDataset(features, labels), batch_size, shuffle=istrain)


# ==========================================
# Metrics
# ==========================================
def cal_accuracy(y_hat, y):
    if y_hat.ndim > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, axis=1)
    return (y_hat == y).sum()

def evaluate_accuracy_gpu(net, data_iter, device, multi=False):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if multi:
                X = {ft: X[ft].to(device) for ft in X}
            else:
                X = X.to(device)
            y = y.to(device)
            correct += cal_accuracy(net(X), y)
            total   += y.numel()
    return correct / total

def topk_accuracy(output, target, topk=(1,5)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [(correct[:k].reshape(-1).float().sum(0, keepdim=True) / target.size(0)).item() for k in topk]


# ==========================================
# Training loop
# ==========================================
def train(net, train_iter, val_iter, test_iter, num_epochs, lr, device, multi=False):
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    ce_loss   = nn.CrossEntropyLoss()
    mse_loss  = nn.MSELoss()
    cos_loss  = nn.CosineEmbeddingLoss()

    best_val_acc, best_state = 0.0, None
    for epoch in range(num_epochs):
        net.train()
        total_loss, correct, total = 0, 0, 0
        for X, y in train_iter:
            if multi:
                X = {ft: X[ft].to(device) for ft in X}
            else:
                X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = net(X)

            if LOSS_TYPE == "crossentropy":
                loss = ce_loss(y_hat, y)
            elif LOSS_TYPE == "mse":
                y_onehot = torch.nn.functional.one_hot(y, num_classes=40).float()
                loss = mse_loss(y_hat, y_onehot)
            elif LOSS_TYPE == "cosine":
                y_onehot = torch.nn.functional.one_hot(y, num_classes=40).float()
                target = torch.ones(y_hat.size(0), device=device)
                loss = cos_loss(y_hat, y_onehot, target)
            elif LOSS_TYPE in ["mse+cosine", "cosine+mse"]:
                y_onehot = torch.nn.functional.one_hot(y, num_classes=40).float()
                target = torch.ones(y_hat.size(0), device=device)
                loss = mse_loss(y_hat, y_onehot) + cos_loss(y_hat, y_onehot, target)

            if USE_VAR_REG:
                loss -= VAR_LAMBDA * torch.var(y_hat, dim=0).mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            correct += cal_accuracy(y_hat, y)
            total   += y.size(0)

        val_acc = evaluate_accuracy_gpu(net, val_iter, device, multi=multi)
        if val_acc > best_val_acc:
            best_val_acc, best_state = val_acc, net.state_dict()

        if epoch % 3 == 0:
            test_acc = evaluate_accuracy_gpu(net, test_iter, device, multi=multi)
            print(f"[{epoch+1}] loss={total_loss/total:.3f}, "
                  f"train_acc={correct/total:.3f}, val_acc={val_acc:.3f}, test_acc={test_acc:.3f}")

    if best_state:
        net.load_state_dict(best_state)
    return net


# ==========================================
# Labels
# ==========================================
All_label = np.tile(np.arange(40).repeat(10), 7).reshape(7, 400)


# ==========================================
# Main
# ==========================================
sub_list = os.listdir(FEATURE_PATHS[FEATURE_TYPES[0]]) if USE_ALL_SUBJECTS else [subject_name]
All_sub_top1, All_sub_top5 = [], []

for subname in sub_list:
    raw_data = {ft: np.load(os.path.join(FEATURE_PATHS[ft], subname)) for ft in FEATURE_TYPES}

    def reshape(ft, arr):
        if ft.startswith("DE"):
            return rearrange(arr, "a b c d e f -> a (b c d) e f")
        elif ft.startswith("PSD"):
            return rearrange(arr, "a b c d e f -> a (b c d) e f")
        elif ft.startswith("segments"):
            return rearrange(arr, "a b c d (w t) -> a (b c w) d t", w=2)
        else:
            raise ValueError(f"Unknown feature type: {ft}")
    All_train = {ft: reshape(ft, raw_data[ft]) for ft in raw_data}

    Top_1, Top_K = [], []
    for test_set_id in range(7):
        val_set_id = (test_set_id - 1) % 7
        train_data, val_data, test_data = {}, {}, {}
        for ft in All_train:
            train_data[ft] = np.concatenate([All_train[ft][i] for i in range(7) if i not in [test_set_id,val_set_id]])
            val_data[ft]   = All_train[ft][val_set_id]
            test_data[ft]  = All_train[ft][test_set_id]

        train_label = np.concatenate([All_label[i] for i in range(7) if i not in [test_set_id,val_set_id]])
        val_label   = All_label[val_set_id]
        test_label  = All_label[test_set_id]

        for ft in train_data:
            tr = train_data[ft].reshape(train_data[ft].shape[0], -1)
            va = val_data[ft].reshape(val_data[ft].shape[0], -1)
            te = test_data[ft].reshape(test_data[ft].shape[0], -1)
            scaler = StandardScaler()
            tr_scaled = scaler.fit_transform(tr)
            va_scaled = scaler.transform(va)
            te_scaled = scaler.transform(te)
            if ft == "segments":
                train_data[ft] = tr_scaled.reshape(-1, 1, C, 200)
                val_data[ft]   = va_scaled.reshape(-1, 1, C, 200)
                test_data[ft]  = te_scaled.reshape(-1, 1, C, 200)
            else:
                train_data[ft] = tr_scaled.reshape(-1, C, T)
                val_data[ft]   = va_scaled.reshape(-1, C, T)
                test_data[ft]  = te_scaled.reshape(-1, C, T)

        multi = len(FEATURE_TYPES) > 1
        if multi:
            train_iter = Get_Dataloader(train_data, train_label, True, batch_size, multi=True)
            val_iter   = Get_Dataloader(val_data,   val_label,   False, batch_size, multi=True)
            test_iter  = Get_Dataloader(test_data,  test_label,  False, batch_size, multi=True)
            encoders   = {ft: MODEL_MAP[ft]() for ft in FEATURE_TYPES}
            emb_dims   = {ft: EMB_DIMS[ft] for ft in FEATURE_TYPES}
            modelnet   = FusionNet(encoders, emb_dims, num_classes=40)
        else:
            ft = FEATURE_TYPES[0]
            train_iter = Get_Dataloader(train_data[ft], train_label, True, batch_size)
            val_iter   = Get_Dataloader(val_data[ft],   val_label,   False, batch_size)
            test_iter  = Get_Dataloader(test_data[ft],  test_label,  False, batch_size)
            modelnet   = SingleNet(MODEL_MAP[ft](), EMB_DIMS[ft], num_classes=40)

        modelnet = train(modelnet, train_iter, val_iter, test_iter, num_epochs, lr, run_device, multi=multi)

        block_top1, block_top5 = [], []
        with torch.no_grad():
            for X, y in test_iter:
                if multi:
                    X = {ft: X[ft].to(run_device) for ft in X}
                else:
                    X = X.to(run_device)
                y = y.to(run_device)
                logits = modelnet(X)
                t1, t5 = topk_accuracy(logits, y, (1,5))
                block_top1.append(t1); block_top5.append(t5)
        Top_1.append(np.mean(block_top1))
        Top_K.append(np.mean(block_top5))

    print(f"{subname} | Top1={np.mean(Top_1):.3f}, Top5={np.mean(Top_K):.3f}")
    All_sub_top1.append(np.mean(Top_1))
    All_sub_top5.append(np.mean(Top_K))

print("\nOverall:")
print("TOP1:", np.mean(All_sub_top1), np.std(All_sub_top1))
print("TOP5:", np.mean(All_sub_top5), np.std(All_sub_top5))
