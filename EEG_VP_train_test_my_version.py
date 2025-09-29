# ==========================================
# EEG classification
# ==========================================
import os
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

# embedding dims (tune here!)
emb_dim_segments = 256
emb_dim_DE       = 64     # authors use 64 for DE
emb_dim_PSD      = 256

WEIGHT_DECAY     = 0   # 0 to match authors' Adam setup

FEATURE_TYPES    = ["DE_glf"]
USE_ALL_SUBJECTS = True
subject_name     = "sub1.npy"


FEATURE_PATHS = {
    "segments_shallow":   "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments",
    "segments_deep":      "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments",
    "segments_eegnet":    "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments",
    "segments_tsconv":    "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments",
    "segments_glf":       "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments",
    "segments_conformer": "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments",
    "DE_glf":  "/content/drive/MyDrive/EEG2Video_data/processed/EEG_DE_1per1s",
    "DE_mlp":  "/content/drive/MyDrive/EEG2Video_data/processed/EEG_DE_1per1s",
    "PSD_glf": "/content/drive/MyDrive/EEG2Video_data/processed/EEG_PSD_1per1s",
    "PSD_mlp": "/content/drive/MyDrive/EEG2Video_data/processed/EEG_PSD_1per1s",
}


# ==========================================
# FusionNet
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
        return self.classifier(fused)


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
            if multi: X = {ft: X[ft].to(device) for ft in X}
            else:     X = X.to(device)
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

    best_val_acc, best_state = 0.0, None
    for epoch in range(num_epochs):
        net.train()
        total_loss, correct, total = 0, 0, 0
        for X, y in train_iter:
            if multi: X = {ft: X[ft].to(device) for ft in X}
            else:     X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = net(X)
            loss = ce_loss(y_hat, y)

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
        if ft.startswith("DE") or ft.startswith("PSD"):
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

        # scale (authors’ style: fit scaler separately for each split)
        for ft in train_data:
            tr = train_data[ft].reshape(train_data[ft].shape[0], -1)
            va = val_data[ft].reshape(val_data[ft].shape[0], -1)
            te = test_data[ft].reshape(test_data[ft].shape[0], -1)

            scaler_tr = StandardScaler().fit(tr)
            tr_scaled = scaler_tr.transform(tr)
            scaler_va = StandardScaler().fit(va)
            va_scaled = scaler_va.transform(va)
            scaler_te = StandardScaler().fit(te)
            te_scaled = scaler_te.transform(te)

            if ft.startswith("segments"):
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

            encoders, emb_dims = {}, {}
            for ft in FEATURE_TYPES:
                if ft.startswith("DE") or ft.startswith("PSD"):
                    emb_size = emb_dim_DE if "DE" in ft else emb_dim_PSD
                    if "glf" in ft:
                        encoders[ft] = models.glfnet_mlp(out_dim=emb_size, emb_dim=emb_size, input_dim=62*5)
                    else:
                        encoders[ft] = models.mlpnet(out_dim=emb_size, input_dim=62*5)
                    emb_dims[ft] = emb_size
                else:
                    if "shallow" in ft:     encoders[ft] = models.shallownet(emb_dim_segments, C, 200)
                    elif "deep" in ft:      encoders[ft] = models.deepnet(emb_dim_segments, C, 200)
                    elif "eegnet" in ft:    encoders[ft] = models.eegnet(emb_dim_segments, C, 200)
                    elif "tsconv" in ft:    encoders[ft] = models.tsconv(emb_dim_segments, C, 200)
                    elif "glf" in ft:       encoders[ft] = models.glfnet(emb_dim_segments, emb_dim_segments, C, 200)
                    elif "conformer" in ft: encoders[ft] = models.conformer(out_dim=emb_dim_segments)
                    emb_dims[ft] = emb_dim_segments

            modelnet = FusionNet(encoders, emb_dims, num_classes=40)

        else:
            ft = FEATURE_TYPES[0]
            train_iter = Get_Dataloader(train_data[ft], train_label, True, batch_size)
            val_iter   = Get_Dataloader(val_data[ft],   val_label,   False, batch_size)
            test_iter  = Get_Dataloader(test_data[ft],  test_label,  False, batch_size)

            if ft.startswith("DE") or ft.startswith("PSD"):
                if "glf" in ft:
                    emb_size = emb_dim_DE if "DE" in ft else emb_dim_PSD
                    modelnet = models.glfnet_mlp(out_dim=40, emb_dim=emb_size, input_dim=62*5)
                else:
                    modelnet = models.mlpnet(out_dim=40, input_dim=62*5)
            else:
                if "shallow" in ft:     modelnet = models.shallownet(40, C, 200)
                elif "deep" in ft:      modelnet = models.deepnet(40, C, 200)
                elif "eegnet" in ft:    modelnet = models.eegnet(40, C, 200)
                elif "tsconv" in ft:    modelnet = models.tsconv(40, C, 200)
                elif "glf" in ft:       modelnet = models.glfnet(40, emb_dim_segments, C, 200)
                elif "conformer" in ft: modelnet = models.conformer(out_dim=40)

        modelnet = train(modelnet, train_iter, val_iter, test_iter, num_epochs, lr, run_device, multi=multi)

        block_top1, block_top5 = [], []
        with torch.no_grad():
            for X, y in test_iter:
                if multi: X = {ft: X[ft].to(run_device) for ft in X}
                else:     X = X.to(run_device)
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
