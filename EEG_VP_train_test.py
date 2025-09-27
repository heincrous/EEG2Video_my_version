# ==========================================
# EEG classification (trial-level 70/15/15 split, one fold)
# ==========================================
import os
import numpy as np
import torch
from torch import nn
from torch.utils import data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from einops import rearrange
import models

# ==========================================
# Config
# ==========================================
batch_size   = 256
num_epochs   = 100  # fewer epochs often better
lr           = 0.001
C            = 62
T            = 5
run_device   = "cuda"
emb_dim_segments = 256
emb_dim_DE   = 64
emb_dim_PSD  = 64

# Select feature type: "segments", "DE", or "PSD"
FEATURE_TYPE = "DE"

# Loss type: "crossentropy", "mse", "cosine", "mse+cosine"
LOSS_TYPE = "crossentropy"

# Variance regularisation toggle
USE_VAR_REG = False
VAR_LAMBDA  = 0.01

FEATURE_PATHS = {
    "segments": "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments",
    "DE":       "/content/drive/MyDrive/EEG2Video_data/processed/EEG_DE_1per1s",
    "PSD":      "/content/drive/MyDrive/EEG2Video_data/processed/EEG_PSD_1per1s",
}

MODEL_MAP = {
    "segments": lambda: models.glfnet(out_dim=40, emb_dim=emb_dim_segments, C=62, T=200),
    "DE":       lambda: models.glfnet_mlp(out_dim=40, emb_dim=emb_dim_DE, input_dim=62*5),
    "PSD":      lambda: models.glfnet_mlp(out_dim=40, emb_dim=emb_dim_PSD, input_dim=62*5),
}

data_path = FEATURE_PATHS[FEATURE_TYPE]

# ==========================================
# Utilities
# ==========================================
def Get_Dataloader(datat, labelt, istrain, batch_size):
    features = torch.tensor(datat, dtype=torch.float32)
    labels   = torch.tensor(labelt, dtype=torch.long)
    return data.DataLoader(data.TensorDataset(features, labels),
                           batch_size, shuffle=istrain)

class Accumulator:
    def __init__(self, n): self.data = [0.0] * n
    def add(self, *args): self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self): self.data = [0.0] * len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def cal_accuracy(y_hat, y):
    if y_hat.ndim > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, axis=1)
    return (y_hat == y).sum()

def evaluate_accuracy_gpu(net, data_iter, device):
    net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            metric.add(cal_accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def topk_accuracy(output, target, topk=(1,5)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k / target.size(0)).item())
        return res

# ==========================================
# Training loop
# ==========================================
def train(net, train_iter, val_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net.to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0)
    ce_loss   = nn.CrossEntropyLoss()
    mse_loss  = nn.MSELoss()
    cos_loss  = nn.CosineEmbeddingLoss()

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(num_epochs):
        net.train()
        metric = Accumulator(3)
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
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
            elif LOSS_TYPE == "mse+cosine":
                y_onehot = torch.nn.functional.one_hot(y, num_classes=40).float()
                target = torch.ones(y_hat.size(0), device=device)
                loss = mse_loss(y_hat, y_onehot) + cos_loss(y_hat, y_onehot, target)

            if USE_VAR_REG:
                var = torch.var(y_hat, dim=0).mean()
                loss -= VAR_LAMBDA * var

            loss.backward()
            optimizer.step()
            metric.add(loss.item() * X.shape[0], cal_accuracy(y_hat, y), X.shape[0])

        train_loss = metric[0] / metric[2]
        train_acc  = metric[1] / metric[2]
        val_acc    = evaluate_accuracy_gpu(net, val_iter, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = net.state_dict()

        if epoch % 3 == 0:
            test_acc = evaluate_accuracy_gpu(net, test_iter, device)
            print(f"[{epoch+1}] loss={train_loss:.3f}, "
                  f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, test_acc={test_acc:.3f}")

    if best_state is not None:
        net.load_state_dict(best_state)
    return net

# ==========================================
# Main
# ==========================================
all_subs = os.listdir(data_path)
print("Available subjects:", all_subs)

sub_choice = "sub1.npy"  # choose one subject for now
sub_list   = [sub_choice]

for subname in sub_list:
    load_npy = np.load(os.path.join(data_path, subname))
    print("Loaded:", subname, load_npy.shape)

    # Flatten into trials
    if FEATURE_TYPE in ["DE", "PSD"]:
        X = rearrange(load_npy, "a b c d e f -> (a b c d) e f")
        y = np.tile(np.arange(40).repeat(10), 7)  # 2800 labels
    elif FEATURE_TYPE == "segments":
        X = rearrange(load_npy, "a b c d (w t) -> (a b c w) d t", w=2)
        y = np.tile(np.arange(40).repeat(10), 7*2)  # 2800 labels

    print("Final trial-level:", X.shape, y.shape)

    # Stratified 70/15/15 split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
    )

    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

    # Scaling
    if FEATURE_TYPE in ["DE", "PSD"]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(len(X_train), -1)).reshape(-1, C, T)
        X_val   = scaler.transform(X_val.reshape(len(X_val), -1)).reshape(-1, C, T)
        X_test  = scaler.transform(X_test.reshape(len(X_test), -1)).reshape(-1, C, T)
    elif FEATURE_TYPE == "segments":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(len(X_train), -1)).reshape(-1, 1, C, 200)
        X_val   = scaler.transform(X_val.reshape(len(X_val), -1)).reshape(-1, 1, C, 200)
        X_test  = scaler.transform(X_test.reshape(len(X_test), -1)).reshape(-1, 1, C, 200)

    # Dataloaders
    train_iter = Get_Dataloader(X_train, y_train, True, batch_size)
    val_iter   = Get_Dataloader(X_val,   y_val,   False, batch_size)
    test_iter  = Get_Dataloader(X_test,  y_test,  False, batch_size)

    # Model + training
    modelnet = MODEL_MAP[FEATURE_TYPE]()
    modelnet = train(modelnet, train_iter, val_iter, test_iter, num_epochs, lr, run_device)

    # Final test eval
    top1_list, top5_list = [], []
    with torch.no_grad():
        for Xb, yb in test_iter:
            Xb, yb = Xb.to(run_device), yb.to(run_device)
            logits = modelnet(Xb)
            t1, t5 = topk_accuracy(logits, yb, (1,5))
            top1_list.append(t1)
            top5_list.append(t5)

    print(f"{subname} | Test Top1={np.mean(top1_list):.3f}, Top5={np.mean(top5_list):.3f}")
