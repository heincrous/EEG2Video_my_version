# ==========================================
# EEG classification (with internal best checkpoint, no saving to disk)
# ==========================================
import os
import numpy as np
import torch
from torch import nn
from torch.utils import data
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from einops import rearrange
import models

# ==========================================
# Config
# ==========================================
batch_size   = 256
num_epochs   = 100
lr           = 0.0001
C            = 62
T            = 5
run_device   = "cuda"
emb_dim_segments = 512
emb_dim_DE = 128
emb_dim_PSD = 128

# Select feature type: "segments", "DE", or "PSD"
FEATURE_TYPE = "DE"

# Loss type: "crossentropy", "mse", "cosine", "mse+cosine"
LOSS_TYPE = "crossentropy"

# Whether to add variance regularisation
USE_VAR_REG = False
VAR_LAMBDA  = 0.01   # regularisation strength (~0.1 usually works well)

FEATURE_PATHS = {
    "segments": "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments",
    "DE":       "/content/drive/MyDrive/EEG2Video_data/processed/EEG_DE_1per1s",
    "PSD":      "/content/drive/MyDrive/EEG2Video_data/processed/EEG_PSD_1per1s",
}

MODEL_MAP = {
    "segments": lambda: models.glfnet(out_dim=40, emb_dim=emb_dim_segments, C=62, T=400),
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
# Training loop (with in-memory checkpointing)
# ==========================================
def train(net, train_iter, val_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net.to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0) # 0.01 is also good

    # define base loss functions
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

            # ------------------------------------------
            # choose loss type
            # ------------------------------------------
            if LOSS_TYPE == "crossentropy":
                loss = ce_loss(y_hat, y)

            elif LOSS_TYPE == "mse":
                # one-hot targets
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

            # ------------------------------------------
            # optional variance regularisation
            # ------------------------------------------
            if USE_VAR_REG:
                var = torch.var(y_hat, dim=0).mean()
                loss -= VAR_LAMBDA * var  # encourage variance across predictions

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
# Label generation
# ==========================================
All_label = np.tile(np.arange(40).repeat(10), 7).reshape(7, 400)

# ==========================================
# Main
# ==========================================
all_subs = os.listdir(data_path)
print("Available subjects:", all_subs)

# Options:
#   sub_choice = "sub1.npy"       # single subject
#   sub_choice = "all"            # all subjects
sub_choice = "sub1.npy"

if sub_choice == "all":
    sub_list = all_subs
else:
    sub_list = [sub_choice]

All_sub_top1, All_sub_top5 = [], []

for subname in sub_list:
    load_npy = np.load(os.path.join(data_path, subname))
    print("Loaded:", subname, load_npy.shape)

    if FEATURE_TYPE in ["DE", "PSD"]:
        # shape: (7,40,5,2,62,5) → (7,400,62,5)
        All_train = rearrange(load_npy, "a b c d e f -> a (b c d) e f")
    elif FEATURE_TYPE == "segments":
        # shape: (7,40,5,62,400) → (7,400,62,400)
        All_train = rearrange(load_npy, "a b c d e f -> a (b c) d e")

    print("Reshaped:", All_train.shape)
    Top_1, Top_K = [], []

    for test_set_id in range(7):
        val_set_id = (test_set_id - 1) % 7

        train_data = np.concatenate([All_train[i] for i in range(7) if i!=test_set_id])
        train_label= np.concatenate([All_label[i] for i in range(7) if i!=test_set_id])
        test_data, test_label = All_train[test_set_id], All_label[test_set_id]
        val_data,  val_label  = All_train[val_set_id],  All_label[val_set_id]

        # ==========================================
        # Scaling and reshaping
        # ==========================================
        if FEATURE_TYPE in ["DE", "PSD"]:
            # flatten
            train_data = train_data.reshape(train_data.shape[0], C*T)
            val_data   = val_data.reshape(val_data.shape[0], C*T)
            test_data  = test_data.reshape(test_data.shape[0], C*T)

            scaler = StandardScaler()
            train_data = scaler.fit_transform(train_data).reshape(-1, C, T)

            scaler = StandardScaler()
            val_data = scaler.fit_transform(val_data).reshape(-1, C, T)

            scaler = StandardScaler()
            test_data = scaler.fit_transform(test_data).reshape(-1, C, T)

        elif FEATURE_TYPE == "segments":
            # flatten
            train_data = train_data.reshape(train_data.shape[0], C*400)
            val_data   = val_data.reshape(val_data.shape[0], C*400)
            test_data  = test_data.reshape(test_data.shape[0], C*400)

            scaler = StandardScaler()
            train_data = scaler.fit_transform(train_data).reshape(-1, 1, C, 400)

            scaler = StandardScaler()
            val_data = scaler.fit_transform(val_data).reshape(-1, 1, C, 400)

            scaler = StandardScaler()
            test_data = scaler.fit_transform(test_data).reshape(-1, 1, C, 400)

        # ==========================================
        # Model + Dataloaders
        # ==========================================
        modelnet = MODEL_MAP[FEATURE_TYPE]()
        train_iter = Get_Dataloader(train_data, train_label, True, batch_size)
        val_iter   = Get_Dataloader(val_data,   val_label,   False, batch_size)
        test_iter  = Get_Dataloader(test_data,  test_label,  False, batch_size)

        modelnet = train(modelnet, train_iter, val_iter, test_iter, num_epochs, lr, run_device)

        block_top1, block_top5 = [], []
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(run_device), y.to(run_device)
                logits = modelnet(X)
                t1,t5 = topk_accuracy(logits,y,(1,5))
                block_top1.append(t1); block_top5.append(t5)
        Top_1.append(np.mean(block_top1))
        Top_K.append(np.mean(block_top5))

    print(f"{subname} | Top1={np.mean(Top_1):.3f}, Top5={np.mean(Top_K):.3f}")
    All_sub_top1.append(np.mean(Top_1))
    All_sub_top5.append(np.mean(Top_K))

print("\nOverall:")
print("TOP1:", np.mean(All_sub_top1), np.std(All_sub_top1))
print("TOP5:", np.mean(All_sub_top5), np.std(All_sub_top5))
