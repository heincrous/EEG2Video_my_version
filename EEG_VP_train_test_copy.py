# ==========================================
# EEG classification (FusionNet with independent scalers per feature & split)
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
# Hyperparameters
# ==========================================
batch_size   = 256
num_epochs   = 50
lr           = 0.001
run_device   = "cuda"

# ==========================================
# Utilities
# ==========================================
def Get_Dataloader(features_dict, labels, istrain, batch_size):
    tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in features_dict.items()}
    labels  = torch.tensor(labels, dtype=torch.long)
    # Debugging: ensure same length
    lengths = [t.shape[0] for t in tensors.values()] + [labels.shape[0]]
    assert len(set(lengths)) == 1, f"Size mismatch: {lengths}"
    return data.DataLoader(data.TensorDataset(*(list(tensors.values()) + [labels])),
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
        for *Xs, y in data_iter:
            Xs = {k: X.to(device) for k, X in zip(net.encoders.keys(), Xs)}
            y  = y.to(device)
            metric.add(cal_accuracy(net(Xs), y), y.numel())
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
# FusionNet wrapper
# ==========================================
class FusionNet(nn.Module):
    def __init__(self, encoders, out_dim):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        total_dim = sum(e.out.out_features for e in encoders.values())
        self.classifier = nn.Linear(total_dim, out_dim)

    def forward(self, inputs):
        feats = []
        for k, enc in self.encoders.items():
            feats.append(enc(inputs[k]))
        fused = torch.cat(feats, dim=1)
        return self.classifier(fused)

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
    loss_fn   = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(num_epochs):
        net.train()
        metric = Accumulator(3)
        for *Xs, y in train_iter:
            Xs = {k: X.to(device) for k, X in zip(net.encoders.keys(), Xs)}
            y  = y.to(device)
            optimizer.zero_grad()
            y_hat = net(Xs)
            loss  = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            metric.add(loss.item() * y.size(0), cal_accuracy(y_hat, y), y.size(0))
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
# Main (example with DE/PSD/segments dirs)
# ==========================================
seg_root = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments"
de_root  = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_DE_1per1s"
psd_root = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_PSD_1per1s"

sub_list = os.listdir(de_root)

All_sub_top1, All_sub_top5 = [], []

for subname in sub_list:
    seg_npy = np.load(os.path.join(seg_root, subname))   # (40,7,5,62,400)
    de_npy  = np.load(os.path.join(de_root,  subname))
    psd_npy = np.load(os.path.join(psd_root, subname))

    # reshape to (7,400,...)
    seg_all = rearrange(seg_npy, "c b r ch t -> b (c r) ch t")   # (7,400,62,400)
    de_all  = rearrange(de_npy,  "a b c d e f -> a (b c d) e f") # (7,400,62,5)
    psd_all = rearrange(psd_npy, "a b c d e f -> a (b c d) e f") # (7,400,62,5)

    Top_1, Top_K = [], []

    for test_set_id in range(7):
        val_set_id = (test_set_id - 1) % 7
        train_idx = [i for i in range(7) if i not in [test_set_id, val_set_id]]

        splits = {}
        for feat_name, arr in zip(["segments","de","psd"], [seg_all,de_all,psd_all]):
            train_data = np.concatenate([arr[i] for i in train_idx])
            val_data   = arr[val_set_id]
            test_data  = arr[test_set_id]

            # flatten → scale → reshape back
            train_scaler = StandardScaler().fit(train_data.reshape(train_data.shape[0], -1))
            train_data   = train_scaler.transform(train_data.reshape(train_data.shape[0], -1)).reshape(train_data.shape)

            val_scaler   = StandardScaler().fit(val_data.reshape(val_data.shape[0], -1))
            val_data     = val_scaler.transform(val_data.reshape(val_data.shape[0], -1)).reshape(val_data.shape)

            test_scaler  = StandardScaler().fit(test_data.reshape(test_data.shape[0], -1))
            test_data    = test_scaler.transform(test_data.reshape(test_data.shape[0], -1)).reshape(test_data.shape)

            splits[feat_name] = {"train": train_data, "val": val_data, "test": test_data}

        train_label = np.concatenate([All_label[i] for i in train_idx])
        val_label   = All_label[val_set_id]
        test_label  = All_label[test_set_id]

        print(f"DEBUG {subname} Block{test_set_id}:",
              {k:(v['train'].shape, v['val'].shape, v['test'].shape) for k,v in splits.items()},
              "labels:", (train_label.shape, val_label.shape, test_label.shape))

        train_iter = Get_Dataloader({k:v["train"] for k,v in splits.items()}, train_label, True, batch_size)
        val_iter   = Get_Dataloader({k:v["val"]   for k,v in splits.items()}, val_label,   False, batch_size)
        test_iter  = Get_Dataloader({k:v["test"]  for k,v in splits.items()}, test_label,  False, batch_size)

        encoders = {
            "segments": models.glfnet(out_dim=40, emb_dim=64, C=62, T=400),
            "de":       models.glfnet_mlp(out_dim=40, emb_dim=64, input_dim=310),
            "psd":      models.glfnet_mlp(out_dim=40, emb_dim=64, input_dim=310),
        }
        modelnet = FusionNet(encoders, out_dim=40)

        modelnet = train(modelnet, train_iter, val_iter, test_iter, num_epochs, lr, run_device)

        block_top1, block_top5 = [], []
        with torch.no_grad():
            for *Xs, y in test_iter:
                Xs = {k: X.to(run_device) for k, X in zip(modelnet.encoders.keys(), Xs)}
                y  = y.to(run_device)
                logits = modelnet(Xs)
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
