import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# === Repo imports ===
repo_root = "/content/EEG2Video_my_version"
sys.path.append(repo_root)
from core_files.models import shallownet, deepnet, eegnet, tsconv, conformer, glfnet_mlp

# -------------------------------------------------
# Fusion model
# -------------------------------------------------
class FusionModel(nn.Module):
    def __init__(self, encoders, num_classes=40):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        total_dim = sum([list(e.modules())[-1].out_features for e in encoders.values()])
        self.classifier = nn.Linear(total_dim, num_classes)

    def forward(self, inputs):
        feats = []
        for name, enc in self.encoders.items():
            feats.append(enc(inputs[name]))
        fused = torch.cat(feats, dim=-1)
        return self.classifier(fused)

# -------------------------------------------------
# Dataset wrapper for dict features
# -------------------------------------------------
class DictDataset(Dataset):
    def __init__(self, Xs, y):
        self.Xs, self.y = Xs, y
        self.keys = list(Xs.keys())
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return {k: torch.tensor(self.Xs[k][idx], dtype=torch.float32) for k in self.keys}, torch.tensor(self.y[idx], dtype=torch.long)

# -------------------------------------------------
# Accuracy
# -------------------------------------------------
def topk_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / target.size(0)).item())
        return res

# -------------------------------------------------
# Train + Eval
# -------------------------------------------------
def train_and_eval(model, train_loader, val_loader, test_loader, device, num_epochs=50, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_val_acc, best_state = 0.0, None

    for _ in range(num_epochs):
        model.train()
        for Xs, y in train_loader:
            Xs = {k: v.to(device) for k,v in Xs.items()}
            y = y.to(device)
            optimizer.zero_grad()
            out = model(Xs)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for Xs, y in val_loader:
                Xs = {k: v.to(device) for k,v in Xs.items()}
                y = y.to(device)
                out = model(Xs)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / val_total if val_total>0 else 0
        if val_acc > best_val_acc:
            best_val_acc, best_state = val_acc, model.state_dict()

    # test
    model.load_state_dict(best_state)
    model.eval()
    top1_list, top5_list = [], []
    with torch.no_grad():
        for Xs, y in test_loader:
            Xs = {k: v.to(device) for k,v in Xs.items()}
            y = y.to(device)
            out = model(Xs)
            accs = topk_accuracy(out, y, topk=(1,5))
            top1_list.append(accs[0]); top5_list.append(accs[1])
    return np.mean(top1_list), np.mean(top5_list)

# -------------------------------------------------
# Feature loader
# -------------------------------------------------
def load_feature(feat_type, subj_name, drive_root):
    if feat_type == "de":
        arr = np.load(os.path.join(drive_root,"EEG_DE_1per1s",f"{subj_name}.npy"))
        return arr.reshape(7,40,10,62,5)
    elif feat_type == "psd":
        arr = np.load(os.path.join(drive_root,"EEG_PSD_1per1s",f"{subj_name}.npy"))
        return arr.reshape(7,40,10,62,5)
    elif feat_type == "windows":
        arr = np.load(os.path.join(drive_root,"EEG_windows",f"{subj_name}.npy"))
        # expected (7,40,10,62,100)
        return arr
    elif feat_type == "segments":
        arr = np.load(os.path.join(drive_root,"EEG_segments",f"{subj_name}.npy"))
        # expected (7,40,10,62,400)
        return arr
    else:
        raise ValueError(f"Unknown feature {feat_type}")

# -------------------------------------------------
# Cross-validation
# -------------------------------------------------
def run_cv(subj_name, drive_root, device, feat_types, encoders):
    datas = {f: load_feature(f, subj_name, drive_root) for f in feat_types}
    top1_scores, top5_scores = [], []

    with tqdm(total=7, desc="Cross-validation") as pbar:
        for test_block in range(7):
            val_block = (test_block-1)%7
            train_blocks = [i for i in range(7) if i not in [test_block,val_block]]

            X_dicts = {"train":{f:[] for f in feat_types},
                       "val":{f:[] for f in feat_types},
                       "test":{f:[] for f in feat_types}}
            y_dicts = {"train":[], "val":[], "test":[]}

            trial_count = min([datas[f].shape[2] for f in feat_types])

            for b in train_blocks:
                for c in range(40):
                    for k in range(trial_count):
                        for f in feat_types:
                            X_dicts["train"][f].append(datas[f][b,c,k])
                        y_dicts["train"].append(c)

            for c in range(40):
                for k in range(trial_count):
                    for f in feat_types:
                        X_dicts["val"][f].append(datas[f][val_block,c,k])
                        X_dicts["test"][f].append(datas[f][test_block,c,k])
                    y_dicts["val"].append(c)
                    y_dicts["test"].append(c)

            for split in ["train","val","test"]:
                for f in feat_types:
                    arr = np.array(X_dicts[split][f])
                    shape = arr.shape
                    scaler = StandardScaler()
                    arr = scaler.fit_transform(arr.reshape(len(arr), -1))
                    X_dicts[split][f] = arr.reshape(shape)
                y_dicts[split] = np.array(y_dicts[split])

            train_loader = DataLoader(DictDataset(X_dicts["train"], y_dicts["train"]),batch_size=256,shuffle=True)
            val_loader   = DataLoader(DictDataset(X_dicts["val"],   y_dicts["val"]),  batch_size=256,shuffle=False)
            test_loader  = DataLoader(DictDataset(X_dicts["test"],  y_dicts["test"]), batch_size=256,shuffle=False)

            model = FusionModel(encoders,num_classes=40).to(device)
            top1, top5 = train_and_eval(model, train_loader, val_loader, test_loader, device)
            top1_scores.append(top1); top5_scores.append(top5)
            pbar.update(1)

    return np.mean(top1_scores), np.mean(top5_scores)

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    drive_root="/content/drive/MyDrive/EEG2Video_data/processed"
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eeg_dir=os.path.join(drive_root,"EEG_DE_1per1s")
    subjects=[f.replace(".npy","") for f in sorted(os.listdir(eeg_dir)) if f.endswith(".npy")]
    for i,s in enumerate(subjects): print(f"{i}: {s}")
    subj_name=subjects[int(input("Enter subject index: ").strip())]

    feat_types=[f.strip().lower() for f in input("Enter features (comma separated: de,psd,windows,segments): ").split(",")]

    encoders={}
    for f in feat_types:
        if f in ["de","psd"]:
            input_dim=62*5
            encoders[f]=glfnet_mlp(out_dim=128,emb_dim=64,input_dim=input_dim).to(device)
        elif f=="windows":
            enc_choice=input("Encoder for windows (shallownet/eegnet/tsconv/conformer): ").strip().lower()
            encoders[f]={"shallownet":shallownet,"eegnet":eegnet,"tsconv":tsconv,"conformer":conformer}[enc_choice](out_dim=128,C=62,T=100).to(device)
        elif f=="segments":
            enc_choice=input("Encoder for segments (shallownet/deepnet/eegnet/tsconv/conformer): ").strip().lower()
            encoders[f]={"shallownet":shallownet,"deepnet":deepnet,"eegnet":eegnet,"tsconv":tsconv,"conformer":conformer}[enc_choice](out_dim=128,C=62,T=400).to(device)

    top1, top5=run_cv(subj_name,drive_root,device,feat_types,encoders)
    print("\n=== Final Results ===")
    print(f"Features: {feat_types}")
    print(f"Top-1 Accuracy: {top1:.3f}")
    print(f"Top-5 Accuracy: {top5:.3f}")

if __name__=="__main__":
    main()
