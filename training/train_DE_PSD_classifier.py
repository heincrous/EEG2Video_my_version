import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tqdm import tqdm

# === Repo imports ===
repo_root = "/content/EEG2Video_my_version"
sys.path.append(repo_root)
from core_files.models import glfnet_mlp

# -------------------------------------------------
# Top-k accuracy helper
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
# Training routine
# -------------------------------------------------
def train_and_eval(model, train_loader, val_loader, test_loader, device, num_epochs=100, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_val_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        # ---- train ----
        model.train()
        correct, total = 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        train_acc = correct / total

        # ---- val ----
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    # ---- test with best model ----
    model.load_state_dict(best_state)
    model.eval()
    top1_list, top5_list = [], []
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            accs = topk_accuracy(out, y, topk=(1, 5))
            top1_list.append(accs[0]); top5_list.append(accs[1])
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return np.mean(top1_list), np.mean(top5_list), np.array(all_preds), np.array(all_labels)

# -------------------------------------------------
# Helper to build loaders
# -------------------------------------------------
def build_loaders(train_x, train_y, val_x, val_y, test_x, test_y):
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x.reshape(len(train_x), -1)).reshape(len(train_x), *train_x.shape[1:])
    val_x   = scaler.transform(val_x.reshape(len(val_x), -1)).reshape(len(val_x), *val_x.shape[1:])
    test_x  = scaler.transform(test_x.reshape(len(test_x), -1)).reshape(len(test_x), *test_x.shape[1:])
    batch_size = 256
    train_loader = DataLoader(TensorDataset(torch.tensor(train_x, dtype=torch.float32),
                                            torch.tensor(train_y, dtype=torch.long)),
                                            batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(val_x, dtype=torch.float32),
                                          torch.tensor(val_y, dtype=torch.long)),
                                          batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(test_x, dtype=torch.float32),
                                           torch.tensor(test_y, dtype=torch.long)),
                                           batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# -------------------------------------------------
# Run one variation
# -------------------------------------------------
def run_variation(mode, feat_type, split_mode, subj_name, drive_root, device):
    # ---- feature dirs ----
    if feat_type == "de":
        eeg_dir = os.path.join(drive_root, "EEG_DE" if mode == "1per2s" else "EEG_DE_1per1s")
        data = np.load(os.path.join(eeg_dir, f"{subj_name}.npy"))
        if mode == "1per1s":
            data = data.reshape(7, 40, 10, 62, 5)
    elif feat_type == "psd":
        eeg_dir = os.path.join(drive_root, "EEG_PSD" if mode == "1per2s" else "EEG_PSD_1per1s")
        data = np.load(os.path.join(eeg_dir, f"{subj_name}.npy"))
        if mode == "1per1s":
            data = data.reshape(7, 40, 10, 62, 5)
    else:  # combo
        eeg_dir_de  = os.path.join(drive_root, "EEG_DE"  if mode == "1per2s" else "EEG_DE_1per1s")
        eeg_dir_psd = os.path.join(drive_root, "EEG_PSD" if mode == "1per2s" else "EEG_PSD_1per1s")
        data_de  = np.load(os.path.join(eeg_dir_de,  f"{subj_name}.npy"))
        data_psd = np.load(os.path.join(eeg_dir_psd, f"{subj_name}.npy"))
        if mode == "1per1s":
            data_de  = data_de.reshape(7, 40, 10, 62, 5)
            data_psd = data_psd.reshape(7, 40, 10, 62, 5)
        data = np.concatenate([data_de, data_psd], axis=-1)

    # ---- build splits ----
    if split_mode == "fixed":
        train_blocks = [i for i in range(7) if i not in [5, 6]]
        val_block, test_block = 5, 6
        train_x, train_y, val_x, val_y, test_x, test_y = [], [], [], [], [], []
        for b in train_blocks:
            for c in range(40):
                for k in range(data.shape[2]):
                    train_x.append(data[b,c,k]); train_y.append(c)
        for c in range(40):
            for k in range(data.shape[2]):
                val_x.append(data[val_block,c,k]); val_y.append(c)
                test_x.append(data[test_block,c,k]); test_y.append(c)
        train_x, val_x, test_x = np.array(train_x), np.array(val_x), np.array(test_x)
        train_y, val_y, test_y = np.array(train_y), np.array(val_y), np.array(test_y)
        train_loader, val_loader, test_loader = build_loaders(train_x, train_y, val_x, val_y, test_x, test_y)

        input_dim = data.shape[-1] * data.shape[-2]
        model = glfnet_mlp(out_dim=40, emb_dim=64, input_dim=input_dim).to(device)
        top1, top5, preds, labels = train_and_eval(model, train_loader, val_loader, test_loader, device)
        return top1, top5

    else:  # cross-validation
        top1_scores, top5_scores = [], []
        for test_block in range(7):
            val_block = (test_block - 1) % 7
            train_blocks = [i for i in range(7) if i not in [test_block, val_block]]
            train_x, train_y, val_x, val_y, test_x, test_y = [], [], [], [], [], []
            for b in train_blocks:
                for c in range(40):
                    for k in range(data.shape[2]):
                        train_x.append(data[b,c,k]); train_y.append(c)
            for c in range(40):
                for k in range(data.shape[2]):
                    val_x.append(data[val_block,c,k]); val_y.append(c)
                    test_x.append(data[test_block,c,k]); test_y.append(c)
            train_x, val_x, test_x = np.array(train_x), np.array(val_x), np.array(test_x)
            train_y, val_y, test_y = np.array(train_y), np.array(val_y), np.array(test_y)
            train_loader, val_loader, test_loader = build_loaders(train_x, train_y, val_x, val_y, test_x, test_y)
            input_dim = data.shape[-1] * data.shape[-2]
            model = glfnet_mlp(out_dim=40, emb_dim=64, input_dim=input_dim).to(device)
            top1, top5, _, _ = train_and_eval(model, train_loader, val_loader, test_loader, device)
            top1_scores.append(top1); top5_scores.append(top5)
        return np.mean(top1_scores), np.mean(top5_scores)

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    drive_root = "/content/drive/MyDrive/EEG2Video_data/processed"
    mode = input("Select mode (1per1s / 1per2s / all): ").strip().lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- subject ----
    eeg_dir = os.path.join(drive_root, "EEG_DE" if mode!="1per1s" else "EEG_DE_1per1s")
    subjects = [f.replace(".npy", "") for f in sorted(os.listdir(eeg_dir)) if f.endswith(".npy")]
    print("\nAvailable subjects:")
    for idx, subj in enumerate(subjects):
        print(f"{idx}: {subj}")
    subj_idx = int(input("\nEnter subject index to process: ").strip())
    subj_name = subjects[subj_idx]

    results = []

    if mode == "all":
        for m in ["1per1s", "1per2s"]:
            for f in ["de", "psd", "combo"]:
                for s in ["fixed", "cv"]:
                    print(f"\n=== Running {m} | {f} | {s} ===")
                    top1, top5 = run_variation(m, f, s, subj_name, drive_root, device)
                    results.append((m, f, s, top1, top5))
        print("\n=== Final Summary Table ===")
        print(f"{'Mode':<8}{'Feature':<8}{'Split':<8}{'Top-1':<10}{'Top-5':<10}")
        for m,f,s,t1,t5 in results:
            print(f"{m:<8}{f:<8}{s:<8}{t1:.3f}{t5:>10.3f}")
    else:
        feat_type = input("Select feature type (de/psd/combo): ").strip().lower()
        split_mode = input("Use fixed split (y/n)? ").strip().lower()
        split_mode = "fixed" if split_mode=="y" else "cv"
        top1, top5 = run_variation(mode, feat_type, split_mode, subj_name, drive_root, device)
        print("\n=== Overall summary ===")
        print(f"Top-1 Accuracy: {top1:.3f}")
        print(f"Top-5 Accuracy: {top5:.3f}")

if __name__ == "__main__":
    main()
