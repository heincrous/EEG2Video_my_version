import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tqdm import tqdm

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

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

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
# Per-class stats
# -------------------------------------------------
def per_class_stats(conf_matrix, sort=False):
    stats = []
    for i in range(conf_matrix.shape[0]):
        total = conf_matrix[i].sum()
        correct = conf_matrix[i, i]
        acc = correct / total if total > 0 else 0.0
        stats.append((i, acc))
    if sort:
        stats.sort(key=lambda x: x[1], reverse=True)
    return stats

# -------------------------------------------------
# Main routine
# -------------------------------------------------
def main():
    drive_root = "/content/drive/MyDrive/EEG2Video_data/processed"

    # ---- mode ----
    mode = input("Select mode (1per2s / 1per1s): ").strip().lower()
    if mode == "1per2s":
        eeg_dir = os.path.join(drive_root, "EEG_DE")
    elif mode == "1per1s":
        eeg_dir = os.path.join(drive_root, "EEG_DE_1per1s")
    else:
        raise ValueError("Invalid mode.")

    # ---- subject ----
    subjects = [f.replace(".npy", "") for f in sorted(os.listdir(eeg_dir)) if f.endswith(".npy")]
    print("\nAvailable subjects:")
    for idx, subj in enumerate(subjects):
        print(f"{idx}: {subj}")
    subj_idx = int(input("\nEnter subject index to process: ").strip())
    subj_name = subjects[subj_idx]

    # ---- load data ----
    data = np.load(os.path.join(eeg_dir, f"{subj_name}.npy"))
    if mode == "1per1s":
        data = data.reshape(7, 40, 10, 62, 5)  # flatten windows
    print(f"\nLoaded {subj_name} ({mode}): {data.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- split mode ----
    split_mode = input("Use fixed split (y/n)? ").strip().lower()

    if split_mode == "y":
        # fixed split
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

        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x.reshape(len(train_x), -1)).reshape(len(train_x), 62, 5)
        val_x   = scaler.transform(val_x.reshape(len(val_x), -1)).reshape(len(val_x), 62, 5)
        test_x  = scaler.transform(test_x.reshape(len(test_x), -1)).reshape(len(test_x), 62, 5)

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

        model = glfnet_mlp(out_dim=40, emb_dim=64, input_dim=310).to(device)
        top1, top5, preds, labels = train_and_eval(model, train_loader, val_loader, test_loader, device)

        print("Test Top-1:", top1)
        print("Test Top-5:", top5)
        print(classification_report(labels, preds))

        conf = np.zeros((40,40), dtype=int)
        for t,p in zip(labels, preds):
            conf[t][p] += 1
        print("Confusion matrix:\n", conf)

        sort_choice = input("Sort per-class accuracies? (y/n): ").strip().lower()
        stats = per_class_stats(conf, sort=(sort_choice=="y"))
        print("\nPer-class accuracies:")
        for cid, acc in stats:
            print(f"Class {cid}: {acc:.3f}")

    else:
        # cross-validation
        top1_scores, top5_scores = [], []
        all_preds, all_labels = [], []

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

            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x.reshape(len(train_x), -1)).reshape(len(train_x), 62, 5)
            val_x   = scaler.transform(val_x.reshape(len(val_x), -1)).reshape(len(val_x), 62, 5)
            test_x  = scaler.transform(test_x.reshape(len(test_x), -1)).reshape(len(test_x), 62, 5)

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

            model = glfnet_mlp(out_dim=40, emb_dim=64, input_dim=310).to(device)
            top1, top5, preds, labels = train_and_eval(model, train_loader, val_loader, test_loader, device)
            top1_scores.append(top1); top5_scores.append(top5)
            all_preds.extend(preds); all_labels.extend(labels)

            print(f"Fold {test_block}: top1={top1:.3f}, top5={top5:.3f}")

        print("\n=== Cross-validation summary ===")
        print("Mean Top-1:", np.mean(top1_scores), "Std:", np.std(top1_scores))
        print("Mean Top-5:", np.mean(top5_scores), "Std:", np.std(top5_scores))
        print(classification_report(all_labels, all_preds))

        conf = np.zeros((40,40), dtype=int)
        for t,p in zip(all_labels, all_preds):
            conf[t][p] += 1
        print("Confusion matrix:\n", conf)

        sort_choice = input("Sort per-class accuracies? (y/n): ").strip().lower()
        stats = per_class_stats(conf, sort=(sort_choice=="y"))
        print("\nPer-class accuracies:")
        for cid, acc in stats:
            print(f"Class {cid}: {acc:.3f}")

if __name__ == "__main__":
    main()
