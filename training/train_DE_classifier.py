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
# Main training routine
# -------------------------------------------------
def main():
    drive_root = "/content/drive/MyDrive/EEG2Video_data/processed"
    bundle_file = f"{drive_root}/BLIP_EEG_bundle.npz"
    subject_id = "sub1"  # change as needed

    data = np.load(bundle_file, allow_pickle=True)
    eeg_dict = data["EEG_data"].item()

    # DE features: shape (7 blocks, 40 classes, 5 clips, 62, 5)
    all_data = eeg_dict[subject_id]["EEG_DE"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = glfnet_mlp(out_dim=40, emb_dim=64, input_dim=310).to(device)

    # Use block 5 for val, block 6 for test, rest for train
    test_block = 6
    val_block = 5
    train_blocks = [i for i in range(7) if i not in [test_block, val_block]]

    train_x, train_y, val_x, val_y, test_x, test_y = [], [], [], [], [], []

    for b in train_blocks:
        for c in range(40):
            for k in range(5):
                train_x.append(all_data[b, c, k])
                train_y.append(c)
    for c in range(40):
        for k in range(5):
            val_x.append(all_data[val_block, c, k])
            val_y.append(c)
            test_x.append(all_data[test_block, c, k])
            test_y.append(c)

    train_x, val_x, test_x = np.array(train_x), np.array(val_x), np.array(test_x)
    train_y, val_y, test_y = np.array(train_y), np.array(val_y), np.array(test_y)

    # Normalize
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x.reshape(len(train_x), -1)).reshape(len(train_x), 62, 5)
    val_x = scaler.transform(val_x.reshape(len(val_x), -1)).reshape(len(val_x), 62, 5)
    test_x = scaler.transform(test_x.reshape(len(test_x), -1)).reshape(len(test_x), 62, 5)

    # DataLoaders
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

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    num_epochs = 100
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # ---- train ----
        model.train()
        total_loss, correct, total = 0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
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
            torch.save(model.state_dict(), "glfnet_sub1_best.pt")

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

    # ---- test ----
    model.load_state_dict(torch.load("glfnet_sub1_best.pt"))
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

    print("Test Top-1:", np.mean(top1_list))
    print("Test Top-5:", np.mean(top5_list))
    print(classification_report(all_labels, all_preds))

if __name__ == "__main__":
    main()
