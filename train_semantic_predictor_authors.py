import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import preprocessing
import torch.nn.functional as F
from tqdm import tqdm
import os

# ==========================================
# Config
# ==========================================
SUBJECT_NAME     = "sub1.npy"
CLASS_SUBSET     = [0, 2, 4, 10, 11, 12, 22, 26, 29, 37]   # set to None for all
NUM_EPOCHS       = 200
BATCH_SIZE       = 32
LR               = 5e-4
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

EEG_PATH_ROOT    = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_DE_1per2s"
CLIP_PATH        = "/content/drive/MyDrive/EEG2Video_data/processed/CLIP_embeddings/CLIP_embeddings.npy"
CKPT_DIR         = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"
EMB_DIR          = "/content/drive/MyDrive/EEG2Video_outputs/semantic_embeddings"


# ==========================================
# Model
# ==========================================
class SemanticPredictor(nn.Module):
    def __init__(self):
        super(SemanticPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(310, 10000), nn.ReLU(),
            nn.Linear(10000, 10000), nn.ReLU(),
            nn.Linear(10000, 10000), nn.ReLU(),
            nn.Linear(10000, 10000), nn.ReLU(),
            nn.Linear(10000, 77 * 768)
        )
    def forward(self, eeg):
        return self.mlp(eeg)


# ==========================================
# Dataset
# ==========================================
class EEGTextDataset(torch.utils.data.Dataset):
    def __init__(self, eeg, text, labels):
        self.eeg = eeg
        self.text = text
        self.labels = labels
    def __len__(self):
        return len(self.eeg)
    def __getitem__(self, idx):
        return self.eeg[idx], self.text[idx], self.labels[idx]


# ==========================================
# Cleanup helper
# ==========================================
def clear_old(feature_type, subject_name, subset):
    subset_tag = ""
    if subset is not None:
        subset_tag = "_subset" + "-".join(str(c) for c in subset)
    ckpt_name = f"semantic_predictor_{feature_type}_{subject_name.replace('.npy','')}{subset_tag}.pt"
    emb_name  = f"embeddings_{feature_type}_{subject_name.replace('.npy','')}{subset_tag}.npy"
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
    emb_path  = os.path.join(EMB_DIR, emb_name)
    for path in [ckpt_path, emb_path]:
        if os.path.exists(path):
            os.remove(path)
            print(f"🧹 Deleted old file: {path}")
        else:
            print(f"— No existing file found: {path}")
    return ckpt_path, emb_path


# ==========================================
# Evaluation
# ==========================================
def evaluate(model, eeg, text, labels):
    model.eval()
    cos = nn.CosineSimilarity(dim=-1)
    with torch.no_grad():
        eeg_tensor = torch.tensor(eeg, dtype=torch.float32).to(DEVICE)
        text_tensor = torch.tensor(text, dtype=torch.float32).to(DEVICE)
        preds = model(eeg_tensor)

        # Cosine similarity
        preds_norm = F.normalize(preds, dim=-1)
        text_norm = F.normalize(text_tensor, dim=-1)
        cos_sim = cos(preds_norm, text_norm).mean().item()

        # Fisher score
        preds_np = preds.cpu().numpy()
        labels_np = np.array(labels)
        classes = np.unique(labels_np)
        class_samples = [preds_np[labels_np == c] for c in classes]
        class_means = [c.mean(axis=0) for c in class_samples]
        overall_mean = np.mean(class_means, axis=0)
        between = sum(len(c) * np.sum((m - overall_mean) ** 2) for c, m in zip(class_samples, class_means))
        within  = sum(np.sum((c - m) ** 2) for c, m in zip(class_samples, class_means))
        fisher_score = between / (within + 1e-8)
    return cos_sim, fisher_score


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    feature_type = "DE_1per2s"
    eeg_path = os.path.join(EEG_PATH_ROOT, SUBJECT_NAME)
    eegdata = np.load(eeg_path)             # (7,40,5,62,5)
    text_embeds = np.load(CLIP_PATH)        # (7,40,5,77,768)
    print("EEG:", eegdata.shape, "| Text:", text_embeds.shape)

    # Flatten
    eegdata = eegdata.reshape(-1, 62*5)            # (1400, 310)
    text_embeds = text_embeds.reshape(-1, 77*768)  # (1400, 59136)
    labels = np.tile(np.repeat(np.arange(40), 5), 7)

    # Split: first 6 blocks train, last block test
    block_size = 40 * 5
    train_idx = np.arange(0, 6 * block_size)
    test_idx  = np.arange(6 * block_size, 7 * block_size)

    eeg_train, eeg_test = eegdata[train_idx], eegdata[test_idx]
    text_train, text_test = text_embeds[train_idx], text_embeds[test_idx]
    labels_train, labels_test = labels[train_idx], labels[test_idx]

    # Apply subset
    if CLASS_SUBSET is not None:
        mask_train = np.isin(labels_train, CLASS_SUBSET)
        mask_test  = np.isin(labels_test, CLASS_SUBSET)
        eeg_train, text_train, labels_train = eeg_train[mask_train], text_train[mask_train], labels_train[mask_train]
        eeg_test, text_test, labels_test = eeg_test[mask_test], text_test[mask_test], labels_test[mask_test]
        print(f"Using subset {CLASS_SUBSET}: train {len(labels_train)}, test {len(labels_test)}")

    # Delete old files
    ckpt_path, emb_path = clear_old(feature_type, SUBJECT_NAME, CLASS_SUBSET)

    # Normalize
    scaler = preprocessing.StandardScaler()
    scaler.fit(eeg_train)
    eeg_train = scaler.transform(eeg_train)
    eeg_test = scaler.transform(eeg_test)

    # Dataloaders
    train_ds = EEGTextDataset(eeg_train, text_train, labels_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Model + optim
    model = SemanticPredictor().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * len(train_loader))

    # Train loop
    for epoch in tqdm(range(NUM_EPOCHS)):
        model.train()
        total_loss = 0
        for eeg, text, _ in train_loader:
            eeg, text = eeg.float().to(DEVICE), text.float().to(DEVICE)
            optimizer.zero_grad()
            pred = model(eeg)
            loss = F.mse_loss(pred, text)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Evaluate on test block
        cos_sim, fisher = evaluate(model, eeg_test, text_test, labels_test)
        print(f"[Epoch {epoch+1}] Train Loss={total_loss:.6f} | Test Cos={cos_sim:.4f} | Fisher={fisher:.4f}")

    # Save checkpoint
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    print(f"✅ Saved checkpoint: {ckpt_path}")

    # Save embeddings
    os.makedirs(EMB_DIR, exist_ok=True)
    model.eval()
    with torch.no_grad():
        eeg_tensor = torch.tensor(eeg_test, dtype=torch.float32).to(DEVICE)
        preds = model(eeg_tensor).cpu().numpy()
        np.save(emb_path, preds.astype(np.float32))
    print(f"✅ Saved semantic embeddings: {emb_path} | Shape: {preds.shape}")
