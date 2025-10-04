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

# Paths
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
    def __init__(self, eeg, text):
        self.eeg = eeg
        self.text = text
    def __len__(self):
        return len(self.eeg)
    def __getitem__(self, idx):
        return self.eeg[idx], self.text[idx]


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

    # Labels
    labels = np.tile(np.repeat(np.arange(40), 5), 7)

    # Apply subset if defined
    if CLASS_SUBSET is not None:
        mask = np.isin(labels, CLASS_SUBSET)
        eegdata = eegdata[mask]
        text_embeds = text_embeds[mask]
        labels = labels[mask]
        print(f"Using subset {CLASS_SUBSET} → {len(labels)} samples")

    # Delete old files
    ckpt_path, emb_path = clear_old(feature_type, SUBJECT_NAME, CLASS_SUBSET)

    # Normalize EEG
    scaler = preprocessing.StandardScaler()
    eegdata = scaler.fit_transform(eegdata)

    # Dataloader
    dataset = EEGTextDataset(eegdata, text_embeds)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model and training setup
    model = SemanticPredictor().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * len(dataloader))

    # Train
    for epoch in tqdm(range(NUM_EPOCHS)):
        model.train()
        total_loss = 0
        for eeg, text in dataloader:
            eeg, text = eeg.float().to(DEVICE), text.float().to(DEVICE)
            optimizer.zero_grad()
            pred = model(eeg)
            loss = F.mse_loss(pred, text)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss = {total_loss:.6f}")

    # Save checkpoint
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    print(f"✅ Saved checkpoint: {ckpt_path}")

    # Save embeddings
    os.makedirs(EMB_DIR, exist_ok=True)
    model.eval()
    with torch.no_grad():
        eeg_tensor = torch.tensor(eegdata, dtype=torch.float32).to(DEVICE)
        preds = model(eeg_tensor).cpu().numpy()
        np.save(emb_path, preds.astype(np.float32))
    print(f"✅ Saved semantic embeddings: {emb_path} | Shape: {preds.shape}")
