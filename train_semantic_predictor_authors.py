import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import preprocessing
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
import os

# ==========================================
# Config
# ==========================================
SUBJECT_NAME   = "sub1.npy"
CLASS_SUBSET   = [0, 2, 4, 10, 11, 12, 22, 26, 29, 37]   # None for all
NUM_EPOCHS     = 50
BATCH_SIZE     = 32
LR             = 5e-4
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

EEG_PATH_ROOT  = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_DE_1per2s"
CLIP_PATH      = "/content/drive/MyDrive/EEG2Video_data/processed/CLIP_embeddings/CLIP_embeddings.npy"
CKPT_DIR       = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"


# ==========================================
# Model
# ==========================================
class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
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
class Dataset(torch.utils.data.Dataset):
    def __init__(self, eeg, text):
        self.eeg = eeg
        self.text = text
        self.len = eeg.shape[0]
    def __len__(self):
        return self.len
    def __getitem__(self, i):
        return self.eeg[i], self.text[i]


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    # Load EEG and CLIP data
    eeg_path = os.path.join(EEG_PATH_ROOT, SUBJECT_NAME)
    eegdata = np.load(eeg_path)             # shape: (7,40,5,62,5)
    textdata = np.load(CLIP_PATH)           # shape: (7,40,5,77,768)
    print("EEG:", eegdata.shape, "| Text:", textdata.shape)

    # Use first 6 blocks for training (authors’ setup)
    eeg = eegdata[:6]
    text = textdata[:6]

    # Flatten to (samples, features)
    eeg = rearrange(torch.from_numpy(eeg), 'a b c d e -> (a b c) (d e)').numpy()       # (1200, 310)
    text = rearrange(torch.from_numpy(text), 'a b c d e -> (a b c) (d e)').numpy()     # (1200, 59136)

    labels = np.tile(np.repeat(np.arange(40), 5), 6)

    # Apply subset
    if CLASS_SUBSET is not None:
        mask = np.isin(labels, CLASS_SUBSET)
        eeg, text, labels = eeg[mask], text[mask], labels[mask]
        print(f"Using subset {CLASS_SUBSET}: {len(labels)} samples")

    # Normalize EEG (authors normalize entire dataset, not per split)
    scaler = preprocessing.StandardScaler()
    eeg = scaler.fit_transform(eeg)

    # Dataset + loader
    dataset = Dataset(eeg, text)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model + optimizer + scheduler
    model = CLIP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * len(dataloader))

    # Training loop
    for epoch in tqdm(range(NUM_EPOCHS)):
        model.train()
        total_loss = 0
        for eeg_batch, text_batch in dataloader:
            eeg_batch = eeg_batch.float().to(DEVICE)
            text_batch = text_batch.float().to(DEVICE)
            optimizer.zero_grad()
            pred = model(eeg_batch)
            loss = F.mse_loss(pred, text_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"[{epoch+1}] Loss={total_loss:.6f}")

    # Save checkpoint
    os.makedirs(CKPT_DIR, exist_ok=True)
    subset_tag = "" if CLASS_SUBSET is None else "_subset" + "-".join(str(c) for c in CLASS_SUBSET)
    ckpt_name = f"semantic_predictor_DE_1per2s_{SUBJECT_NAME.replace('.npy','')}{subset_tag}.pt"
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    print(f"✅ Saved checkpoint: {ckpt_path}")
