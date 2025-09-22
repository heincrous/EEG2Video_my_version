import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn import preprocessing
import torch.nn.functional as F
from tqdm import tqdm

# -------------------------------------------------------------------------
# Semantic Predictor (authors' MLP)
# -------------------------------------------------------------------------
class SemanticPredictor(nn.Module):
    def __init__(self):
        super(SemanticPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(310, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 77 * 768)
        )
    def forward(self, eeg):
        return self.mlp(eeg)

# -------------------------------------------------------------------------
# Dataset wrapper (EEG DE features + BLIP embeddings)
# -------------------------------------------------------------------------
class EEGTextDataset(Dataset):
    def __init__(self, eeg_list_path, text_list_path):
        with open(eeg_list_path, 'r') as f:
            eeg_files = [line.strip() for line in f.readlines()]
        with open(text_list_path, 'r') as f:
            text_files = [line.strip() for line in f.readlines()]

        assert len(eeg_files) == len(text_files), "Mismatch between EEG and text file counts"

        eeg_list, text_list = [], []
        for eeg_f, txt_f in zip(eeg_files, text_files):
            eeg = np.load(eeg_f)   # shape [310]
            txt = np.load(txt_f)   # shape [77,768]

            eeg_list.append(eeg.reshape(1, -1))
            text_list.append(txt.reshape(1, -1))

        self.eeg = np.concatenate(eeg_list, axis=0)
        self.text = np.concatenate(text_list, axis=0)

        # normalize EEG features
        scaler = preprocessing.StandardScaler().fit(self.eeg)
        self.eeg = scaler.transform(self.eeg)

    def __len__(self):
        return self.eeg.shape[0]

    def __getitem__(self, idx):
        return self.eeg[idx], self.text[idx]

# -------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------
if __name__ == "__main__":
    drive_root = "/content/drive/MyDrive/EEG2Video_data/processed"

    eeg_train_list  = os.path.join(drive_root, "EEG_features/train_list.txt")
    text_train_list = os.path.join(drive_root, "BLIP_embeddings/train_list.txt")

    dataset = EEGTextDataset(eeg_train_list, text_train_list)

    print("Sanity check:")
    print("EEG shape:", dataset.eeg.shape)    # (N, 310)
    print("Text shape:", dataset.text.shape)  # (N, 77*768)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SemanticPredictor().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200 * len(dataloader))

    for epoch in tqdm(range(20)): # originally 200 epochs
        model.train()
        epoch_loss = 0
        for eeg, text in dataloader:
            eeg = eeg.float().cuda()
            text = text.float().cuda()

            optimizer.zero_grad()
            pred = model(eeg)
            loss = F.mse_loss(pred, text)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}")

    save_dir = "/content/drive/MyDrive/EEG2Video_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'state_dict': model.state_dict()},
               os.path.join(save_dir, "semantic_predictor.pt"))
    print("Model saved to:", os.path.join(save_dir, "semantic_predictor.pt"))
