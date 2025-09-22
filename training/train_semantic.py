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
    def __init__(self, list_path, eeg_root, text_root):
        with open(list_path, 'r') as f:
            files = [line.strip() for line in f.readlines()]

        eeg_list, text_list = [], []
        for f in files:
            base = os.path.basename(f).replace('.mp4', '.npy')
            eeg_path = os.path.join(eeg_root, base)
            text_path = os.path.join(text_root, base)

            eeg = np.load(eeg_path)  # DE features shape [310]
            txt = np.load(text_path) # BLIP->CLIP embedding [77,768]

            eeg_list.append(eeg.reshape(1, -1))
            text_list.append(txt.reshape(1, -1))

        self.eeg = np.concatenate(eeg_list, axis=0)
        self.text = np.concatenate(text_list, axis=0)

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
    drive_root = "/content/EEG2Video_my_version/processed"

    train_list = os.path.join(drive_root, "Video_mp4/train_list.txt")  # index list
    eeg_root   = os.path.join(drive_root, "EEG_features/train/sub1/Block1/..")  # adjust: use full train subdir
    text_root  = os.path.join(drive_root, "BLIP_embeddings/train/Block1/..")    # adjust accordingly

    dataset = EEGTextDataset(train_list, eeg_root, text_root)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SemanticPredictor().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200 * len(dataloader))

    for epoch in tqdm(range(200)):
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

    torch.save({'state_dict': model.state_dict()},
               os.path.join(drive_root, "semantic_predictor.pt"))
