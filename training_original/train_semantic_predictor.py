import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import preprocessing
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
import os

from utils.gt_label import GT_LABEL  # use the shared GT_LABEL

# ----------------------------
# Semantic Predictor Network
# ----------------------------
class SemanticPredictor(nn.Module):
    def __init__(self, input_dim):
        super(SemanticPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 512)  # match your current text embeddings
        )

    def forward(self, eeg):
        return self.mlp(eeg)  # shape: (batch, 512)


# ----------------------------
# Dataset
# ----------------------------
class Dataset(torch.utils.data.Dataset):
    def __init__(self, eeg, text):
        scaler = preprocessing.StandardScaler().fit(eeg)
        eeg = scaler.transform(eeg)
        self.eeg = torch.tensor(eeg, dtype=torch.float32)
        self.text = torch.tensor(text, dtype=torch.float32)
        self.len = eeg.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.eeg[idx], self.text[idx]


# ----------------------------
# Training Loop
# ----------------------------
if __name__ == "__main__":
    # Paths (update for your Drive)
    eeg_data_path = "/content/drive/MyDrive/Data/Processed/EEG_timewindows_100/sub1.npy"
    text_embedding_path = "/content/drive/MyDrive/Data/Raw/text_embedding.npy"
    save_path = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_predictor.pt"

    # Load data
    eegdata = np.load(eeg_data_path)  # shape: [blocks, classes, clips, …]
    text_embedding = np.load(text_embedding_path)  # shape: [samples, 512]

    print("EEG raw:", eegdata.shape)
    print("Text embedding raw:", text_embedding.shape)

    EEG = []
    for i in range(6):  # use first 6 blocks
        indices = [list(GT_LABEL[i]).index(element) for element in range(1, 41)]  # classes 1–40
        chosen_eeg = eegdata[i][indices, :]
        EEG.append(chosen_eeg)
    EEG = np.stack(EEG, axis=0)
    EEG = torch.from_numpy(EEG)
    EEG = rearrange(EEG, "a b c d e f -> (a b c) d (e f)")  # [N, d, f]
    EEG = EEG.reshape(EEG.shape[0], -1)  # flatten to [N, features]

    print("EEG after reshape:", EEG.shape)

    # Match text embeddings to EEG sample count
    Text = text_embedding[:EEG.shape[0], ...]
    Text = torch.from_numpy(Text).reshape(Text.shape[0], -1)

    print("Text after reshape:", Text.shape)

    # Build dataset + dataloader
    dataset = Dataset(EEG, Text)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    model = SemanticPredictor(input_dim=EEG.shape[1]).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200 * len(dataloader)
    )

    # Training loop
    for epoch in tqdm(range(200)):
        model.train()
        epoch_loss = 0
        for eeg, text in dataloader:
            eeg, text = eeg.cuda(), text.cuda()
            optimizer.zero_grad()
            eeg_embeddings = model(eeg)
            loss = F.mse_loss(eeg_embeddings, text)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss={epoch_loss:.4f}")

    # Save checkpoint
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, save_path)
    print(f"Semantic predictor saved to {save_path}")
