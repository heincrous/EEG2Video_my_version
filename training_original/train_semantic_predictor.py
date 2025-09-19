import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import preprocessing
from tqdm import tqdm
from einops import rearrange

from utils.gt_label import GT_LABEL  # shared GT_LABEL

# ----------------------------
# Semantic Predictor Network
# ----------------------------
class SemanticPredictor(nn.Module):
    def __init__(self, input_dim):
        super(SemanticPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 77 * 768)  # full CLIP hidden states
        )

    def forward(self, eeg):
        out = self.mlp(eeg)              # [B, 59136]
        out = out.view(-1, 77, 768)      # [B, 77, 768]
        return out

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
    # Paths
    eeg_data_path = "/content/drive/MyDrive/Data/Processed/EEG_timewindows_100/sub1.npy"
    text_embedding_path = "/content/drive/MyDrive/Data/Raw/text_embeddings_full.npy"
    save_path = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_predictor_full.pt"

    # Load data
    eegdata = np.load(eeg_data_path)  # shape: [blocks, classes, clips, 4, 62, 100]
    text_embedding = np.load(text_embedding_path)  # shape: [samples, 77, 768]

    print("EEG raw:", eegdata.shape)
    print("Text embedding raw:", text_embedding.shape)

    # Flatten EEG
    EEG = []
    for i in range(6):  # use first 6 blocks
        indices = [list(GT_LABEL[i]).index(element) for element in range(1, 41)]
        chosen_eeg = eegdata[i][indices, :]
        EEG.append(chosen_eeg)
    EEG = np.stack(EEG, axis=0)
    EEG = torch.from_numpy(EEG)
    EEG = rearrange(EEG, "a b c d e f -> (a b c) d (e f)")
    EEG = EEG.reshape(EEG.shape[0], -1)  # [N, features]
    EEG = EEG.numpy()

    print("EEG after reshape:", EEG.shape)

    # Flatten text embeddings
    Text = text_embedding[:EEG.shape[0], ...]   # [N, 77, 768]
    Text = Text.reshape(Text.shape[0], -1)      # [N, 59136]

    print("Text after reshape:", Text.shape)

    # Build dataset + dataloader
    dataset = Dataset(EEG, Text)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # smaller batch size

    # Model
    model = SemanticPredictor(input_dim=EEG.shape[1]).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200 * len(dataloader)
    )

    # Training loop
    for epoch in tqdm(range(30)):  # 30 epochs for quick test
        model.train()
        epoch_loss = 0
        for eeg, text in dataloader:
            eeg, text = eeg.cuda(), text.cuda()
            optimizer.zero_grad()
            eeg_embeddings = model(eeg).reshape(text.shape)  # [B, 59136]
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
