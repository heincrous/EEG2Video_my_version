import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from tqdm import tqdm

# -------------------------------------------------------------------------
# Semantic Predictor (smaller MLP, still deep)
# -------------------------------------------------------------------------
class SemanticPredictor(nn.Module):
    def __init__(self):
        super(SemanticPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(310, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 77 * 768)
        )
    def forward(self, eeg):
        return self.mlp(eeg)

# -------------------------------------------------------------------------
# Dataset wrapper (lazy load to save RAM)
# -------------------------------------------------------------------------
class EEGTextDataset(Dataset):
    def __init__(self, eeg_list_path, text_list_path):
        with open(eeg_list_path, 'r') as f:
            self.eeg_files = [line.strip() for line in f.readlines()]
        with open(text_list_path, 'r') as f:
            self.text_files = [line.strip() for line in f.readlines()]

        assert len(self.eeg_files) == len(self.text_files), "Mismatch between EEG and text file counts"

        # fit scaler on EEG only (load once per file)
        eeg_all = []
        for eeg_f in self.eeg_files:
            eeg_all.append(np.load(eeg_f).reshape(-1))
        eeg_all = np.vstack(eeg_all)
        self.scaler = StandardScaler().fit(eeg_all)

    def __len__(self):
        return len(self.eeg_files)

    def __getitem__(self, idx):
        eeg = np.load(self.eeg_files[idx]).reshape(-1)
        txt = np.load(self.text_files[idx]).reshape(-1)
        eeg = self.scaler.transform([eeg])[0]
        return torch.tensor(eeg, dtype=torch.float32), torch.tensor(txt, dtype=torch.float32)

# -------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------
if __name__ == "__main__":
    drive_root = "/content/drive/MyDrive/EEG2Video_data/processed"

    eeg_train_list  = os.path.join(drive_root, "EEG_features/train_list.txt")
    text_train_list = os.path.join(drive_root, "BLIP_embeddings/train_list.txt")

    dataset = EEGTextDataset(eeg_train_list, text_train_list)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True,
                            pin_memory=True, num_workers=2)

    print("Sanity check:")
    eeg, txt = dataset[0]
    print("EEG sample shape:", eeg.shape)   # (310,)
    print("Text sample shape:", txt.shape)  # (77*768,)

    model = SemanticPredictor().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)  # match epochs

    for epoch in range(20):  # can scale to 200
        model.train()
        epoch_loss = 0
        for eeg, text in tqdm(dataloader):
            eeg, text = eeg.cuda(non_blocking=True), text.cuda(non_blocking=True)

            optimizer.zero_grad()
            pred = model(eeg)
            loss = F.mse_loss(pred, text)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1}: loss={epoch_loss/len(dataloader):.6f}")

    save_dir = "/content/drive/MyDrive/EEG2Video_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'state_dict': model.state_dict()},
               os.path.join(save_dir, "semantic_predictor.pt"))
    print("Model saved to:", os.path.join(save_dir, "semantic_predictor.pt"))
