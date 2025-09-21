# import numpy as np
# from DE_PSD import DE_PSD
# from tqdm import tqdm

# # Extract DE or PSD features with a 2-second window, that is, for each 2-second EEG segment, we extract a DE or PSD feature.
# # Input the shape of (7 * 40 * 5 * 62 * 2s*fre), meaning 7 blocks, 40 concepts, 5 video clips, 62 channels, and 2s*fre time-points.
# # Output the DE or PSD feature with (7 * 40 * 5 * 62 * 5), the last 5 indicates the frequency bands' number.

# fre = 200

# for subname in range(1,21):

#     loaded_data = np.load('data/EEG2Video/Segmented_Rawf_200Hz_2s/sub'+ str(subname) + '.npy')
#     # (7 * 40 * 5 * 62 * 2*fre)

#     print("Successfully loaded .npy file.")
#     print("Loaded data:")

#     DE_data = np.empty((0, 40, 5, 62, 5))
#     PSD_data = np.empty((0, 40, 5, 62, 5))

#     for block_id in range(7):
#         print("block: ", block_id)
#         now_data = loaded_data[block_id]
#         de_block_data = np.empty((0, 5, 62, 5))
#         psd_block_data = np.empty((0, 5, 62, 5))
#         for class_id in tqdm(range(40)):
#             de_class_data = np.empty((0, 62, 5))
#             psd_class_data = np.empty((0, 62, 5))
#             for i in range(5):
#                 de, psd = DE_PSD(now_data[class_id, i, :, :].reshape(62, 2*fre), fre, 2)
#                 de_class_data = np.concatenate((de_class_data, de.reshape(1, 62, 5)))
#                 psd_class_data = np.concatenate((psd_class_data, psd.reshape(1, 62, 5)))
#             de_block_data = np.concatenate((de_block_data, de_class_data.reshape(1, 5, 62, 5)))
#             psd_block_data = np.concatenate((psd_block_data, psd_class_data.reshape(1, 5, 62, 5)))
#         DE_data = np.concatenate((DE_data, de_block_data.reshape(1, 40, 5, 62, 5)))
#         PSD_data = np.concatenate((PSD_data, psd_block_data.reshape(1, 40, 5, 62, 5)))

#     np.save("data/EEG2Video/DE_1per2s/" + subname +".npy", DE_data)
#     np.save("data/EEG2Video/PSD_1per2s/" + subname + ".npy", PSD_data)

# ---------------------------------------------------------------------------------------------------------------
# NEW VERSION
# ---------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# -----------------------
# Model (authors’ MLP: DE -> BLIP embeddings)
# -----------------------
class CLIP(nn.Module):
    def __init__(self, input_dim=310):
        super(CLIP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 10000),
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

# -----------------------
# Dataset (per-clip DE features + BLIP embeddings)
# -----------------------
class EEGTextDataset(Dataset):
    def __init__(self, de_root, text_root):
        self.samples = []

        # collect (DE_file, BLIP_file) pairs
        for subj in os.listdir(de_root):
            subj_path = os.path.join(de_root, subj)
            if not os.path.isdir(subj_path):
                continue
            for block in os.listdir(subj_path):
                eeg_block = os.path.join(subj_path, block)
                txt_block = os.path.join(text_root, block)
                if not os.path.isdir(eeg_block) or not os.path.isdir(txt_block):
                    continue
                for f in os.listdir(eeg_block):
                    eeg_file = os.path.join(eeg_block, f)
                    txt_file = os.path.join(txt_block, f)
                    if os.path.exists(txt_file):
                        self.samples.append((eeg_file, txt_file))

        if len(self.samples) == 0:
            raise RuntimeError("No DE–BLIP pairs found. Check directory structure.")

        # fit StandardScaler across all clips
        all_eeg = []
        for eeg_file, _ in self.samples:
            eeg = np.load(eeg_file)        # (62,5)
            eeg = eeg.reshape(-1)          # flatten to (310,)
            all_eeg.append(eeg)
        all_eeg = np.stack(all_eeg, axis=0)  # (N,310)
        self.scaler = StandardScaler().fit(all_eeg)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        eeg_file, txt_file = self.samples[idx]

        eeg = np.load(eeg_file).reshape(-1)   # (310,)
        eeg = self.scaler.transform([eeg])[0]

        text = np.load(txt_file).reshape(-1)  # (77*768,)

        eeg = torch.tensor(eeg, dtype=torch.float32)
        text = torch.tensor(text, dtype=torch.float32)
        return eeg, text

# -----------------------
# Training loop
# -----------------------
if __name__ == "__main__":
    BASE = "/content/drive/MyDrive/EEG2Video_data/processed/Split_4train1test/train"
    CKPT_DIR = "/content/drive/MyDrive/EEG2Video_checkpoints"
    os.makedirs(CKPT_DIR, exist_ok=True)

    de_dir = os.path.join(BASE, "EEG_features/DE_1per2s")
    text_dir = os.path.join(BASE, "BLIP_embeddings")

    dataset = EEGTextDataset(de_dir, text_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIP(input_dim=310).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200 * len(dataloader)
    )

    epochs = int(input("Enter number of epochs: "))

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for eeg, text in dataloader:
            eeg, text = eeg.to(device), text.to(device)
            optimizer.zero_grad()
            preds = model(eeg)
            loss = F.mse_loss(preds, text)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

    ckpt_path = os.path.join(CKPT_DIR, "semantic_predictor.pt")
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    print(f"Training complete. Checkpoint saved to {ckpt_path}")