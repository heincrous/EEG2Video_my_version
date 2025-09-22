import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# -------------------------------------------------------------------------
# Model (paste MyEEGNet_embedding, PositionalEncoding, myTransformer here)
# -------------------------------------------------------------------------

def loss(true, pred):
    return F.mse_loss(pred, true)

# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------
class EEGVideoDataset(Dataset):
    def __init__(self, eeg_list_path, video_root, fit_scaler=True):
        # read EEG paths
        with open(eeg_list_path) as f:
            self.eeg_files = [line.strip() for line in f.readlines()]

        self.video_root = video_root
        self.pairs = []

        for eeg_path in self.eeg_files:
            # example eeg_path: .../EEG_windows/sub1/Block1/class22_clip03.npy
            rel = eeg_path.split("EEG_windows/")[-1]   # sub1/Block1/class22_clip03.npy
            rel_no_sub = "/".join(rel.split("/")[1:])  # Block1/class22_clip03.npy
            vid_path = os.path.join(video_root, rel_no_sub)
            if not os.path.exists(vid_path):
                raise FileNotFoundError(f"Video latent not found for {eeg_path}: {vid_path}")
            self.pairs.append((eeg_path, vid_path))

        # fit scaler over EEG (flattened)
        if fit_scaler:
            eeg_all = []
            for eeg_path, _ in self.pairs:
                eeg = np.load(eeg_path)  # shape [7,62,100]
                eeg_all.append(eeg.reshape(-1))
            eeg_all = np.vstack(eeg_all)
            self.scaler = StandardScaler().fit(eeg_all)
        else:
            self.scaler = None

        print(f"Train dataset with {len(self.pairs)} EEG–video pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        eeg_path, vid_path = self.pairs[idx]
        eeg = np.load(eeg_path)   # [7,62,100]
        vid = np.load(vid_path)   # [frames,4,36,64]

        eeg = eeg.reshape(-1)
        eeg = self.scaler.transform([eeg])[0]
        eeg = eeg.reshape(7,62,100)

        return torch.tensor(eeg, dtype=torch.float32), torch.tensor(vid, dtype=torch.float32)

# -------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------
if __name__ == "__main__":
    drive_root = "/content/drive/MyDrive/EEG2Video_data/processed"
    eeg_train_list = os.path.join(drive_root, "EEG_windows/train_list.txt")
    video_root     = os.path.join(drive_root, "Video_latents/train")

    train_dataset = EEGVideoDataset(eeg_train_list, video_root)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

    model = myTransformer().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200 * len(train_loader))

    for epoch in range(20):  # scale up later
        model.train()
        epoch_loss = 0
        for eeg, video in train_loader:
            eeg, video = eeg.cuda(), video.cuda()
            b, f, c, h, w = video.shape

            # pad dummy frame
            padded_video = torch.zeros((b,1,c,h,w), device=video.device)
            full_video = torch.cat((padded_video, video), dim=1)

            optimizer.zero_grad()
            _, out = model(eeg, full_video)
            l = loss(video, out[:, :-1, :])
            l.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += l.item()

        print(f"Epoch {epoch+1}: avg_loss={epoch_loss/len(train_loader):.6f}")

    save_dir = "/content/drive/MyDrive/EEG2Video_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "seq2seqmodel.pt")
    torch.save({'state_dict': model.state_dict()}, save_path)
    print("Model saved to:", save_path)
