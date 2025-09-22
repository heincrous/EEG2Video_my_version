"""
SEQ2SEQ TRAINING WITH EEG WINDOWS → VIDEO LATENTS
-------------------------------------------------
- EEG_windows: subject-dependent, shape [7,62,100]
- Video_latents: subject-independent, shape [N,4,36,64]
- Train autoregressively: prepend zero frame, predict next frame
- Loss: MSE between predicted frames and ground truth frames
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from einops import rearrange
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# -------------------------
# EEG Encoder
# -------------------------
class MyEEGNet_embedding(nn.Module):
    def __init__(self, d_model=128, C=62, T=100, F1=16, D=4, F2=16, cross_subject=False):
        super(MyEEGNet_embedding, self).__init__()
        self.drop_out = 0.25 if cross_subject else 0.5

        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(1, F1, (1, 64), bias=False),
            nn.BatchNorm2d(F1)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(F1, F1*D, (C, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.drop_out)
        )
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(F1*D, F1*D, (1, 16), groups=F1*D, bias=False),
            nn.Conv2d(F1*D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.drop_out)
        )
        self.embedding = nn.Linear(48, d_model)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = x.view(x.shape[0], -1)
        x = self.embedding(x)
        return x

# -------------------------
# Positional Encoding
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

# -------------------------
# Seq2Seq Transformer
# -------------------------
class Seq2SeqEEG2Video(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.img_embedding = nn.Linear(4*36*64, d_model)
        self.eeg_embedding = MyEEGNet_embedding(d_model=d_model, C=62, T=100)

        self.positional_encoding = PositionalEncoding(d_model)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=2
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=4
        )

        self.predictor = nn.Linear(d_model, 4*36*64)

    def forward(self, eeg, video):
        # eeg: [B,7,62,100]
        b, n, c, t = eeg.shape
        eeg = eeg.reshape(b*n, 1, c, t)
        src = self.eeg_embedding(eeg).reshape(b, n, -1)  # [B,7,d_model]

        tgt = video.reshape(b, video.shape[1], -1)  # [B,F,4*36*64]
        tgt = self.img_embedding(tgt)

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        enc_out = self.encoder(src)
        dec_out = self.decoder(tgt, enc_out, tgt_mask=tgt_mask)

        out = self.predictor(dec_out).reshape(b, tgt.shape[1], 4, 36, 64)
        return out

# -------------------------
# Dataset
# -------------------------
class EEGVideoDataset(Dataset):
    def __init__(self, eeg_list, video_list):
        self.eeg_files = [l.strip() for l in open(eeg_list).readlines()]
        self.video_files = [l.strip() for l in open(video_list).readlines()]
        assert len(self.eeg_files) == len(self.video_files)
        self.scaler = None

        # fit scaler on EEG
        eeg_all = []
        for path in self.eeg_files:
            eeg = np.load(path)  # [7,62,100]
            eeg_all.append(eeg.reshape(-1, 100))
        eeg_all = np.vstack(eeg_all)
        self.scaler = StandardScaler().fit(eeg_all)

    def __len__(self):
        return len(self.eeg_files)

    def __getitem__(self, idx):
        eeg = np.load(self.eeg_files[idx])  # [7,62,100]
        video = np.load(self.video_files[idx])  # [F,4,36,64]

        # normalize EEG per window
        b, c, t = eeg.shape
        eeg = eeg.reshape(-1, t)
        eeg = self.scaler.transform(eeg)
        eeg = eeg.reshape(b, c, t)

        eeg = torch.tensor(eeg, dtype=torch.float32)
        video = torch.tensor(video, dtype=torch.float32)

        return eeg, video

# -------------------------
# Training
# -------------------------
if __name__ == "__main__":
    drive_root = "/content/drive/MyDrive/EEG2Video_data/processed"

    eeg_train_list = os.path.join(drive_root, "EEG_windows/train_list.txt")
    vid_train_list = os.path.join(drive_root, "Video_latents/train_list.txt")

    dataset = EEGVideoDataset(eeg_train_list, vid_train_list)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    model = Seq2SeqEEG2Video().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200*len(dataloader))
    criterion = nn.MSELoss()

    for epoch in range(20):
        model.train()
        epoch_loss = 0
        for eeg, video in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            eeg, video = eeg.cuda(), video.cuda()

            # autoregressive input
            b, f, c, h, w = video.shape
            zero_frame = torch.zeros((b,1,c,h,w), device=video.device)
            full_video = torch.cat([zero_frame, video], dim=1)  # prepend zero frame

            optimizer.zero_grad()
            out = model(eeg, full_video)  # predict [b,f+1,4,36,64]
            loss = criterion(out[:, :-1], video)  # align predicted seq to gt
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss={epoch_loss/len(dataloader):.6f}")

    save_path = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_model.pt"
    torch.save({"state_dict": model.state_dict()}, save_path)
    print("Model saved to:", save_path)
