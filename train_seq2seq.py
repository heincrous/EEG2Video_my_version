# ==========================================
# EEG2Video Seq2Seq Training (Processed Data, Authors' Logic)
# ==========================================
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from einops import rearrange


# ==========================================
# Config
# ==========================================
config = {
    "drive_root": "/content/drive/MyDrive/EEG2Video_data/processed",
    "ckpt_root":  "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_checkpoints",
    "subject": "sub1.npy",
    "batch_size": 32,
    "num_epochs": 200,
    "lr": 5e-4,
}


# ==========================================
# EEG Encoder
# ==========================================
class MyEEGNet_embedding(nn.Module):
    def __init__(self, d_model=128, C=62, T=200, F1=16, D=4, F2=16, cross_subject=False):
        super().__init__()
        self.drop_out = 0.25 if cross_subject else 0.5
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(1, F1, (1, 64), bias=False),
            nn.BatchNorm2d(F1)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.drop_out)
        )
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
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
        return self.embedding(x)


# ==========================================
# Positional Encoding
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


# ==========================================
# Transformer Seq2Seq
# ==========================================
class myTransformer(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.img_embedding = nn.Linear(4 * 36 * 64, d_model)
        self.eeg_embedding = MyEEGNet_embedding(d_model=d_model, C=62, T=100)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=2
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=4
        )

        self.positional_encoding = PositionalEncoding(d_model, dropout=0)
        self.txtpredictor = nn.Linear(d_model, 13)
        self.predictor = nn.Linear(d_model, 4 * 36 * 64)

    def forward(self, src, tgt):
        src = self.eeg_embedding(src.reshape(src.shape[0] * src.shape[1], 1, 62, 100)).reshape(src.shape[0], 7, -1)
        tgt = tgt.reshape(tgt.shape[0], tgt.shape[1], -1)
        tgt = self.img_embedding(tgt)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-2]).to(tgt.device)
        encoder_output = self.transformer_encoder(src)

        new_tgt = torch.zeros((tgt.shape[0], 1, tgt.shape[2])).to(tgt.device)
        for i in range(6):
            decoder_output = self.transformer_decoder(new_tgt, encoder_output, tgt_mask=tgt_mask[:i+1, :i+1])
            new_tgt = torch.cat((new_tgt, decoder_output[:, -1:, :]), dim=1)

        encoder_output = torch.mean(encoder_output, dim=1)
        return self.txtpredictor(encoder_output), self.predictor(new_tgt).reshape(new_tgt.shape[0], new_tgt.shape[1], 4, 36, 64)


# ==========================================
# Dataset
# ==========================================
class EEGVideoDataset(torch.utils.data.Dataset):
    def __init__(self, eeg, video):
        self.eeg = eeg
        self.video = video
        self.len = eeg.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.eeg[idx], self.video[idx]


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    eeg_path    = f"{config['drive_root']}/EEG_windows_100/{config['subject']}"
    latent_path = f"{config['drive_root']}/Video_latents/{config['subject'].replace('.npy','_latents.npy')}"
    save_ckpt   = f"{config['ckpt_root']}/seq2seq_{config['subject'].replace('.npy','')}.pt"

    eegdata = np.load(eeg_path)          # (7, 40, 5, 62, 100)
    latent_data = np.load(latent_path)   # (7, 40, 5, 6, 4, 36, 64)

    # Flatten blocks and trials into batch
    EEG = rearrange(eegdata, "g p d c l -> (g p d) 7 c l")
    VIDEO = rearrange(latent_data, "g p d f c h w -> (g p d) f c h w")

    # Standardize EEG across all samples
    b, seq, c, l = EEG.shape
    EEG_reshaped = EEG.numpy().reshape(b, -1)
    scaler = StandardScaler().fit(EEG_reshaped)
    EEG_scaled = scaler.transform(EEG_reshaped)
    EEG = torch.from_numpy(EEG_scaled).reshape(b, seq, c, l)

    dataset = EEGVideoDataset(EEG, VIDEO)
    train_dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    model = myTransformer().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"] * len(train_dataloader))
    criterion = nn.MSELoss()

    for epoch in tqdm(range(config["num_epochs"])):
        model.train()
        epoch_loss = 0
        for eeg, video in train_dataloader:
            eeg = eeg.float().cuda()
            b, f, c, h, w = video.shape
            padded_video = torch.zeros((b, 1, c, h, w))
            full_video = torch.cat((padded_video, video), dim=1).float().cuda()

            optimizer.zero_grad()
            _, out = model(eeg, full_video)
            video = video.float().cuda()
            loss = criterion(video, out[:, :-1, :])
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        print(f"[{epoch}] Loss: {epoch_loss:.6f}")

    torch.save({'state_dict': model.state_dict()}, save_ckpt)
    print(f"Saved seq2seq checkpoint to {save_ckpt}")
