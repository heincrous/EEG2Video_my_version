# ==========================================
# EEG → Video Seq2Seq Transformer
# ==========================================
import os
import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


# ==========================================
# EEGNet Embedding
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
# Transformer Model
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
        self.predictor = nn.Linear(512, 4 * 36 * 64)

    def forward(self, src, tgt):
        src = self.eeg_embedding(src.reshape(src.shape[0] * src.shape[1], 1, 62, 100))
        src = src.reshape(src.shape[0] // 7, 7, -1)

        tgt = tgt.reshape(tgt.shape[0], tgt.shape[1], -1)
        tgt = self.img_embedding(tgt)

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-2]).to(tgt.device)

        encoder_output = self.transformer_encoder(src)
        new_tgt = torch.zeros((tgt.shape[0], 1, tgt.shape[2]), device=tgt.device)

        for i in range(6):
            decoder_output = self.transformer_decoder(new_tgt, encoder_output, tgt_mask=tgt_mask[:i+1, :i+1])
            new_tgt = torch.cat((new_tgt, decoder_output[:, -1:, :]), dim=1)
        
        return self.predictor(new_tgt).reshape(
            new_tgt.shape[0], new_tgt.shape[1], 4, 36, 64
        )


# ==========================================
# Utility Functions
# ==========================================
def loss(true, pred):
    return nn.MSELoss()(true, pred)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, eeg, video):
        self.eeg = eeg
        self.video = video

    def __len__(self):
        return self.eeg.shape[0]

    def __getitem__(self, idx):
        return self.eeg[idx], self.video[idx]


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    EEG_PATH_ROOT = "/content/drive/MyDrive/EEG2Video_data/processed"
    FEATURE_TYPE = "EEG_windows_100"
    SUBJECT_NAME = "sub1.npy"

    eeg_path = os.path.join(EEG_PATH_ROOT, FEATURE_TYPE, SUBJECT_NAME)
    latent_path = os.path.join(EEG_PATH_ROOT, "Video_latents", "Video_latents.npy")

    print(f"Loading EEG from: {eeg_path}")
    print(f"Loading Latents from: {latent_path}")

    eeg_data = np.load(eeg_path, allow_pickle=True)
    latent_data = np.load(latent_path, allow_pickle=True)
    print(f"EEG shape: {eeg_data.shape}, Latent shape: {latent_data.shape}")

    train_eeg, test_eeg = eeg_data[:6], eeg_data[6:]
    train_lat, test_lat = latent_data[:6], latent_data[6:]

    CLASS_SUBSET = [0, 9, 11, 15, 18, 22, 24, 30, 33, 38]
    train_eeg, test_eeg = train_eeg[:, CLASS_SUBSET], test_eeg[:, CLASS_SUBSET]
    train_lat, test_lat = train_lat[:, CLASS_SUBSET], test_lat[:, CLASS_SUBSET]

    train_eeg = rearrange(train_eeg, "b c s w ch t -> (b c s) w ch t")
    test_eeg = rearrange(test_eeg, "b c s w ch t -> (b c s) w ch t")
    train_lat = rearrange(train_lat, "b c s f ch h w -> (b c s) f ch h w")
    test_lat = rearrange(test_lat, "b c s f ch h w -> (b c s) f ch h w")

    # === EEG normalization ===
    b, f, c, t = train_eeg.shape
    train_eeg_flat = train_eeg.reshape(b, f * c * t)
    test_eeg_flat = test_eeg.reshape(test_eeg.shape[0], -1)

    scaler = StandardScaler()
    scaler.fit(train_eeg_flat)
    train_eeg_scaled = scaler.transform(train_eeg_flat)
    test_eeg_scaled = scaler.transform(test_eeg_flat)

    train_eeg = train_eeg_scaled.reshape(b, f, c, t)
    test_eeg = test_eeg_scaled.reshape(test_eeg.shape[0], f, c, t)

    train_eeg = torch.from_numpy(train_eeg).float()
    test_eeg = torch.from_numpy(test_eeg).float()
    train_lat = torch.from_numpy(train_lat).float()
    test_lat = torch.from_numpy(test_lat).float()

    print(f"[EEG scaler] train mean={train_eeg.mean():.5f}, std={train_eeg.std():.5f}")
    print(f"[EEG scaler] test  mean={test_eeg.mean():.5f}, std={test_eeg.std():.5f}")

    dataset = Dataset(train_eeg, train_lat)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = myTransformer().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200 * len(dataloader))

    for epoch in tqdm(range(400)):
        model.train()
        epoch_loss = 0
        for eeg, video in dataloader:
            eeg = eeg.float().cuda()
            video = video.float().cuda()
            b, _, c, w, h = video.shape
            full_video = torch.cat((torch.zeros((b, 1, c, w, h), device='cuda'), video), dim=1)
            optimizer.zero_grad()
            out = model(eeg, full_video)
            l = loss(video, out[:, :-1, :])
            l.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += l.item()
        print(epoch_loss)

    # === Inference ===
    model.eval()
    test_latent = test_lat.float().cuda()
    test_eeg = test_eeg.float().cuda()
    b, _, c, w, h = test_latent.shape
    full_video = torch.cat((torch.zeros((b, 1, c, w, h), device='cuda'), test_latent), dim=1)
    out = model(test_eeg, full_video)
    latent_out = out[:, :-1, :].cpu().detach().numpy()

    np.save("/content/drive/MyDrive/EEG2Video_checkpoints/latent_out_block7_40_classes.npy", latent_out)
    torch.save({'state_dict': model.state_dict()}, "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seqmodel.pt")
