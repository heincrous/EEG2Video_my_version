# ==========================================
# EEG2Video Seq2Seq Training
# ==========================================
import os, math, joblib
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
batch_size       = 32
num_epochs       = 200
lr               = 0.0005
run_device       = "cuda"

# default is subject 1 only; set to True to use all subjects in folder
USE_ALL_SUBJECTS = "sub1.npy"

# restrict to certain classes (0–39); set to None for all
CLASS_SUBSET     = [1, 10, 12, 16, 19, 23, 25, 31, 34, 39]

# paths
EEG_DIR          = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_windows_100"
LATENT_DIR       = "/content/drive/MyDrive/EEG2Video_data/processed/Video_latents"
SEQ2SEQ_CKPT_DIR = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_checkpoints"


# ==========================================
# EEG Encoder
# ==========================================
class MyEEGNet_embedding(nn.Module):
    def __init__(self, d_model=128, C=62, T=100, F1=16, D=4, F2=16, cross_subject=False):
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
        return self.dropout(x + self.pe[:, : x.size(1)].requires_grad_(False))


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
        # src: (batch, 7, 62, 100) windows
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
    def __len__(self): return self.eeg.shape[0]
    def __getitem__(self, idx): return self.eeg[idx], self.video[idx]


# ==========================================
# Train function
# ==========================================
def train_subject(subname):
    eeg_path    = os.path.join(EEG_DIR, subname)
    latent_path = os.path.join(LATENT_DIR, "Video_latents.npy")

    eegdata    = np.load(eeg_path)      # (7,40,5,7,62,5,100)
    latentdata  = np.load(latent_path)  # (7,40,5,6,4,36,64)

    # rearrange: collapse blocks, classes, trials
    EEG   = rearrange(eegdata,    "g p d w c f l -> (g p d) w c l")   # (1400,7,62,100)
    VIDEO = rearrange(latentdata, "g p d f c h w -> (g p d) f c h w") # (1400,6,4,36,64)

    # labels 0–39
    labels_block = np.repeat(np.arange(40), 5)   # 200 per block
    labels_all   = np.tile(labels_block, 7)      # 1400 total

    # apply class filtering
    if CLASS_SUBSET is not None:
        mask = np.isin(labels_all, CLASS_SUBSET)
        EEG, VIDEO, labels_all = EEG[mask], VIDEO[mask], labels_all[mask]

    # block-based split
    samples_per_block = (len(CLASS_SUBSET) if CLASS_SUBSET else 40) * 5
    train_idx = np.arange(0, 5*samples_per_block)
    val_idx   = np.arange(5*samples_per_block, 6*samples_per_block)
    test_idx  = np.arange(6*samples_per_block, 7*samples_per_block)

    EEG_train, EEG_val, EEG_test = EEG[train_idx], EEG[val_idx], EEG[test_idx]
    VID_train, VID_val, VID_test = VIDEO[train_idx], VIDEO[val_idx], VIDEO[test_idx]

    # scale EEG (flatten windows)
    b, w, c, l = EEG_train.shape
    scaler = StandardScaler().fit(EEG_train.reshape(b, -1))
    def scale(arr):
        return torch.from_numpy(scaler.transform(arr.reshape(arr.shape[0], -1))).reshape(arr.shape[0], w, c, l)
    EEG_train, EEG_val, EEG_test = map(scale, [EEG_train, EEG_val, EEG_test])

    # build datasets
    train_set = EEGVideoDataset(EEG_train, VID_train)
    val_set   = EEGVideoDataset(EEG_val,   VID_val)
    test_set  = EEGVideoDataset(EEG_test,  VID_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

    # model + optimizer
    model = myTransformer().to(run_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))
    criterion = nn.MSELoss()

    best_loss, best_state = float("inf"), None
    for epoch in tqdm(range(num_epochs), desc=f"Training {subname}"):
        model.train()
        epoch_loss = 0
        for eeg, video in train_loader:
            eeg = eeg.float().to(run_device)
            b, f, c, h, w = video.shape
            padded_video = torch.zeros((b, 1, c, h, w))
            full_video   = torch.cat((padded_video, video), dim=1).float().to(run_device)

            optimizer.zero_grad()
            _, out = model(eeg, full_video)
            video = video.float().to(run_device)
            loss  = criterion(video, out[:, :-1, :])
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        print(f"[{epoch}] Loss: {epoch_loss:.6f}")
        if epoch_loss < best_loss:
            best_loss, best_state = epoch_loss, model.state_dict()

    # save best model + scaler
    os.makedirs(SEQ2SEQ_CKPT_DIR, exist_ok=True)
    subset_tag = ""
    if CLASS_SUBSET is not None:
        subset_tag = "_subset" + "-".join(str(c) for c in CLASS_SUBSET)

    base_name = subname.replace(".npy","") + subset_tag
    ckpt_name = f"seq2seq_{base_name}.pt"
    scaler_name = f"scaler_{base_name}.pkl"

    torch.save({"state_dict": best_state}, os.path.join(SEQ2SEQ_CKPT_DIR, ckpt_name))
    joblib.dump(scaler, os.path.join(SEQ2SEQ_CKPT_DIR, scaler_name))

    print(f"Saved checkpoint: {ckpt_name}")
    print(f"Saved scaler: {scaler_name}")


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    sub_list = os.listdir(EEG_DIR) if USE_ALL_SUBJECTS else [subject_name]
    for sub in sub_list:
        if sub.endswith(".npy"):
            train_subject(sub)
