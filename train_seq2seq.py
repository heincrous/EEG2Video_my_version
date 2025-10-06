# ==========================================
# EEG → Video Latent Seq2Seq (Author-Matched Training)
#
# Data shape reference:
# ------------------------------------------
# EEG (pre-windowed):        (7, 40, 5, 7, 62, 100)
# Latents (video features):  (7, 40, 5, 6, 4, 36, 64)
#
# After preprocessing in script:
# Train EEG  → (7*40*5, 7, 62, 100)
# Train Lat  → (7*40*5, 6, 4, 36, 64)
#
# Sequence alignment:
# - 7 EEG windows (temporal context)
# - 6 latent frames (video dynamics)
# - Goal: learn mapping EEG[7] → Latent[6]
# ==========================================
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from einops import rearrange
from tqdm import tqdm


# ==========================================
# Config
# ==========================================
FEATURE_TYPE     = "EEG_windows_100"
SUBJECT_NAME     = "sub1.npy"
CLASS_SUBSET     = [0, 9, 11, 15, 18, 22, 24, 30, 33, 38]
SUBSET_ID        = "1"

EPOCHS           = 200
BATCH_SIZE       = 32
LR               = 5e-4          # match authors
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

EEG_PATH_ROOT    = "/content/drive/MyDrive/EEG2Video_data/processed"
LATENT_PATH      = os.path.join(EEG_PATH_ROOT, "Video_latents", "Video_latents.npy")
CKPT_SAVE_PATH   = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_checkpoints"
EMB_SAVE_PATH    = "/content/drive/MyDrive/EEG2Video_outputs/seq2seq_latents"

os.makedirs(CKPT_SAVE_PATH, exist_ok=True)
os.makedirs(EMB_SAVE_PATH, exist_ok=True)


# ==========================================
# Clean-up Utility
# ==========================================
def cleanup_previous_run():
    prefix_ckpt = f"seq2seq_{FEATURE_TYPE}_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}"
    prefix_emb  = f"latent_out_{FEATURE_TYPE}_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}"

    deleted = 0
    for root, _, files in os.walk(CKPT_SAVE_PATH):
        for f in files:
            if f.startswith(prefix_ckpt):
                os.remove(os.path.join(root, f))
                deleted += 1
    for root, _, files in os.walk(EMB_SAVE_PATH):
        for f in files:
            if f.startswith(prefix_emb):
                os.remove(os.path.join(root, f))
                deleted += 1

    print(f"🧹 Deleted {deleted} old file(s) for subset {SUBSET_ID} ({FEATURE_TYPE}).")


# ==========================================
# EEGNet
# ==========================================
class MyEEGNet_embedding(nn.Module):
    def __init__(self, d_model=128, C=62, T=100, F1=16, D=4, F2=16):
        super(MyEEGNet_embedding, self).__init__()
        drop_out = 0.5
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
            nn.Dropout(drop_out)
        )
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(drop_out)
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
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)].requires_grad_(False))


# ==========================================
# Transformer
# ==========================================
class Seq2SeqTransformer(nn.Module):
    def __init__(self, d_model=512):
        super(Seq2SeqTransformer, self).__init__()
        self.eeg_embedding = MyEEGNet_embedding(d_model=d_model, C=62, T=100)
        self.img_embedding = nn.Linear(4 * 36 * 64, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=2
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=4
        )
        self.pos_enc = PositionalEncoding(d_model, dropout=0)
        self.predictor = nn.Linear(d_model, 4 * 36 * 64)

    def forward(self, src, tgt):
        # reorder so window (100) is last, matching author EEGNet input
        src = rearrange(src, "b w ch t -> b ch t w")
        src = self.eeg_embedding(src.reshape(src.shape[0] * src.shape[-1], 1, 62, 100))
        src = src.reshape(-1, 7, src.shape[-1])  # restore temporal order
        tgt = self.img_embedding(tgt.reshape(tgt.shape[0], tgt.shape[1], -1))
        src = self.pos_enc(src)
        tgt = self.pos_enc(tgt)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        encoder_output = self.encoder(src)
        new_tgt = torch.zeros((tgt.shape[0], 1, tgt.shape[2])).to(tgt.device)
        for i in range(6):
            decoder_output = self.decoder(new_tgt, encoder_output, tgt_mask=tgt_mask[:i+1, :i+1])
            new_tgt = torch.cat((new_tgt, decoder_output[:, -1:, :]), dim=1)
        return self.predictor(new_tgt).reshape(new_tgt.shape[0], new_tgt.shape[1], 4, 36, 64)


# ==========================================
# Data Loading and Shaping Utility
# ==========================================
def load_data():
    eeg_path = os.path.join(EEG_PATH_ROOT, FEATURE_TYPE, SUBJECT_NAME)
    print(f"Loading EEG from: {eeg_path}")
    eeg_data = np.load(eeg_path, allow_pickle=True)
    latent_data = np.load(LATENT_PATH, allow_pickle=True)
    print(f"EEG shape: {eeg_data.shape}, Latent shape: {latent_data.shape}")
    return eeg_data, latent_data


def prepare_data(eeg_data, latent_data):
    train_eeg, test_eeg = eeg_data[:6], eeg_data[6:]
    train_lat, test_lat = latent_data[:6], latent_data[6:]

    if CLASS_SUBSET:
        train_eeg = train_eeg[:, CLASS_SUBSET]
        test_eeg  = test_eeg[:, CLASS_SUBSET]
        train_lat = train_lat[:, CLASS_SUBSET]
        test_lat  = test_lat[:, CLASS_SUBSET]

    train_eeg = rearrange(train_eeg, "b c s w ch t -> (b c s) w ch t")
    test_eeg  = rearrange(test_eeg,  "b c s w ch t -> (b c s) w ch t")
    train_lat = rearrange(train_lat, "b c s f ch h w -> (b c s) f ch h w")
    test_lat  = rearrange(test_lat,  "b c s f ch h w -> (b c s) f ch h w")

    # EEG normalization same as authors (StandardScaler)
    scaler = StandardScaler().fit(train_eeg.reshape(train_eeg.shape[0], -1))
    train_eeg = torch.from_numpy(scaler.transform(train_eeg.reshape(train_eeg.shape[0], -1)).astype(np.float32)).reshape(train_eeg.shape)
    test_eeg  = torch.from_numpy(scaler.transform(test_eeg.reshape(test_eeg.shape[0], -1)).astype(np.float32)).reshape(test_eeg.shape)

    # Latent normalization identical to authors (per-channel mean/std)
    train_lat = torch.tensor(train_lat, dtype=torch.float32)
    test_lat  = torch.tensor(test_lat, dtype=torch.float32)
    mean = torch.mean(train_lat, dim=(0, 2, 3, 4), keepdim=True)
    std  = torch.std(train_lat, dim=(0, 2, 3, 4), keepdim=True)
    train_lat = (train_lat - mean) / (std + 1e-8)
    test_lat  = (test_lat - mean) / (std + 1e-8)

    return train_eeg, test_eeg, train_lat, test_lat


# ==========================================
# Training and Evaluation Utility
# ==========================================
def train_model(model, dataloader, optimizer, scheduler):
    model.train()
    for epoch in tqdm(range(1, EPOCHS + 1)):
        epoch_loss = 0
        for eeg, video in dataloader:
            eeg = eeg.float().to(DEVICE)
            video = video.float().to(DEVICE)
            b, f, c, h, w = video.shape
            full_video = torch.cat((torch.zeros((b, 1, c, h, w), device=DEVICE), video), dim=1)
            optimizer.zero_grad()
            out = model(eeg, full_video)
            loss = F.mse_loss(out[:, :-1], video)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0:
            print(f"[Epoch {epoch}/{EPOCHS}] Avg Loss: {epoch_loss / len(dataloader):.6f}")
    return model


def evaluate_model(model, test_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for eeg, video in test_loader:
            eeg = eeg.float().to(DEVICE)
            video = video.float().to(DEVICE)
            b, f, c, h, w = video.shape
            full_video = torch.cat((torch.zeros((b, 1, c, h, w), device=DEVICE), video), dim=1)
            out = model(eeg, full_video)
            loss = F.mse_loss(out[:, :-1], video)
            total_loss += loss.item()
    print(f"Final Test MSE: {total_loss / len(test_loader):.6f}")


# ==========================================
# Inference and Saving Utility
# ==========================================
def run_inference(model, test_loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for eeg, vid in test_loader:
            eeg = eeg.float().to(DEVICE)
            vid = vid.float().to(DEVICE)
            b, f, c, h, w = vid.shape
            full_vid = torch.cat((torch.zeros((b, 1, c, h, w), device=DEVICE), vid), 1)
            out = model(eeg, full_vid)
            preds.append(out[:, :-1].cpu().numpy())
    return np.concatenate(preds)


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    cleanup_previous_run()
    eeg_data, latent_data = load_data()
    train_eeg, test_eeg, train_lat, test_lat = prepare_data(eeg_data, latent_data)

    train_loader = DataLoader(list(zip(train_eeg, train_lat)), batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(list(zip(test_eeg, test_lat)), batch_size=BATCH_SIZE, shuffle=False)

    model = Seq2SeqTransformer().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader))

    print(f"🚀 Starting author-matched training for {EPOCHS} epochs...")
    model = train_model(model, train_loader, optimizer, scheduler)
    print("✅ Training complete.")
    evaluate_model(model, test_loader)

    latent_out = run_inference(model, test_loader)
    np.save(os.path.join(EMB_SAVE_PATH, f"latent_out_{FEATURE_TYPE}_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}.npy"), latent_out)
    torch.save({"state_dict": model.state_dict()},
               os.path.join(CKPT_SAVE_PATH, f"seq2seq_{FEATURE_TYPE}_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}.pt"))
    print("💾 Final checkpoint and latent outputs saved.")
