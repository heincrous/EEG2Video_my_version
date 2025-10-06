# ==========================================
# EEG → Video Latent Seq2Seq
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
LR               = 5e-4
P                = 0.5
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
        drop_out = P
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
        # Encode EEG sequence
        src = self.eeg_embedding(src.reshape(src.shape[0] * src.shape[1], 1, 62, 100)).reshape(src.shape[0], 7, -1)

        # Encode target latent sequence
        tgt = tgt.reshape(tgt.shape[0], tgt.shape[1], -1)
        tgt = self.img_embedding(tgt)

        # Apply positional encoding
        src = self.pos_enc(src)
        tgt = self.pos_enc(tgt)

        # Transformer encoder–decoder
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(-2)).to(tgt.device)
        encoder_output = self.encoder(src)
        new_tgt = torch.zeros((tgt.shape[0], 1, tgt.shape[2]), device=tgt.device)
        for i in range(6):
            decoder_output = self.decoder(new_tgt, encoder_output, tgt_mask=tgt_mask[:i+1, :i+1])
            new_tgt = torch.cat((new_tgt, decoder_output[:, -1:, :]), dim=1)

        return self.predictor(new_tgt).reshape(new_tgt.shape[0], new_tgt.shape[1], 4, 36, 64)


# ==========================================
# Data Loading and Shaping Utility
# ==========================================
def load_data():
    eeg_path = os.path.join(EEG_PATH_ROOT, FEATURE_TYPE, SUBJECT_NAME)
    print(f"Loading EEG features from: {FEATURE_TYPE}/{SUBJECT_NAME}")
    eeg_data = np.load(eeg_path, allow_pickle=True)
    latent_data = np.load(LATENT_PATH, allow_pickle=True)
    print(f"EEG shape: {eeg_data.shape}, Latent shape: {latent_data.shape}")
    return eeg_data, latent_data


def prepare_data(eeg_data, latent_data):
    # === Train/test split ===
    train_eeg, test_eeg = eeg_data[:6], eeg_data[6:]
    train_lat, test_lat = latent_data[:6], latent_data[6:]
    print(f"Split done → train_eeg {train_eeg.shape}, test_eeg {test_eeg.shape}")
    print(f"Split done → train_lat {train_lat.shape}, test_lat {test_lat.shape}")

    # === Subset selection ===
    if CLASS_SUBSET:
        train_eeg = train_eeg[:, CLASS_SUBSET]
        test_eeg  = test_eeg[:, CLASS_SUBSET]
        train_lat = train_lat[:, CLASS_SUBSET]
        test_lat  = test_lat[:, CLASS_SUBSET]
        print(f"Applied class subset ({len(CLASS_SUBSET)} classes).")

    # === Latent normalization (same as authors' 'normalizetion' function) ===
    def normalize_latents(x):
        x = torch.tensor(x, dtype=torch.float32)
        mean = torch.mean(x, dim=(0, 2, 3, 4), dtype=torch.float64)
        std  = torch.std(x, dim=(0, 2, 3, 4))
        normalized = (x - mean.view(1, 4, 1, 1, 1)) / std.view(1, 4, 1, 1, 1)
        return normalized

    train_lat = normalize_latents(train_lat)
    test_lat  = normalize_latents(test_lat)
    print(f"After latent norm → train mean={train_lat.mean():.5f}, std={train_lat.std():.5f}")
    print(f"After latent norm → test  mean={test_lat.mean():.5f}, std={test_lat.std():.5f}")

    # === EEG normalization (identical logic to authors) ===
    # Flatten windows and normalize across time dimension
    b, c, d, w, ch, t = train_eeg.shape
    train_eeg = rearrange(train_eeg, "b c d w ch t -> (b c d w) (ch t)")
    test_eeg  = rearrange(test_eeg,  "b c d w ch t -> (b c d w) (ch t)")
    scaler = StandardScaler().fit(train_eeg)
    train_eeg = scaler.transform(train_eeg)
    test_eeg  = scaler.transform(test_eeg)
    print(f"EEG scaler → train mean={np.mean(train_eeg):.6f}, std={np.std(train_eeg):.6f}")
    print(f"EEG scaler → test  mean={np.mean(test_eeg):.6f}, std={np.std(test_eeg):.6f}")

    # === Reshape EEG back to (samples, 1, 62, 100)
    train_eeg = rearrange(train_eeg, "(b c d w) (ch t) -> (b c d w) 1 ch t", b=6, c=len(CLASS_SUBSET), d=5, w=7, ch=62, t=100)
    test_eeg  = rearrange(test_eeg,  "(b c d w) (ch t) -> (b c d w) 1 ch t",  b=1, c=len(CLASS_SUBSET), d=5, w=7, ch=62, t=100)
    train_eeg = torch.from_numpy(train_eeg).float()
    test_eeg  = torch.from_numpy(test_eeg).float()
    print(f"EEG final shape → train {train_eeg.shape}, test {test_eeg.shape}")

    # === Reshape latents to (samples, frames, 4, 36, 64)
    train_lat = rearrange(train_lat, "b c s f ch h w -> (b c s) f ch h w")
    test_lat  = rearrange(test_lat,  "b c s f ch h w -> (b c s) f ch h w")
    print(f"Latent final shape → train {train_lat.shape}, test {test_lat.shape}")

    return train_eeg, test_eeg, train_lat, test_lat


# ==========================================
# Training and Evaluation Utility
# ==========================================
def train_model(model, dataloader, optimizer, scheduler, test_loader):
    criterion = nn.MSELoss()  # mean reduction, same as authors
    model.train()
    for epoch in tqdm(range(1, EPOCHS + 1)):
        epoch_loss = 0.0
        for eeg, video in dataloader:
            eeg = eeg.float().to(DEVICE)
            video = video.float().to(DEVICE)
            b, f, c, h, w = video.shape

            # prepend zero frame like authors
            padded = torch.zeros((b, 1, c, h, w), device=DEVICE)
            full_video = torch.cat((padded, video), dim=1)

            optimizer.zero_grad()
            out = model(eeg, full_video)  # your model only returns latents
            loss = criterion(out[:, :-1], video)  # predict 6 frames
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()  # once per epoch
        if epoch % 10 == 0:
            print(f"\n[Epoch {epoch:03d}/{EPOCHS}] Mean Loss: {epoch_loss:.6f}")
            evaluate_model(model, test_loader)



def evaluate_model(model, test_loader):
    criterion = nn.MSELoss()
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for eeg, video in test_loader:
            eeg = eeg.float().to(DEVICE)
            video = video.float().to(DEVICE)
            b, f, c, h, w = video.shape
            padded = torch.zeros((b, 1, c, h, w), device=DEVICE)
            full_video = torch.cat((padded, video), dim=1)

            out = model(eeg, full_video)
            total_loss += criterion(out[:, :-1], video).item()

    print(f"  Test MSE (mean): {total_loss:.6f}\n")


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
    preds = np.concatenate(preds)
    return preds


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

    print(f"Starting training for {FEATURE_TYPE} on subset {SUBSET_ID}...")
    model = train_model(model, train_loader, optimizer, scheduler, test_loader)
    print("Training complete.\n")
    evaluate_model(model, test_loader)

    latent_out = run_inference(model, test_loader)

    mean_val = latent_out.mean()
    std_val  = latent_out.std()

    np.save(
        os.path.join(
            EMB_SAVE_PATH,
            f"latent_out_{FEATURE_TYPE}_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}.npy"
        ),
        latent_out
    )

    torch.save(
        {"state_dict": model.state_dict()},
        os.path.join(
            CKPT_SAVE_PATH,
            f"seq2seq_{FEATURE_TYPE}_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}.pt"
        )
    )

    print(f"Saved → seq2seq_{FEATURE_TYPE}_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}.pt")
    print(f"Saved → latent_out_{FEATURE_TYPE}_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}.npy (shape: {latent_out.shape})")
    print(f"Latent stats → mean: {mean_val:.5f}, std: {std_val:.5f}")




