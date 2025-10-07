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
from torch.utils.data import TensorDataset
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

EPOCHS           = 300
BATCH_SIZE       = 32
LR               = 1e-4
P                = 0.2
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
        self.drop_out = P
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
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        encoder_output = self.encoder(src)

        new_tgt = torch.zeros((tgt.shape[0], 1, tgt.shape[2]), device=tgt.device)
        for i in range(6):
            decoder_output = self.decoder(new_tgt, encoder_output, tgt_mask=tgt_mask[:i+1, :i+1])
            new_tgt = torch.cat((new_tgt, decoder_output[:, -1:, :]), dim=1)

        # match authors’ output structure
        return self.predictor(new_tgt).reshape(new_tgt.shape[0], new_tgt.shape[1], 4, 36, 64)


# ==========================================
# Data Loading and Shaping Utility
# ==========================================
def load_data():
    eeg_path = os.path.join(EEG_PATH_ROOT, FEATURE_TYPE, SUBJECT_NAME)
    print(f"Loading EEG features from: {FEATURE_TYPE}/{SUBJECT_NAME}")
    eeg_data = np.load(eeg_path, allow_pickle=True)
    latent_data = np.load(LATENT_PATH, allow_pickle=True)
    print("[RAW latent stats] mean =", latent_data.mean(), "std =", latent_data.std())
    print(f"EEG shape: {eeg_data.shape}, Latent shape: {latent_data.shape}")
    return eeg_data, latent_data


def prepare_data(eeg_data, latent_data):
    # Split subjects (7 total)
    train_eeg, test_eeg = eeg_data[:6], eeg_data[6:]
    train_lat, test_lat = latent_data[:6], latent_data[6:]

    # Apply class subset if specified
    if CLASS_SUBSET:
        train_eeg = train_eeg[:, CLASS_SUBSET]
        test_eeg  = test_eeg[:, CLASS_SUBSET]
        train_lat = train_lat[:, CLASS_SUBSET]
        test_lat  = test_lat[:, CLASS_SUBSET]

    # Flatten EEG window hierarchy
    train_eeg = rearrange(train_eeg, "b c s w ch t -> (b c s) w ch t")
    test_eeg  = rearrange(test_eeg,  "b c s w ch t -> (b c s) w ch t")

    # === EEG normalization (exact author replication) ===
    b, w, ch, t = train_eeg.shape
    train_flat = train_eeg.reshape(-1, ch * t)
    test_flat  = test_eeg.reshape(-1, ch * t)

    scaler = StandardScaler()
    all_flat = np.concatenate([train_flat, test_flat], axis=0)
    scaler.fit(all_flat)
    train_eeg = scaler.transform(train_flat)
    test_eeg  = scaler.transform(test_flat)

    train_eeg = train_eeg.reshape(b, w, ch, t)
    test_eeg  = test_eeg.reshape(-1, w, ch, t)

    print(f"[EEG scaler] train mean={train_eeg.mean():.5f}, std={train_eeg.std():.5f}")
    print(f"[EEG scaler] test  mean={test_eeg.mean():.5f}, std={test_eeg.std():.5f}")

    # Convert to tensors
    train_eeg = torch.from_numpy(train_eeg).float()
    test_eeg  = torch.from_numpy(test_eeg).float()

    # === Latent reshaping ===
    train_lat = rearrange(train_lat, "b c s f ch h w -> (b c s) f ch h w")
    test_lat  = rearrange(test_lat,  "b c s f ch h w -> (b c s) f ch h w")

    # Convert to tensors before permuting
    train_lat = torch.tensor(train_lat, dtype=torch.float32)
    test_lat  = torch.tensor(test_lat, dtype=torch.float32)

    # === Latent normalization (authors’ method, per-channel not per-frame) ===
    # Our layout: (B, F=6, C=4, H=36, W=64)
    train_lat = train_lat.permute(0, 2, 1, 3, 4)  # (B, 4, 6, 36, 64)
    test_lat  = test_lat.permute(0, 2, 1, 3, 4)

    train_lat_mean = torch.mean(train_lat, dim=(0, 2, 3, 4))
    train_lat_std  = torch.std(train_lat, dim=(0, 2, 3, 4))

    train_lat = (train_lat - train_lat_mean.reshape(1, 4, 1, 1, 1)) / train_lat_std.reshape(1, 4, 1, 1, 1)
    test_lat  = (test_lat  - train_lat_mean.reshape(1, 4, 1, 1, 1)) / train_lat_std.reshape(1, 4, 1, 1, 1)

    # Return to original layout
    train_lat = train_lat.permute(0, 2, 1, 3, 4)  # (B, 6, 4, 36, 64)
    test_lat  = test_lat.permute(0, 2, 1, 3, 4)

    print(f"[Latent norm] train mean={train_lat.mean():.5f}, std={train_lat.std():.5f}")
    print(f"[Latent norm] test  mean={test_lat.mean():.5f}, std={test_lat.std():.5f}")

    return train_eeg, test_eeg, train_lat, test_lat


# ==========================================
# Training and Evaluation Utility
# ==========================================
def train_model(model, dataloader, optimizer, scheduler, test_loader):
    for epoch in tqdm(range(1, EPOCHS + 1)):
        model.train()
        epoch_mse, epoch_cos = 0.0, 0.0
        grad_norms, pred_means, pred_stds = [], [], []

        for step, (eeg, video) in enumerate(dataloader):
            eeg   = eeg.to(DEVICE, dtype=torch.float32)
            video = video.to(DEVICE, dtype=torch.float32)

            b, f, c, h, w = video.shape

            padded_video = torch.zeros((b, 1, c, h, w), device=DEVICE)
            full_video = torch.cat((padded_video, video), dim=1)

            optimizer.zero_grad()
            out = model(eeg, full_video)
            pred = out[:, :-1, :]

            # Combined loss = MSE + cosine penalty
            mse = F.mse_loss(pred, video, reduction='mean')
            cos = 1 - F.cosine_similarity(pred.flatten(1), video.flatten(1), dim=1).mean()
            var_reg = (pred.std() - 0.7).pow(2)

            # Warmup for first 50 epochs: prioritize cosine
            if epoch < 50:
                loss = 0.3 * mse + 2.0 * cos + 0.1 * var_reg
            else:
                loss = 0.7 * mse + 1.5 * cos + 0.05 * var_reg

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # record stats
            epoch_mse += mse.item()
            epoch_cos += 1 - cos.item()
            grad_norms.append(grad_norm.item())
            pred_means.append(pred.mean().item())
            pred_stds.append(pred.std().item())

            # --- lightweight inline diagnostics ---
            if step % 200 == 0:
                print(f"[Batch {step}] loss={loss.item():.4f} mse={mse.item():.4f} cos={1-cos.item():.4f} "
                      f"| pred μ={pred.mean():.4f}, σ={pred.std():.4f}, grad_norm={grad_norm.item():.4f}")

        # summarize epoch
        avg_mse = epoch_mse / len(dataloader)
        avg_cos = epoch_cos / len(dataloader)
        mean_grad = np.mean(grad_norms)
        mean_pred, std_pred = np.mean(pred_means), np.mean(pred_stds)

        print(f"\n[Epoch {epoch}] MSE={avg_mse:.6f} | Cosine={avg_cos:.6f} "
              f"| Pred μ={mean_pred:.4f}, σ={std_pred:.4f}, GradNorm={mean_grad:.4f}")

        # occasional evaluation
        if epoch % 10 == 0:
            print("-"*65)
            evaluate_model(model, test_loader)
            print("="*65 + "\n")
    return model


def evaluate_model(model, test_loader):
    model.eval()
    total_mse, total_cos, total_samples = 0.0, 0.0, 0
    pred_means, pred_stds = [], []

    with torch.no_grad():
        for eeg, video in test_loader:
            eeg   = eeg.to(DEVICE, dtype=torch.float32)
            video = video.to(DEVICE, dtype=torch.float32)
            b, f, c, h, w = video.shape
            total_samples += b

            full_video = torch.cat((torch.zeros((b, 1, c, h, w), device=DEVICE), video), dim=1)
            out = model(eeg, full_video)
            pred = out[:, :-1, :]

            mse = F.mse_loss(pred, video, reduction='mean')
            cos = F.cosine_similarity(pred.flatten(1), video.flatten(1), dim=1).mean()

            total_mse += mse.item() * b
            total_cos += cos.item() * b
            pred_means.append(pred.mean().item())
            pred_stds.append(pred.std().item())

    avg_mse = total_mse / total_samples
    avg_cos = total_cos / total_samples
    print(f"Eval → MSE={avg_mse:.6f} | Cosine={avg_cos:.6f} "
          f"| Pred μ={np.mean(pred_means):.4f}, σ={np.mean(pred_stds):.4f}\n")


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
            preds.append(out[:, 1:, :].cpu().numpy())  # keep last 6 frames only
    preds = np.concatenate(preds)
    return preds


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    cleanup_previous_run()
    eeg_data, latent_data = load_data()
    train_eeg, test_eeg, train_lat, test_lat = prepare_data(eeg_data, latent_data)

    train_loader = DataLoader(TensorDataset(train_eeg, train_lat), batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(TensorDataset(test_eeg, test_lat), batch_size=BATCH_SIZE, shuffle=False)

    model = Seq2SeqTransformer().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    print(f"Starting training for {FEATURE_TYPE} on subset {SUBSET_ID}...")
    model = train_model(model, train_loader, optimizer, scheduler, test_loader)
    print("✅ Training complete.\n")
    evaluate_model(model, test_loader)

    latent_out = run_inference(model, test_loader)
    mean_val, std_val = latent_out.mean(), latent_out.std()

    np.save(os.path.join(EMB_SAVE_PATH,
            f"latent_out_{FEATURE_TYPE}_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}.npy"),
            latent_out)

    torch.save({"state_dict": model.state_dict()},
            os.path.join(CKPT_SAVE_PATH,
            f"seq2seq_{FEATURE_TYPE}_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}.pt"))

    print(f"Saved → seq2seq_{FEATURE_TYPE}_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}.pt")
    print(f"Saved → latent_out_{FEATURE_TYPE}_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}.npy (shape: {latent_out.shape})")
    print(f"Latent stats → mean: {mean_val:.5f}, std: {std_val:.5f}")




