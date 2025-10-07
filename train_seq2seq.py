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
LR               = 5e-4
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
    def __init__(self, d_model=512, C=62, T=100, F1=16, D=4, F2=16):
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
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dropout=0.1, batch_first=True),
            num_layers=2
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=4, dropout=0.1, batch_first=True),
            num_layers=4
        )
        self.pos_enc = PositionalEncoding(d_model, dropout=0)
        self.predictor = nn.Linear(d_model, 4 * 36 * 64)
        self.teacher_forcing_ratio = 0.6

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
        src = src + 0.01 * torch.randn_like(src)
        tgt = tgt + 0.005 * torch.randn_like(tgt)
        encoder_output = self.encoder(src)

        # new_tgt = torch.zeros((tgt.shape[0], 1, tgt.shape[2]), device=tgt.device)
        # for i in range(6):
        #     decoder_output = self.decoder(new_tgt, encoder_output, tgt_mask=tgt_mask[:i+1, :i+1])
        #     new_tgt = torch.cat((new_tgt, decoder_output[:, -1:, :]), dim=1)

        # out = self.predictor(new_tgt)
        # out = F.layer_norm(out, out.shape[-1:])  # normalize per latent vector
        # return out.reshape(new_tgt.shape[0], new_tgt.shape[1], 4, 36, 64)

        # ------------------------------------------
        # 🧠 Teacher Forcing version
        # ------------------------------------------
        teacher_forcing_ratio = self.teacher_forcing_ratio

        new_tgt = torch.zeros((tgt.shape[0], 1, tgt.shape[2]), device=tgt.device)
        out_frames = []

        for i in range(tgt.shape[1]):  # 6 latent frames
            decoder_output = self.decoder(new_tgt, encoder_output, tgt_mask=tgt_mask[:i+1, :i+1])
            next_frame = self.predictor(decoder_output[:, -1:, :])

            # --- 🧠 Teacher forcing logic ---
            use_teacher = self.training and (torch.rand(1).item() < teacher_forcing_ratio)
            if use_teacher:
                next_input = tgt[:, i:i+1, :]  # use ground truth latent embedding
            else:
                next_input = decoder_output[:, -1:, :]  # use model prediction

            new_tgt = torch.cat((new_tgt, next_input), dim=1)
            out_frames.append(next_frame)

        # Combine predicted frames
        out = torch.cat(out_frames, dim=1)
        out = F.layer_norm(out, out.shape[-1:])  # normalize per latent vector
        return out.reshape(out.shape[0], out.shape[1], 4, 36, 64)


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

    train_lat = F.layer_norm(train_lat, train_lat.shape[-3:])
    test_lat  = F.layer_norm(test_lat,  test_lat.shape[-3:])

    return train_eeg, test_eeg, train_lat, test_lat


# ==========================================
# Training and Evaluation Utility
# ==========================================
def train_model(model, dataloader, optimizer, scheduler, test_loader):
    print(f"{'Epoch':<8}{'Train MSE':<12}{'Cos':<10}{'Grad':<10}{'μ':<10}{'σ':<10}")
    print("-"*60)
    for epoch in tqdm(range(1, EPOCHS + 1)):
        model.train()
        # Gradually reduce teacher forcing ratio each epoch
        if hasattr(model, 'teacher_forcing_ratio'):
            model.teacher_forcing_ratio = max(0.1, model.teacher_forcing_ratio * 0.99)

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

            mse = F.mse_loss(pred, video, reduction='mean')

            # Normalize for cosine
            pred_norm = F.normalize(pred.flatten(1), dim=1)
            video_norm = F.normalize(video.flatten(1), dim=1)
            cos = 1 - (pred_norm * video_norm).sum(dim=1).mean()

            # Variance regularization
            var_reg = (pred.std() - 1.0).pow(2)

            if epoch < 100:
                loss = mse + 0.5 * cos + 0.01 * var_reg
            elif epoch < 200:
                loss = 0.7 * mse + 0.3 * cos + 0.01 * var_reg
            else:
                loss = 0.5 * mse + 0.5 * cos + 0.01 * var_reg

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # record stats
            epoch_mse += mse.item()
            epoch_cos += 1 - cos.item()
            grad_norms.append(grad_norm.item())
            pred_means.append(pred.mean().item())
            pred_stds.append(pred.std().item())

        scheduler.step()

        # summarize epoch
        avg_mse = epoch_mse / len(dataloader)
        avg_cos = epoch_cos / len(dataloader)
        mean_grad = np.mean(grad_norms)
        mean_pred, std_pred = np.mean(pred_means), np.mean(pred_stds)

        print(f"{epoch:<8}{avg_mse:<12.4f}{avg_cos:<10.4f}{mean_grad:<10.3f}{mean_pred:<10.3f}{std_pred:<10.3f}")

        # === Diagnostic: EEG embedding variance ===
        with torch.no_grad():
            sample_eeg = eeg[:1].to(DEVICE)
            emb = model.eeg_embedding(sample_eeg.reshape(-1, 1, 62, 100))
            print(f"   [EEG emb σ={emb.std().item():.3f}, μ={emb.mean().item():.3f}]")

        # occasional evaluation
        if epoch % 10 == 0:
            print("\n" + "-"*60)
            evaluate_model(model, test_loader)
            print("-"*60 + "\n")
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
            pred_norm = F.normalize(pred.flatten(1), dim=1)
            video_norm = F.normalize(video.flatten(1), dim=1)
            cos = (pred_norm * video_norm).sum(dim=1).mean()

            total_mse += mse.item() * b
            total_cos += cos.item() * b
            pred_means.append(pred.mean().item())
            pred_stds.append(pred.std().item())

    avg_mse = total_mse / total_samples
    avg_cos = total_cos / total_samples
    print(f"📊 Eval ▶ MSE={avg_mse:.4f} | Cos={avg_cos:.4f} | μ={np.mean(pred_means):.3f} σ={np.mean(pred_stds):.3f}\n")


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
            preds.append(out.cpu().numpy())  # use all frames directly (already 6 frames)
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

    model = Seq2SeqTransformer(d_model=512).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

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




