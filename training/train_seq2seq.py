import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import joblib

# ------------------------------------------------
# CONFIGURATION
# ------------------------------------------------
CONFIG = {
    # EEGNet params
    "d_model": 512,
    "C": 62,
    "T": 100,
    "F1": 32,
    "D": 4,
    "F2": 32,
    "cross_subject": False,

    # Transformer params
    "nhead": 8,
    "encoder_layers": 3,
    "decoder_layers": 6,

    # Training params
    "batch_size_all": 256,
    "batch_size_single": 128,
    "learning_rate": 3e-4,
    "epochs": 50,
    "loss_fn": nn.MSELoss,

    # Paths
    "bundle_dir": "/content/drive/MyDrive/EEG2Video_data/processed/SubjectBundles/",
    "save_root": "/content/drive/MyDrive/EEG2Video_checkpoints/",
}

# ------------------------------------------------
# Model components
# ------------------------------------------------
class MyEEGNet_embedding(nn.Module):
    def __init__(self, d_model, C, T, F1, D, F2, cross_subject=False):
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=5000):
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


class myTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.img_embedding = nn.Linear(4 * 36 * 64, cfg["d_model"])
        self.eeg_embedding = MyEEGNet_embedding(
            d_model=cfg["d_model"], C=cfg["C"], T=cfg["T"],
            F1=cfg["F1"], D=cfg["D"], F2=cfg["F2"],
            cross_subject=cfg["cross_subject"]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=cfg["d_model"], nhead=cfg["nhead"], batch_first=True),
            num_layers=cfg["encoder_layers"]
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=cfg["d_model"], nhead=cfg["nhead"], batch_first=True),
            num_layers=cfg["decoder_layers"]
        )
        self.positional_encoding = PositionalEncoding(cfg["d_model"], dropout=0)
        self.predictor = nn.Linear(cfg["d_model"], 4 * 36 * 64)

    def forward(self, src, tgt_frames):
        # EEG embedding
        src = self.eeg_embedding(src.reshape(src.shape[0]*src.shape[1], 1, 62, 100))
        src = src.reshape(src.shape[0]//7, 7, -1)
        src = self.positional_encoding(src)
        memory = self.transformer_encoder(src)

        # autoregressive decoding
        b, f, c, h, w = tgt_frames.shape
        outputs = []
        prev_tokens = torch.zeros((b,1,c*h*w), device=tgt_frames.device)
        for t in range(f):
            tgt_emb = self.img_embedding(prev_tokens)
            tgt_emb = self.positional_encoding(tgt_emb)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(1)).to(tgt_emb.device)
            out = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            pred = self.predictor(out[:, -1])  # last step prediction
            outputs.append(pred)
            prev_tokens = torch.cat([prev_tokens, pred.unsqueeze(1)], dim=1)
        outputs = torch.stack(outputs, dim=1)
        return outputs.view(b, f, c, h, w)

# ------------------------------------------------
# Dataset
# ------------------------------------------------
class EEGVideoBundle(Dataset):
    def __init__(self, eeg, vid):
        self.eeg = eeg
        self.vid = vid
    def __len__(self): return len(self.eeg)
    def __getitem__(self, idx):
        return torch.from_numpy(self.eeg[idx]).float(), torch.from_numpy(self.vid[idx]).float()


def scale_subject(train_path, test_path):
    train_npz = np.load(train_path, allow_pickle=True)
    test_npz  = np.load(test_path, allow_pickle=True)
    eeg_train = train_npz["EEG_windows"]
    eeg_test  = test_npz["EEG_windows"]
    vid_train = train_npz["Video_latents"]
    vid_test  = test_npz["Video_latents"]
    scaler = StandardScaler()
    eeg_flat = eeg_train.reshape(-1, 62*100)
    scaler.fit(eeg_flat)
    eeg_train = scaler.transform(eeg_flat).reshape(eeg_train.shape)
    eeg_test  = scaler.transform(eeg_test.reshape(-1, 62*100)).reshape(eeg_test.shape)
    return eeg_train, vid_train, eeg_test, vid_test, scaler


# ------------------------------------------------
# Main training
# ------------------------------------------------
if __name__ == "__main__":
    bundle_dir = CONFIG["bundle_dir"]
    save_root = CONFIG["save_root"]
    save_dir = os.path.join(save_root, "seq2seq_checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    all_bundles = sorted([f for f in os.listdir(bundle_dir) if f.endswith("_train.npz")])
    subjects = [f.replace("_train.npz","") for f in all_bundles]

    print("\nAvailable subjects:")
    for idx, subj in enumerate(subjects):
        print(f"{idx}: {subj}")

    choice = input("\nEnter subject indices to process (comma separated) or 'all': ").strip()
    num_epochs = int(input("\nEnter number of epochs: ") or CONFIG["epochs"])

    if choice.lower() == "all":
        print("\n=== Training on ALL subjects (per-subject normalization) ===")
        eeg_train_all, vid_train_all, eeg_test_all, vid_test_all, scalers = [], [], [], [], {}
        for subj in subjects:
            train_path = os.path.join(bundle_dir, f"{subj}_train.npz")
            test_path  = os.path.join(bundle_dir, f"{subj}_test.npz")
            eeg_train, vid_train, eeg_test, vid_test, scaler = scale_subject(train_path, test_path)
            eeg_train_all.append(eeg_train); vid_train_all.append(vid_train)
            eeg_test_all.append(eeg_test); vid_test_all.append(vid_test)
            scalers[subj] = scaler
        eeg_train = np.concatenate(eeg_train_all, axis=0)
        vid_train = np.concatenate(vid_train_all, axis=0)
        eeg_test  = np.concatenate(eeg_test_all, axis=0)
        vid_test  = np.concatenate(vid_test_all, axis=0)
        train_loader = DataLoader(EEGVideoBundle(eeg_train, vid_train), batch_size=CONFIG["batch_size_all"], shuffle=True)
        val_loader   = DataLoader(EEGVideoBundle(eeg_test, vid_test), batch_size=CONFIG["batch_size_all"], shuffle=False)
        model = myTransformer(CONFIG).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))
        loss_fn = CONFIG["loss_fn"]()
        for epoch in tqdm(range(num_epochs), desc="All-subject training"):
            model.train(); batch_losses = []
            for eeg, video in train_loader:
                eeg, video = eeg.cuda(), video.cuda()
                optimizer.zero_grad()
                out = model(eeg, video)
                loss = loss_fn(video, out)
                loss.backward()
                optimizer.step(); scheduler.step()
                batch_losses.append(loss.item())
            train_mean = np.mean(batch_losses)

            model.eval(); val_losses = []
            with torch.no_grad():
                for eeg, video in val_loader:
                    eeg, video = eeg.cuda(), video.cuda()
                    out = model(eeg, video)
                    val_losses.append(loss_fn(video, out).item())
            val_mean = np.mean(val_losses)
            print(f"All Epoch {epoch+1}/{num_epochs} | Train={train_mean:.4f} | Val={val_mean:.4f}")

        torch.save({'state_dict': model.state_dict()}, os.path.join(save_dir, "seq2seqmodel_all.pt"))
        joblib.dump(scalers, os.path.join(save_dir, "scalers_all.pkl"))
    else:
        selected_idx = [int(c.strip()) for c in choice.split(",") if c.strip().isdigit()]
        selected_subjects = [subjects[i] for i in selected_idx]
        for subj in selected_subjects:
            print(f"\n=== Training {subj} ===")
            train_path = os.path.join(bundle_dir, f"{subj}_train.npz")
            test_path  = os.path.join(bundle_dir, f"{subj}_test.npz")
            eeg_train, vid_train, eeg_test, vid_test, scaler = scale_subject(train_path, test_path)
            train_loader = DataLoader(EEGVideoBundle(eeg_train, vid_train), batch_size=CONFIG["batch_size_single"], shuffle=True)
            val_loader   = DataLoader(EEGVideoBundle(eeg_test, vid_test), batch_size=CONFIG["batch_size_single"], shuffle=False)
            model = myTransformer(CONFIG).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))
            loss_fn = CONFIG["loss_fn"]()
            for epoch in tqdm(range(num_epochs), desc=f"{subj} training"):
                model.train(); batch_losses = []
                for eeg, video in train_loader:
                    eeg, video = eeg.cuda(), video.cuda()
                    optimizer.zero_grad()
                    out = model(eeg, video)
                    loss = loss_fn(video, out)
                    loss.backward()
                    optimizer.step(); scheduler.step()
                    batch_losses.append(loss.item())
                train_mean = np.mean(batch_losses)

                model.eval(); val_losses = []
                with torch.no_grad():
                    for eeg, video in val_loader:
                        eeg, video = eeg.cuda(), video.cuda()
                        out = model(eeg, video)
                        val_losses.append(loss_fn(video, out).item())
                val_mean = np.mean(val_losses)
                print(f"{subj} Epoch {epoch+1}/{num_epochs} | Train={train_mean:.4f} | Val={val_mean:.4f}")
            torch.save({'state_dict': model.state_dict()}, os.path.join(save_dir, f"seq2seqmodel_{subj}.pt"))
            joblib.dump(scaler, os.path.join(save_dir, f"scaler_{subj}.pkl"))
