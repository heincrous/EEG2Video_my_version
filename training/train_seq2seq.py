import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import joblib  # for saving scaler
from skimage.metrics import structural_similarity as ssim

# ------------------------------------------------
# Model components
# ------------------------------------------------
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
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class myTransformer(nn.Module):
    def __init__(self, d_model=512):
        super(myTransformer, self).__init__()
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
        self.txtpredictor = nn.Linear(512, 13)
        self.predictor = nn.Linear(512, 4 * 36 * 64)

    def forward(self, src, tgt):
        src = self.eeg_embedding(src.reshape(src.shape[0]*src.shape[1],1,62,100))
        src = src.reshape(src.shape[0]//7, 7, -1)

        tgt = tgt.reshape(tgt.shape[0], tgt.shape[1], -1)
        tgt = self.img_embedding(tgt)

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        encoder_output = self.transformer_encoder(src)

        new_tgt = torch.zeros((tgt.shape[0], 1, tgt.shape[2])).to(tgt.device)
        for i in range(6):
            decoder_output = self.transformer_decoder(new_tgt, encoder_output)
            new_tgt = torch.cat((new_tgt, decoder_output[:, -1:, :]), dim=1)

        encoder_output = torch.mean(encoder_output, dim=1)
        return self.txtpredictor(encoder_output), self.predictor(new_tgt).reshape(
            new_tgt.shape[0], new_tgt.shape[1], 4, 36, 64
        )


# ------------------------------------------------
# Dataset (subject-specific)
# ------------------------------------------------
class EEGVideoDataset(Dataset):
    def __init__(self, eeg_files, vid_files, eeg_dir, vid_dir, scaler=None, fit=False):
        self.eeg_dir = eeg_dir
        self.vid_dir = vid_dir
        self.scaler = scaler
        self.fit = fit

        # build video map
        self.vid_map = {os.path.normpath(v): v for v in vid_files}
        self.pairs = []
        for e in eeg_files:
            key = "/".join(e.split("/")[1:])  # <-- strip subject only
            if key in self.vid_map:
                self.pairs.append((e, self.vid_map[key]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        eeg_file, vid_file = self.pairs[idx]
        eeg = np.load(os.path.join(self.eeg_dir, eeg_file))   # [7,62,100]
        vid = np.load(os.path.join(self.vid_dir, vid_file))   # [6,4,36,64]

        eeg_flat = eeg.reshape(-1, 62*100)
        if self.scaler is not None:
            if self.fit:
                self.scaler.partial_fit(eeg_flat)
            eeg_flat = self.scaler.transform(eeg_flat)
        eeg = eeg_flat.reshape(eeg.shape)

        eeg = torch.from_numpy(eeg).float()
        vid = torch.from_numpy(vid).float()
        return eeg, vid


# ------------------------------------------------
# Main training per subject
# ------------------------------------------------
if __name__ == "__main__":
    base_dir = "/content/drive/MyDrive/EEG2Video_data/processed/"
    save_root = "/content/drive/MyDrive/EEG2Video_checkpoints/"
    save_dir = os.path.join(save_root, "seq2seq_checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    subjects = [s for s in sorted(os.listdir(os.path.join(base_dir,"EEG_windows"))) if s.startswith("sub")]

    print("\nAvailable subjects:")
    for idx, subj in enumerate(subjects):
        print(f"{idx}: {subj}")

    choice = input("\nEnter subject indices to process (comma separated): ")
    selected_idx = [int(c.strip()) for c in choice.split(",") if c.strip().isdigit()]
    selected_subjects = [subjects[i] for i in selected_idx]

    num_epochs = int(input("\nEnter number of epochs: "))

    with open(os.path.join(base_dir,"EEG_windows/train_list.txt")) as f:
        eeg_train_all = [line.strip() for line in f]
    with open(os.path.join(base_dir,"Video_latents/train_list.txt")) as f:
        vid_train_all = [line.strip() for line in f]
    with open(os.path.join(base_dir,"EEG_windows/test_list.txt")) as f:
        eeg_test_all = [line.strip() for line in f]
    with open(os.path.join(base_dir,"Video_latents/test_list.txt")) as f:
        vid_test_all = [line.strip() for line in f]

    for subj in selected_subjects:
        print(f"\n=== Training {subj} ===")

        eeg_train = [e for e in eeg_train_all if e.startswith(subj)]
        eeg_test = [e for e in eeg_test_all if e.startswith(subj)]

        scaler = StandardScaler()
        tmp_dataset = EEGVideoDataset(eeg_train, vid_train_all,
                                      os.path.join(base_dir,"EEG_windows"),
                                      os.path.join(base_dir,"Video_latents"),
                                      scaler=scaler, fit=True)
        for i in range(len(tmp_dataset)):
            _ = tmp_dataset[i]

        train_dataset = EEGVideoDataset(eeg_train, vid_train_all,
                                        os.path.join(base_dir,"EEG_windows"),
                                        os.path.join(base_dir,"Video_latents"),
                                        scaler=scaler, fit=False)
        val_dataset = EEGVideoDataset(eeg_test, vid_test_all,
                                      os.path.join(base_dir,"EEG_windows"),
                                      os.path.join(base_dir,"Video_latents"),
                                      scaler=scaler, fit=False)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        model = myTransformer().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs * len(train_loader)
        )
        loss_fn = nn.MSELoss()

        for epoch in tqdm(range(num_epochs), desc=f"{subj} training"):
            # ---- Training ----
            model.train()
            batch_losses = []
            for eeg, video in train_loader:
                eeg, video = eeg.cuda(), video.cuda()
                b, f, c, h, w = video.shape
                padded_video = torch.zeros((b,1,c,h,w)).cuda()
                full_video = torch.cat((padded_video, video), dim=1)
                optimizer.zero_grad()
                _, out = model(eeg, full_video)
                l = loss_fn(video, out[:, :-1, :])
                l.backward()
                optimizer.step()
                scheduler.step()
                batch_losses.append(l.item())
            train_mean = np.mean(batch_losses)
            train_std = np.std(batch_losses)

            # ---- Validation ----
            model.eval()
            val_losses, ssim_scores = [], []
            with torch.no_grad():
                for eeg, video in val_loader:
                    eeg, video = eeg.cuda(), video.cuda()
                    b, f, c, h, w = video.shape
                    padded_video = torch.zeros((b,1,c,h,w)).cuda()
                    full_video = torch.cat((padded_video, video), dim=1)
                    _, out = model(eeg, full_video)
                    loss_val = loss_fn(video, out[:, :-1, :]).item()
                    val_losses.append(loss_val)

                    out_np = out[:, :-1, :].cpu().numpy().reshape(b, f, c, h, w)
                    vid_np = video.cpu().numpy()
                    for i in range(b):
                        for t in range(f):
                            s = ssim(vid_np[i,t,0], out_np[i,t,0], data_range=1.0)
                            ssim_scores.append(s)

            val_mean = np.mean(val_losses)
            val_ssim = np.mean(ssim_scores)

            print(f"{subj} Epoch {epoch+1}/{num_epochs} "
                  f"| Train loss mean={train_mean:.4f}, std={train_std:.4f} "
                  f"| Val loss={val_mean:.4f}, SSIM={val_ssim:.4f}")

        ckpt_path = os.path.join(save_dir, f"seq2seqmodel_{subj}.pt")
        torch.save({'state_dict': model.state_dict()}, ckpt_path)

        scaler_path = os.path.join(save_dir, f"scaler_{subj}.pkl")
        joblib.dump(scaler, scaler_path)

        print(f"Saved checkpoint -> {ckpt_path}")
        print(f"Saved scaler     -> {scaler_path}")
