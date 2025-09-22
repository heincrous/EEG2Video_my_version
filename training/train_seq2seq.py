import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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
        x = self.embedding(x)
        return x

# -------------------------
# Positional Encoding
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
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
# Transformer
# -------------------------
class MyTransformer(nn.Module):
    def __init__(self, d_model=512, pred_frames=24):
        super(MyTransformer, self).__init__()
        self.pred_frames = pred_frames
        self.img_embedding = nn.Linear(4 * 36 * 64, d_model)
        self.eeg_embedding = MyEEGNet_embedding(d_model=d_model, C=62, T=100)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=2
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=4
        )

        self.positional_encoding = PositionalEncoding(d_model, dropout=0.0)
        self.predictor = nn.Linear(d_model, 4 * 36 * 64)

    def forward(self, eeg, start_token):
        # eeg: [B,7,62,100]
        b, n, c, t = eeg.shape
        eeg = eeg.reshape(b*n, 1, c, t)
        src = self.eeg_embedding(eeg).reshape(b, n, -1)  # [B,7,d_model]
        src = self.positional_encoding(src)
        enc_out = self.encoder(src)

        # start_token: [B,1,4,36,64] (usually zero frame)
        new_tgt = start_token.reshape(b, 1, -1)  # flatten
        new_tgt = self.img_embedding(new_tgt)
        new_tgt = self.positional_encoding(new_tgt)

        # rollout exactly F frames
        for i in range(self.pred_frames):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(new_tgt.size(1)).to(new_tgt.device)
            dec_out = self.decoder(new_tgt, enc_out, tgt_mask=tgt_mask)
            new_tgt = torch.cat((new_tgt, dec_out[:, -1:, :]), dim=1)

        out = self.predictor(new_tgt[:, 1:])  # drop the start token
        out = out.reshape(b, self.pred_frames, 4, 36, 64)
        return out

# -------------------------
# Dataset
# -------------------------
class EEGVideoDataset(Dataset):
    def __init__(self, eeg_list, video_list, base_dir="/content/drive/MyDrive/EEG2Video_data/processed", debug=False):
        self.base_dir = base_dir
        self.eeg_files = [l.strip() for l in open(eeg_list).readlines()]
        self.video_files = [l.strip() for l in open(video_list).readlines()]
        assert len(self.eeg_files) == len(self.video_files)

        eeg_all = []
        for rel_path in self.eeg_files:
            abs_path = os.path.join(self.base_dir, "EEG_windows", rel_path)
            eeg = np.load(abs_path)  # [7,62,100]
            eeg_all.append(eeg.reshape(-1, 100))
        eeg_all = np.vstack(eeg_all)
        self.scaler = StandardScaler().fit(eeg_all)

        vid_all = []
        for rel_path in self.video_files:
            abs_path = os.path.join(self.base_dir, "Video_latents", rel_path)
            vid = np.load(abs_path)  # [F,4,36,64]
            vid_all.append(vid.reshape(-1, 4*36*64))
        vid_all = np.vstack(vid_all)
        self.latent_scaler = StandardScaler().fit(vid_all)

        self.debug = debug

    def __len__(self):
        return len(self.eeg_files)

    def __getitem__(self, idx):
        eeg_path = os.path.join(self.base_dir, "EEG_windows", self.eeg_files[idx])
        vid_path = os.path.join(self.base_dir, "Video_latents", self.video_files[idx])

        eeg = np.load(eeg_path)      # [7,62,100]
        video = np.load(vid_path)    # [F,4,36,64]

        b, c, t = eeg.shape
        eeg = eeg.reshape(-1, t)
        eeg = self.scaler.transform(eeg)
        eeg = eeg.reshape(b, c, t)

        f, ch, h, w = video.shape
        video = video.reshape(-1, ch*h*w)
        video = self.latent_scaler.transform(video)
        video = video.reshape(f, ch, h, w)

        eeg = torch.tensor(eeg, dtype=torch.float32)
        video = torch.tensor(video, dtype=torch.float32)

        if self.debug and idx < 2:
            print(f"[DEBUG ALIGNMENT] eeg={self.eeg_files[idx]} | video={self.video_files[idx]}")

        return eeg, video

# -------------------------
# Training Loop
# -------------------------
if __name__ == "__main__":
    drive_root = "/content/drive/MyDrive/EEG2Video_data/processed"

    eeg_train_list = os.path.join(drive_root, "EEG_windows/train_list.txt")
    vid_train_list = os.path.join(drive_root, "Video_latents/train_list_dup.txt")

    dataset = EEGVideoDataset(eeg_train_list, vid_train_list, debug=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    # assume all videos have same length F (e.g., 24)
    sample_vid = np.load(os.path.join(drive_root, "Video_latents", dataset.video_files[0]))
    pred_frames = sample_vid.shape[0]

    model = MyTransformer(d_model=512, pred_frames=pred_frames).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200*len(dataloader))
    criterion = nn.MSELoss()

    for epoch in range(20):
        model.train()
        epoch_loss = 0
        for eeg, video in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            eeg, video = eeg.cuda(), video.cuda()

            # create start token (zero frame)
            b, f, c, h, w = video.shape
            zero_frame = torch.zeros((b,1,c,h,w), device=video.device)

            optimizer.zero_grad()
            out = model(eeg, zero_frame)
            loss = criterion(out, video)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss={epoch_loss/len(dataloader):.6f}")

        with torch.no_grad():
            eeg, video = next(iter(dataloader))
            eeg, video = eeg.cuda(), video.cuda()
            zero_frame = torch.zeros((video.size(0),1,video.size(2),video.size(3),video.size(4)), device=video.device)
            pred = model(eeg, zero_frame)

            print("[DEBUG] GT latents: mean {:.4f}, std {:.4f}".format(video.mean().item(), video.std().item()))
            print("[DEBUG] Predicted : mean {:.4f}, std {:.4f}".format(pred.mean().item(), pred.std().item()))

    save_path = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_checkpoint.pt"
    torch.save({"state_dict": model.state_dict()}, save_path)
    print("Model saved to:", save_path)
