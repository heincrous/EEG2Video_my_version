import math
import random
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from copy import deepcopy

# Import GT_LABEL from gt_label
from utils.gt_label import GT_LABEL

max_length = 16


class MyEEGNet_embedding(nn.Module):
    def __init__(self, d_model=128, C=62, T=200, F1=16, D=4, F2=16, cross_subject=False):
        super(MyEEGNet_embedding, self).__init__()
        if cross_subject:
            self.drop_out = 0.25
        else:
            self.drop_out = 0.5

        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, 64), bias=False),
            nn.BatchNorm2d(F1)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=F1 * D, kernel_size=(C, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.drop_out)
        )

        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(in_channels=F1 * D, out_channels=F1 * D, kernel_size=(1, 16), groups=F1 * D, bias=False),
            nn.Conv2d(in_channels=F1 * D, out_channels=F2, kernel_size=(1, 1), bias=False),
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


class myTransformer(nn.Module):
    def __init__(self, d_model=512):
        super(myTransformer, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=d_model)
        self.img_embedding = nn.Linear(4 * 36 * 64, d_model)
        self.eeg_embedding = MyEEGNet_embedding(d_model=d_model, C=62, T=100, F1=16, D=4, F2=16, cross_subject=False)

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
        src = self.eeg_embedding(src.reshape(src.shape[0] * src.shape[1], 1, 62, 100)).reshape(src.shape[0], 7, -1)
        tgt = tgt.reshape(tgt.shape[0], tgt.shape[1], tgt.shape[2] * tgt.shape[3] * tgt.shape[4])
        tgt = self.img_embedding(tgt)

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-2]).to(tgt.device)

        encoder_output = self.transformer_encoder(src)

        new_tgt = torch.zeros((tgt.shape[0], 1, tgt.shape[2])).cuda()
        for i in range(6):
            decoder_output = self.transformer_decoder(new_tgt, encoder_output, tgt_mask=tgt_mask[:i+1, :i+1])
            new_tgt = torch.cat((new_tgt, decoder_output[:, -1:, :]), dim=1)

        encoder_output = torch.mean(encoder_output, dim=1)

        return self.txtpredictor(encoder_output), self.predictor(new_tgt).reshape(new_tgt.shape[0], new_tgt.shape[1], 4, 36, 64)


def evaluate_accuracy_auto(net, data_iter, device):
    loss = nn.MSELoss()
    total_loss = 0
    net.eval()
    if not device:
        device = next(iter(net.parameters())).device
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        y_auto = torch.zeros(y.shape[0], 1, 768).to(device)
        y_auto_hat = net(X, y_auto)
        for i in range(9):
            y_auto = torch.cat((y_auto, y_auto_hat[:, -1, :].reshape(y.shape[0], 1, 768)))
            y_auto_hat = net(X, y_auto)

        y_auto = torch.cat((y_auto, y_auto_hat[:, -1, :].reshape(y.shape[0], 1, 768)))
        y_auto = y_auto[:, 1:, :]
        test_loss = loss(y_auto, y)
        total_loss += test_loss.item()
    return total_loss / len(data_iter)


class Dataset():
    def __init__(self, eeg, video):
        self.eeg = eeg
        self.video = video
        self.len = eeg.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.eeg[item], self.video[item]


def loss(true, pred):
    return nn.MSELoss()(true, pred)


def normalizetion(data):
    mean = torch.mean(data, dim=(0, 2, 3, 4), dtype=torch.float64)
    std = torch.std(data, dim=(0, 2, 3, 4))

    normalized_data = (data - mean.reshape(1, 4, 1, 1, 1)) / std.reshape(1, 4, 1, 1, 1)

    return normalized_data


if __name__ == "__main__":
    # Load data
    eegdata = np.load('../data/SEED-DV/Segmented_Rawf_200Hz_2s/sub1.npy')
    latent_data = np.load('1200_latent.npy')
    latent_data = torch.from_numpy(latent_data)
    test_latent = torch.load('40classes_latents.pt')

    # Select data indices from GT_LABEL (smaller dataset for testing)
    chosed_index = []
    for i in range(7):
        index = [list(GT_LABEL[i]).index(element) for element in range(1, 6)]  # Test on 5 classes
        chosed_index.append(index)

    new_eeg = np.zeros((7, 40, 5, 62, 400))
    for i in range(7):
        new_eeg[i] = eegdata[i][chosed_index[i], :, :, :]

    latent_data = rearrange(latent_data, "(g p d) c f h w -> g p d c f h w", g=6, p=40, d=5)
    new_latent = np.zeros((6, 40, 5, 4, 6, 36, 64))
    for i in range(6):
        new_latent[i] = latent_data[i][chosed_index[i], :, :, :, :, :]

    new_latent = rearrange(new_latent, "g p d c f h w -> (g p d) c f h w")
    new_eeg = torch.from_numpy(new_eeg)
    new_latent = torch.from_numpy(new_latent)

    window_size = 100
    overlap = 50
    EEG = []
    for i in range(0, new_eeg.shape[-1] - window_size + 1, window_size - overlap):
        EEG.append(new_eeg[..., i:i + window_size])
    EEG = torch.stack(EEG, dim=-1)

    test_eeg = EEG[6, :]
    EEG = EEG[0:6, :]
    EEG = torch.reshape(EEG, (EEG.shape[0] * EEG.shape[1] * EEG.shape[2], EEG.shape[3], EEG.shape[4], EEG.shape[5]))
    test_eeg = torch.reshape(test_eeg, (test_eeg.shape[0] * test_eeg.shape[1], test_eeg.shape[2], test_eeg.shape[3], test_eeg.shape[4]))

    # Normalize data
    normalize = StandardScaler()
    EEG = normalize.fit_transform(EEG)
    test_eeg = normalize.transform(test_eeg)

    # Dataset and dataloader
    dataset = Dataset(EEG, new_latent)
    train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # Reduced batch size for testing

    # Model setup
    model = myTransformer()
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5 * len(train_dataloader))  # Reduced epochs for testing

    # Training loop
    for epoch in tqdm(range(5)):  # Reduced epochs to 5 for testing
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(train_dataloader):
            eeg, video = batch
            eeg = eeg.float().cuda()

            b, _, c, w, h = video.shape
            padded_video = torch.zeros((b, 1, c, w, h))
            full_video = torch.cat((padded_video, video), dim=1).float().cuda()
            optimizer.zero_grad()

            txt_label, out = model(eeg, full_video)

            video = video.float().cuda()
            l = loss(video, out[:, :-1, :])
            l.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += l.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss}")
