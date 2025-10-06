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
max_length = 16


class MyEEGNet_embedding(nn.Module):
    def __init__(self, d_model=128, C=62, T=200, F1=16, D=4, F2=16, cross_subject=False):
        super(MyEEGNet_embedding, self).__init__()
        if (cross_subject == True):
            self.drop_out = 0.25
        else:
            self.drop_out = 0.5

        self.block_1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=1,  # input shape (1, C, T)
                out_channels=F1,  # num_filters
                kernel_size=(1, 64),  # filter size
                bias=False
            ),  # output shape (F1, C, T)
            nn.BatchNorm2d(F1)  # output shape (F1, C, T)
        )

        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=F1,  # input shape (F1, C, T)
                out_channels=F1 * D,  # num_filters
                kernel_size=(C, 1),  # filter size
                groups=F1,
                bias=False
            ),  # output shape (F1 * D, 1, T)
            nn.BatchNorm2d(F1 * D),  # output shape (F1 * D, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (F1 * D, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (F1 * D, 1, T//4)
        )

        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            # The Separable Convolution can be diveded into two steps
            # The first conv dosen't change the channels, only layers to layers respectively
            nn.Conv2d(
                in_channels=F1 * D,  # input shape (F1 * D, 1, T//4)
                out_channels=F1 * D,
                kernel_size=(1, 16),  # filter size
                groups=F1 * D,
                bias=False
            ),  # output shape (F1 * D, 1, T//4)
            # The second conv changes the channels, use 1x1 conv to combine channels' information
            nn.Conv2d(
                in_channels=F1 * D,  # input shape (F1 * D, 1, T//4)
                out_channels=F2,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False
            ),  # output shape (F2, 1, T//4)
            nn.BatchNorm2d(F2),  # output shape (F2, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (F2, 1, T//32)
            nn.Dropout(self.drop_out)
        )

        self.embedding = nn.Linear(48, d_model)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)

        # print(x.shape)
        x = x.view(x.shape[0], -1)
        x = self.embedding(x)
        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class myTransformer(nn.Module):

    def __init__(self, d_model=512):
        super(myTransformer, self).__init__()

        # 定义词向量，词典数为10。我们不预测两位小数。
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

        # 定义位置编码器
        self.positional_encoding = PositionalEncoding(d_model, dropout=0)

        self.txtpredictor = nn.Linear(512, 13)

        # 定义最后的线性层，这里并没有用Softmax，因为没必要。
        # 因为后面的CrossEntropyLoss中自带了
        self.predictor = nn.Linear(512, 4 * 36 * 64)

    def forward(self, src, tgt):
        # 对src和tgt进行编码
        # x = torch.rand(size=(32, 10, 62, 200))
        src = self.eeg_embedding(src.reshape(src.shape[0] * src.shape[1], 1, 62, 100)).reshape(src.shape[0], 7, -1)
        # print("src.shape = ", src.shape)

        tgt = tgt.reshape(tgt.shape[0], tgt.shape[1], tgt.shape[2] * tgt.shape[3] * tgt.shape[4])
        tgt = self.img_embedding(tgt)
        # print("tgt.shape = ", tgt.shape)
        # 给src和tgt的token增加位置信息
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # 生成mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-2]).to(tgt.device)
        # print("mask:", tgt_mask)
        # src_key_padding_mask = myTransformer.get_key_padding_mask(src)
        # tgt_key_padding_mask = myTransformer.get_key_padding_mask(tgt)

        # 使用 Transformer Encoder 对输入序列进行编码
        encoder_output = self.transformer_encoder(src)
        #print("en.shape = ", encoder_output.shape)

        # 使用 Transformer Decoder 对目标序列进行解码
        #print(tgt.shape)
        new_tgt = torch.zeros((tgt.shape[0], 1, tgt.shape[2])).cuda()
        for i in range(6):
            decoder_output = self.transformer_decoder(new_tgt, encoder_output, tgt_mask=tgt_mask[:i+1, :i+1])

            # print(new_tgt.shape)
            new_tgt = torch.cat((new_tgt, decoder_output[:, -1:, :]), dim=1)

        #decoder_output = self.transformer_decoder(tgt, encoder_output, tgt_mask=tgt_mask)

        #print("new_tgt.shape = ", new_tgt.shape)

        encoder_output = torch.mean(encoder_output, dim=1)
        # print(encoder_output.shape)

        return self.txtpredictor(encoder_output), self.predictor(new_tgt).reshape(new_tgt.shape[0],
                                                                                         new_tgt.shape[1], 4, 36,
                                                                                         64)


# criteria = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

def evaluate_accuracy_auto(net, data_iter, device):
    loss = nn.MSELoss()
    total_loss = 0
    if isinstance(net, nn.Module):
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
        print(y_auto.shape)
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


    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.eeg[item], self.video[item]


def loss(true, pred):
    l = nn.MSELoss()
    return l(true, pred)


def normalizetion(data):
    mean = torch.mean(data, dim=(0, 2, 3, 4), dtype=torch.float64)
    std = torch.std(data, dim=(0, 2, 3, 4))

    print("mean:", mean)
    print("std:", std)
    normalized_data = (data - mean.reshape(1, 4, 1, 1, 1)) / std.reshape(1, 4, 1, 1, 1)

    return normalized_data

if __name__ == "__main__":
    eegdata = np.load('/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments/sub1.npy')  # shape: (7, 40, 5, 62, 400)
    latent_data = np.load('/content/drive/MyDrive/EEG2Video_data/processed/Video_latents.npy')  # shape: (7, 40, 5, 6, 4, 36, 64)

    new_eeg = eegdata  # already aligned to GT label

    train_latent = latent_data[:6]  # shape: (6, 40, 5, 6, 4, 36, 64)
    test_latent  = latent_data[6]   # shape: (40, 5, 6, 4, 36, 64)

    train_latent = rearrange(train_latent, "g p d s c h w -> (g p d) s c h w")
    test_latent  = rearrange(test_latent, "p d s c h w -> (p d) s c h w")

    new_eeg = torch.from_numpy(new_eeg)
    train_latent = torch.from_numpy(train_latent)

    # ===================== EEG windowing =====================
    window_size = 100
    overlap = 50
    EEG = []
    for i in range(0, new_eeg.shape[-1] - window_size + 1, window_size - overlap):
        EEG.append(new_eeg[..., i:i + window_size])
    EEG = torch.stack(EEG, dim=-1)  # shape: (7, 40, 5, 62, 100, N)

    test_eeg = EEG[6, :]  # block 6
    EEG = EEG[:6, :]      # blocks 0-5

    EEG = torch.reshape(EEG, (EEG.shape[0] * EEG.shape[1] * EEG.shape[2], EEG.shape[3], EEG.shape[4], EEG.shape[5]))
    test_eeg = torch.reshape(test_eeg, (test_eeg.shape[0] * test_eeg.shape[1], test_eeg.shape[2], test_eeg.shape[3], test_eeg.shape[4]))

    b, c, l, f = EEG.shape
    EEG = EEG.flatten(1)
    test_eeg = test_eeg.flatten(1)

    normalize = StandardScaler()
    normalize.fit(EEG)
    EEG = normalize.transform(EEG)
    test_eeg = normalize.transform(test_eeg)

    EEG = rearrange(EEG, 'b (c l f) -> b c l f', c=c, l=l, f=f)
    test_eeg = rearrange(test_eeg, 'b (c l f) -> b c l f', c=c, l=l, f=f)

    EEG = rearrange(EEG, "b c l f -> b f c l")
    test_eeg = rearrange(test_eeg, "b c l f -> b f c l")

    EEG = torch.from_numpy(EEG)
    test_eeg = torch.from_numpy(test_eeg)

    train_latent = rearrange(train_latent, "b s c h w -> b c h w")
    test_latent = rearrange(test_latent, "b s c h w -> b c h w")

    print("EEG:", EEG.shape)
    print("test_eeg:", test_eeg.shape)
    print("latent:", train_latent.shape)
    print("test_latent:", test_latent.shape)

    dataset = Dataset(EEG, train_latent)
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = myTransformer().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200 * len(train_dataloader))

    for epoch in tqdm(range(200)):
        model.train()
        epoch_loss = 0
        for eeg, video in train_dataloader:
            eeg = eeg.float().cuda()
            b, _, c, w, h = video.shape
            padded_video = torch.zeros((b, 1, c, w, h)).float().cuda()
            full_video = torch.cat((padded_video, video), dim=1)

            optimizer.zero_grad()
            txt_label, out = model(eeg, full_video)
            loss_val = loss(video.cuda(), out[:, :-1, :])
            loss_val.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss_val.item()
        print(epoch_loss)

    # Inference
    model.eval()
    test_latent = test_latent.float().cuda()
    test_eeg = test_eeg.float().cuda()
    b, _, c, w, h = test_latent.shape
    padded_video = torch.zeros((b, 1, c, w, h)).float().cuda()
    full_video = torch.cat((padded_video, test_latent), dim=1).float().cuda()
    txt_label, out = model(test_eeg, full_video)
    latent_out = out[:, :-1, :].cpu().detach().numpy()
    np.save('/content/drive/MyDrive/EEG2Video_outputs/latent_out_block7_40_classes.npy', latent_out)

    model_dict = model.state_dict()
    torch.save({'state_dict': model_dict}, '/content/drive/MyDrive/EEG2Video_checkpoints/seq2seqmodel.pt')



