# ==========================================
# Different EEG encoders for comparison
#
# SA / GA:
#   shallownet, deepnet, eegnet, conformer, tsconv
# ==========================================

# === Standard libraries ===
import math

# === Third-party libraries ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange


# ==========================================
# ShallowNet
# ==========================================
class shallownet(nn.Module):
    def __init__(self, out_dim=512, C=62, T=200, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, (1, 25)),
            nn.Conv2d(32, 32, (C, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            feat_dim = self.net(torch.zeros(1,1,C,T)).view(1,-1).size(1)
        self.out = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return self.out(x.view(x.size(0), -1))


# ==========================================
# DeepNet
# ==========================================
class deepnet(nn.Module):
    def __init__(self, out_dim=512, C=62, T=200, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10)),
            nn.Conv2d(25, 25, (C, 1)),
            nn.BatchNorm2d(25), nn.ELU(),
            nn.MaxPool2d((1, 2)), nn.Dropout(dropout),

            nn.Conv2d(25, 50, (1, 10)), nn.BatchNorm2d(50), nn.ELU(),
            nn.MaxPool2d((1, 2)), nn.Dropout(dropout),

            nn.Conv2d(50, 100, (1, 10)), nn.BatchNorm2d(100), nn.ELU(),
            nn.MaxPool2d((1, 2)), nn.Dropout(dropout),

            nn.Conv2d(100, 200, (1, 10)), nn.BatchNorm2d(200), nn.ELU(),
            nn.MaxPool2d((1, 2)), nn.Dropout(dropout),
        )
        with torch.no_grad():
            feat_dim = self.net(torch.zeros(1,1,C,T)).view(1,-1).size(1)
        self.out = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return self.out(x.view(x.size(0), -1))


# ==========================================
# EEGNet
# ==========================================
class eegnet(nn.Module):
    def __init__(self, out_dim=512, C=62, T=200, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64)), nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (C, 1)), nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 2)), nn.Dropout(dropout),

            nn.Conv2d(16, 16, (1, 16)), nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 2)), nn.Dropout2d(dropout),
        )
        with torch.no_grad():
            feat_dim = self.net(torch.zeros(1,1,C,T)).view(1,-1).size(1)
        self.out = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return self.out(x.view(x.size(0), -1))


# ==========================================
# TSConv
# ==========================================
class tsconv(nn.Module):
    def __init__(self, out_dim=512, C=62, T=200, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40), nn.ELU(),
            nn.Conv2d(40, 40, (C, 1)),
            nn.BatchNorm2d(40), nn.ELU(),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            feat_dim = self.net(torch.zeros(1,1,C,T)).view(1,-1).size(1)
        self.out = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return self.out(x.view(x.size(0), -1))


# ==========================================
# Transformer-based Conformer and Utilities
# ==========================================
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, C=62):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25)),
            nn.Conv2d(40, 40, (C, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )
        self.proj = nn.Conv2d(40, emb_size, (1,1))

    def forward(self, x):
        x = self.conv(x)
        x = self.proj(x)
        return rearrange(x, "b e h w -> b (h w) e")


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor) -> Tensor:
        q = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(self.keys(x),    "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(self.values(x),  "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', q, k)
        att = F.softmax(energy / math.sqrt(self.emb_size), dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn): super().__init__(); self.fn = fn
    def forward(self, x): return self.fn(x) + x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion*emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion*emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=8, drop_p=0.5, exp=4):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, exp, drop_p),
                nn.Dropout(drop_p)
            ))
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[
            TransformerEncoderBlock(emb_size) for _ in range(depth)
        ])


class ClassificationHead(nn.Module):
    def __init__(self, emb_size, out_dim, dropout=0.5):
        super().__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.fc   = nn.Sequential(
            nn.Linear(emb_size, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x.mean(dim=1)   # mean pool tokens
        x = self.norm(x)
        return self.fc(x)


class conformer(nn.Module):
    def __init__(self, out_dim=512, C=62, T=200, emb_size=40, depth=3):
        super().__init__()
        self.patch = PatchEmbedding(emb_size, C)
        self.encoder = TransformerEncoder(depth, emb_size)
        with torch.no_grad():
            feat_dim = self.patch(torch.zeros(1,1,C,T))
            feat_dim = self.encoder(feat_dim).view(1,-1).size(1)
        self.fc = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        x = self.patch(x)
        x = self.encoder(x)
        return self.fc(x.view(x.size(0), -1))


# ==========================================
# GLFNet and MLP variants
# ==========================================
class glfnet(nn.Module):
    def __init__(self, out_dim, emb_dim, C, T, dropout=0.5):
        super().__init__()
        self.globalnet = shallownet(emb_dim, C, T, dropout=dropout)
        self.occipital_index = list(range(50, 62))
        self.occipital_localnet = shallownet(emb_dim, 12, T, dropout=dropout)
        self.out = nn.Sequential(
            nn.Linear(emb_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        global_feature = self.globalnet(x)
        occipital_x = x[:, :, self.occipital_index, :]
        occipital_feature = self.occipital_localnet(occipital_x)
        return self.out(torch.cat((global_feature, occipital_feature), 1))


class mlpnet(nn.Module):
    def __init__(self, out_dim, input_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class glfnet_mlp(nn.Module):
    def __init__(self, out_dim, emb_dim, input_dim, dropout=0.5):
        super().__init__()
        self.globalnet = mlpnet(emb_dim, input_dim, dropout=dropout)
        self.occipital_index = list(range(50, 62))
        self.occipital_localnet = mlpnet(emb_dim, 12 * (input_dim // 62), dropout=dropout)
        self.out = nn.Sequential(
            nn.Linear(emb_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        global_feature = self.globalnet(x)
        occipital_x = x[:, self.occipital_index, :]
        occipital_feature = self.occipital_localnet(occipital_x)
        return self.out(torch.cat((global_feature, occipital_feature), 1))
