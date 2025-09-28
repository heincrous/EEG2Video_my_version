"""
Different EEG encoders for comparison
Includes:
- shallownet
- deepnet
- eegnet
- tsconv
- conformer
- glfnet (global + local features)
- mlpnet, glfnet_mlp (for DE/PSD tabular features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce


# ==========================================
# CNN-based encoders for EEG segments
# ==========================================
class shallownet(nn.Module):
    def __init__(self, out_dim, C, T):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25)),
            nn.Conv2d(40, 40, (C, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.Dropout(0.5),
        )
        # infer flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            flat_dim = self.net(dummy).view(1, -1).size(1)
        self.out = nn.Linear(flat_dim, out_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.out(x)


class deepnet(nn.Module):
    def __init__(self, out_dim, C, T):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10)),
            nn.Conv2d(25, 25, (C, 1)),
            nn.BatchNorm2d(25), nn.ELU(),
            nn.MaxPool2d((1, 2)), nn.Dropout(0.5),

            nn.Conv2d(25, 50, (1, 10)),
            nn.BatchNorm2d(50), nn.ELU(),
            nn.MaxPool2d((1, 2)), nn.Dropout(0.5),

            nn.Conv2d(50, 100, (1, 10)),
            nn.BatchNorm2d(100), nn.ELU(),
            nn.MaxPool2d((1, 2)), nn.Dropout(0.5),

            nn.Conv2d(100, 200, (1, 10)),
            nn.BatchNorm2d(200), nn.ELU(),
            nn.MaxPool2d((1, 2)), nn.Dropout(0.5),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            flat_dim = self.net(dummy).view(1, -1).size(1)
        self.out = nn.Linear(flat_dim, out_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.out(x)


class eegnet(nn.Module):
    def __init__(self, out_dim, C, T):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (C, 1)),
            nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 2)), nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16)),
            nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 2)), nn.Dropout2d(0.5),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            flat_dim = self.net(dummy).view(1, -1).size(1)
        self.out = nn.Linear(flat_dim, out_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.out(x)


class tsconv(nn.Module):
    def __init__(self, out_dim, C, T):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40), nn.ELU(),
            nn.Conv2d(40, 40, (C, 1)),
            nn.BatchNorm2d(40), nn.ELU(),
            nn.Dropout(0.5),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            flat_dim = self.net(dummy).view(1, -1).size(1)
        self.out = nn.Linear(flat_dim, out_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.out(x)


# ==========================================
# Transformer-based encoder (conformer)
# ==========================================
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25)),
            nn.Conv2d(40, 40, (62, 1)),
            nn.BatchNorm2d(40), nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        return self.projection(x)


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
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys    = rearrange(self.keys(x),    "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.values(x),  "b n (h d) -> b h n d", h=self.num_heads)
        energy  = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        scaling = self.emb_size ** 0.5
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn): super().__init__(); self.fn = fn
    def forward(self, x, **kwargs): return x + self.fn(x, **kwargs)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(), nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=8, drop_p=0.5,
                 forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, forward_expansion, forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[
            TransformerEncoderBlock(emb_size) for _ in range(depth)
        ])


class ClassificationHead(nn.Module):
    def __init__(self, emb_size, out_dim):
        super().__init__()
        self.pool = Reduce('b n e -> b e', reduction='mean')
        self.norm = nn.LayerNorm(emb_size)
        self.fc   = nn.Linear(emb_size, out_dim)
    def forward(self, x):
        return self.fc(self.norm(self.pool(x)))


class conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=3, out_dim=128):
        super().__init__(
            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, out_dim)
        )


# ==========================================
# GLF-net and MLPs for DE/PSD
# ==========================================
class mlpnet(nn.Module):
    def __init__(self, out_dim, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, x): return self.net(x)


class glfnet(nn.Module):
    def __init__(self, out_dim, emb_dim, C, T):
        super().__init__()
        self.globalnet   = shallownet(emb_dim, C, T)
        self.local_index = list(range(50, 62))
        self.localnet    = shallownet(emb_dim, 12, T)
        self.out         = nn.Linear(emb_dim*2, out_dim)

    def forward(self, x):
        global_feature = self.globalnet(x)
        local_feature  = self.localnet(x[:, :, self.local_index, :])
        return self.out(torch.cat((global_feature, local_feature), 1))


class glfnet_mlp(nn.Module):
    def __init__(self, out_dim, emb_dim, input_dim):
        super().__init__()
        self.globalnet   = mlpnet(emb_dim, input_dim)
        self.local_index = list(range(50, 62))
        self.localnet    = mlpnet(emb_dim, 12*5)
        self.out         = nn.Linear(emb_dim*2, out_dim)

    def forward(self, x):
        global_feature = self.globalnet(x)
        local_feature  = self.localnet(x[:, self.local_index, :])
        return self.out(torch.cat((global_feature, local_feature), 1))


# ==========================================
# Quick test
# ==========================================
if __name__ == "__main__":
    model = glfnet_mlp(out_dim=3, emb_dim=64, input_dim=310)
    x = torch.rand(size=(1, 62, 5))
    y = model(x)
    print("Input:", x.shape, "Output:", y.shape)
