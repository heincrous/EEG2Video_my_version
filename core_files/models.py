import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

# ================================
# Helper to dynamically compute flatten size
# ================================
def _get_flattened_size(net, C, T):
    with torch.no_grad():
        x = torch.zeros(1, 1, C, T)
        y = net(x)
        return y.view(1, -1).shape[1]

# ================================
# ShallowNet
# ================================
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
        flat_dim = _get_flattened_size(self.net, C, T)
        self.out = nn.Linear(flat_dim, out_dim)

    def forward(self, x):
        B, W, C, T = x.shape
        x = x.view(B*W, 1, C, T)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x.view(B, W, -1)

# ================================
# DeepNet
# ================================
class deepnet(nn.Module):
    def __init__(self, out_dim, C, T):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10)),
            nn.Conv2d(25, 25, (C, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(25, 50, (1, 10)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(50, 100, (1, 10)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(100, 200, (1, 10)),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5),
        )
        flat_dim = _get_flattened_size(self.net, C, T)
        self.out = nn.Linear(flat_dim, out_dim)

    def forward(self, x):
        B, W, C, T = x.shape
        x = x.view(B*W, 1, C, T)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x.view(B, W, -1)

# ================================
# EEGNet
# ================================
class eegnet(nn.Module):
    def __init__(self, out_dim, C, T):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (C, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout2d(0.5),
        )
        flat_dim = _get_flattened_size(self.net, C, T)
        self.out = nn.Linear(flat_dim, out_dim)

    def forward(self, x):
        B, W, C, T = x.shape
        x = x.view(B*W, 1, C, T)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x.view(B, W, -1)

# ================================
# TsConv
# ================================
class tsconv(nn.Module):
    def __init__(self, out_dim, C, T):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (C, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        flat_dim = _get_flattened_size(self.net, C, T)
        self.out = nn.Linear(flat_dim, out_dim)

    def forward(self, x):
        B, W, C, T = x.shape
        x = x.view(B*W, 1, C, T)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x.view(B, W, -1)

# ================================
# MLPNet (for DE/PSD features)
# ================================
class mlpnet(nn.Module):
    def __init__(self, out_dim, input_dim=310):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# ================================
# Conformer (authors' version, ViT-style)
# ================================
from einops.layers.torch import Rearrange, Reduce

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (62, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # slice into patches along time dim
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1)),  # project channel dim → emb_size
            Rearrange('b e 1 w -> b w e'),    # flatten patches into tokens
        )

    def forward(self, x):
        B, W, C, T = x.shape
        x = x.view(B*W, 1, C, T)      # merge windows
        x = self.shallownet(x)
        x = self.projection(x)        # (B*W, n_patches, emb_size)
        return x.view(B, -1, x.size(-1))  # (B, n_patches*W, emb_size)

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

    def forward(self, x, mask=None):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys    = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy  = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            energy.masked_fill_(~mask, float("-inf"))
        scaling = self.emb_size ** 0.5
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=10, drop_p=0.5,
                 forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p),
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion,
                                 drop_p=forward_drop_p),
                nn.Dropout(drop_p),
            )),
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[
            TransformerEncoderBlock(emb_size) for _ in range(depth)
        ])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, out_dim):
        super().__init__(
            Reduce('b n e -> b e', 'mean'),  # mean pool tokens
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, out_dim),
        )

class conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=3, out_dim=77*768):
        super().__init__(
            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, out_dim),
        )

# ================================
# GLFNet (for DE/PSD features)
# ================================
class glfnet(nn.Module):
    def __init__(self, out_dim, emb_dim, C, T):
        super().__init__()
        # global branch on all channels
        self.globalnet = shallownet(emb_dim, C, T)
        # occipital branch (last 12 channels)
        self.occipital_index = list(range(50, 62))
        self.occipital_localnet = shallownet(emb_dim, 12, T)
        # combine
        self.out = nn.Linear(emb_dim * 2, out_dim)

    def forward(self, x):  # x: (B, W, C, T)
        B, W, C, T = x.shape
        x = x.view(B*W, 1, C, T)
        global_feat = self.globalnet.net(x)
        global_feat = global_feat.view(global_feat.size(0), -1)
        global_feat = self.globalnet.out(global_feat)

        occipital_x = x[:, :, self.occipital_index, :]
        local_feat = self.occipital_localnet.net(occipital_x)
        local_feat = local_feat.view(local_feat.size(0), -1)
        local_feat = self.occipital_localnet.out(local_feat)

        out = self.out(torch.cat((global_feat, local_feat), dim=1))
        return out.view(B, W, -1)

# ================================
# GLFNet-MLP (for DE/PSD features)
# ================================
class glfnet_mlp(nn.Module):
    def __init__(self, out_dim, emb_dim, input_dim):
        super().__init__()
        # global branch
        self.globalnet = mlpnet(emb_dim, input_dim)
        # occipital branch (12 channels × 5 bands = 60 input)
        self.occipital_index = list(range(50, 62))
        self.occipital_localnet = mlpnet(emb_dim, 12*5)
        # combine
        self.out = nn.Linear(emb_dim * 2, out_dim)

    def forward(self, x):  # x: (B, C, 5)
        global_feat = self.globalnet(x)
        occipital_x = x[:, self.occipital_index, :]
        local_feat = self.occipital_localnet(occipital_x)
        out = self.out(torch.cat((global_feat, local_feat), dim=1))
        return out

