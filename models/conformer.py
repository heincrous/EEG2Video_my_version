# ==========================================
# Config block (tunable but defaults unchanged)
# ==========================================
CONFIG = {
    "dropout": 0.5,              # dropout probability
    "layer_width": 40,           # embedding dimension
    "num_heads": 10,             # number of attention heads
    "kernel_size": (1, 25),      # temporal patch kernel
    "pool_size": (1, 75),        # pooling window
    "pool_stride": (1, 15),      # pooling stride
    "activation": "ELU",         # "ELU", "GELU", "SiLU", etc.
    "normalization": "BatchNorm",# "BatchNorm", "LayerNorm", "GroupNorm"
}


# ==========================================
# Model definition
# ==========================================
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=CONFIG["layer_width"], cfg=CONFIG):
        super().__init__()
        w = emb_size
        p = cfg["dropout"]

        # activation
        act_fn = getattr(nn, cfg["activation"])() if hasattr(nn, cfg["activation"]) else nn.ELU()

        # normalization
        if cfg["normalization"] == "BatchNorm":
            norm_layer = nn.BatchNorm2d(w)
        elif cfg["normalization"] == "LayerNorm":
            norm_layer = nn.LayerNorm([w, 1, 1])
        elif cfg["normalization"] == "GroupNorm":
            norm_layer = nn.GroupNorm(4, w)
        else:
            norm_layer = nn.BatchNorm2d(w)

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, w, cfg["kernel_size"]),
            nn.Conv2d(w, w, (62, 1)),
            norm_layer,
            act_fn,
            nn.AvgPool2d(cfg["pool_size"], cfg["pool_stride"]),
            nn.Dropout(p),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(w, emb_size, (1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x):
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, cfg=CONFIG):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = cfg["num_heads"]
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(cfg["dropout"])
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        scaling = self.emb_size ** 0.5
        att = torch.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, cfg=CONFIG, expansion=4):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, cfg=CONFIG, expansion=4):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, cfg),
                nn.Dropout(cfg["dropout"])
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, cfg, expansion),
                nn.Dropout(cfg["dropout"])
            ))
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size, cfg=CONFIG):
        super().__init__(*[
            TransformerEncoderBlock(emb_size, cfg) for _ in range(depth)
        ])


class ClassificationHead(nn.Module):
    def __init__(self, emb_size, out_dim):
        super().__init__()
        self.fc = None  # will be defined dynamically
        self.out_dim = out_dim  # store target output dimension

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        if self.fc is None:
            self.fc = nn.Linear(x.size(1), self.out_dim, device=x.device)
        return self.fc(x)


class conformer(nn.Sequential):
    def __init__(self, emb_size=CONFIG["layer_width"], depth=3, out_dim=4, cfg=CONFIG):
        super().__init__(
            PatchEmbedding(emb_size, cfg),
            TransformerEncoder(depth, emb_size, cfg),
            ClassificationHead(emb_size, out_dim)
        )
