# ==========================================
# Config block (tunable but defaults unchanged)
# ==========================================
CONFIG = {
    "dropout": 0.5,              # dropout probability
    "layer_width": 40,           # number of convolutional filters
    "kernel_size": (1, 25),      # temporal kernel
    "pool_size": (1, 51),        # pooling window
    "pool_stride": (1, 5),       # pooling stride
    "activation": "ELU",         # "ELU", "GELU", "SiLU", etc.
    "normalisation": "BatchNorm",# "BatchNorm", "LayerNorm", "GroupNorm"
}


# ==========================================
# Model definition
# ==========================================
import torch
import torch.nn as nn


class tsconv(nn.Module):
    def __init__(self, out_dim, C, T, cfg=CONFIG):
        super(tsconv, self).__init__()

        w = cfg["layer_width"]
        p = cfg["dropout"]

        # Activation
        act_fn = getattr(nn, cfg["activation"])() if hasattr(nn, cfg["activation"]) else nn.ELU()

        # Normalisation
        if cfg["normalisation"] == "BatchNorm":
            norm_layer = nn.BatchNorm2d(w)
        elif cfg["normalisation"] == "LayerNorm":
            norm_layer = nn.LayerNorm([w, 1, 1])
        elif cfg["normalisation"] == "GroupNorm":
            norm_layer = nn.GroupNorm(4, w)
        else:
            norm_layer = nn.BatchNorm2d(w)

        # Convolutional stack
        self.net = nn.Sequential(
            nn.Conv2d(1, w, cfg["kernel_size"]),
            nn.AvgPool2d(cfg["pool_size"], cfg["pool_stride"]),
            norm_layer,
            act_fn,
            nn.Conv2d(w, w, (C, 1)),
            norm_layer,
            act_fn,
            nn.Dropout(p),
        )

        # Dynamically infer feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            feat_dim = self.net(dummy).view(1, -1).shape[1]

        self.out = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x
