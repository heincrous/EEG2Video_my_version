# ==========================================
# Config block (tunable but defaults unchanged)
# ==========================================
CONFIG = {
    # model structure
    "dropout": 0.5,                   # dropout probability
    "layer_width": 40,                # number of convolutional filters
    "kernel_size": (1, 25),           # temporal kernel
    "pool_size": (1, 51),             # pooling window
    "pool_stride": (1, 5),            # pooling stride
    "activation": "ELU",              # "ELU", "GELU", "SiLU", etc.
    "normalization": "BatchNorm",     # "BatchNorm", "LayerNorm", "GroupNorm"
    "input_dim": {"C": 62, "T": 400}  # default input shape
}


# ==========================================
# Model definition
# ==========================================
import torch
import torch.nn as nn


class shallownet(nn.Module):
    def __init__(self, out_dim, C=None, T=None, cfg=CONFIG):
        super(shallownet, self).__init__()

        # pull from config or override
        C = C or cfg["input_dim"]["C"]
        T = T or cfg["input_dim"]["T"]
        w = cfg["layer_width"]
        p = cfg["dropout"]

        # choose activation
        act_fn = getattr(nn, cfg["activation"])() if hasattr(nn, cfg["activation"]) else nn.ELU()

        # choose normalization
        if cfg["normalization"] == "BatchNorm":
            norm = nn.BatchNorm2d(w)
        elif cfg["normalization"] == "LayerNorm":
            norm = nn.LayerNorm([w, 1, 1])
        elif cfg["normalization"] == "GroupNorm":
            norm = nn.GroupNorm(4, w)
        else:
            norm = nn.BatchNorm2d(w)

        self.net = nn.Sequential(
            nn.Conv2d(1, w, cfg["kernel_size"], (1, 1)),
            nn.Conv2d(w, w, (C, 1), (1, 1)),
            norm,
            act_fn,
            nn.AvgPool2d(cfg["pool_size"], cfg["pool_stride"]),
            nn.Dropout(p),
        )

        # dynamically compute flattened feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            feat_dim = self.net(dummy).view(1, -1).shape[1]

        self.out = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x
