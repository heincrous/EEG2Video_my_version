# ==========================================
# Config block (tunable but defaults unchanged)
# ==========================================
CONFIG = {
    "dropout": 0.0,               # dropout probability
    "layer_widths": [512, 256],   # hidden layer sizes
    "activation": "GELU",         # activation type: "GELU", "ReLU", "SiLU", etc.
    "normalization": None,        # "BatchNorm1d", "LayerNorm", or None
    "input_dim": {"C": 62, "T": 5}  # default input shape
}


# ==========================================
# Model definition
# ==========================================
import torch
import torch.nn as nn


class mlpnet(nn.Module):
    def __init__(self, out_dim, input_dim=None, cfg=CONFIG):
        super(mlpnet, self).__init__()

        # pull dimensions
        C = cfg["input_dim"]["C"]
        T = cfg["input_dim"]["T"]
        in_dim = input_dim or (C * T)

        w = cfg["layer_widths"]
        p = cfg["dropout"]

        # activation
        act_fn = getattr(nn, cfg["activation"])() if hasattr(nn, cfg["activation"]) else nn.GELU()

        # normalization factory
        def norm_layer(dim):
            norm = cfg["normalization"]
            if norm == "BatchNorm1d":
                return nn.BatchNorm1d(dim)
            elif norm == "LayerNorm":
                return nn.LayerNorm(dim)
            else:
                return nn.Identity()

        # network definition
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, w[0]),
            norm_layer(w[0]),
            act_fn,
            nn.Dropout(p),

            nn.Linear(w[0], w[1]),
            norm_layer(w[1]),
            act_fn,
            nn.Dropout(p),

            nn.Linear(w[1], out_dim),
        )

    def forward(self, x):
        return self.net(x)
