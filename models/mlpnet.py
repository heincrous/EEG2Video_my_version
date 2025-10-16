# ==========================================
# Config block (tunable but defaults unchanged)
# ==========================================
CONFIG = {
    # model structure
    "dropout": 0.0,             # authors used none
    "layer_widths": [512, 256], # hidden layer sizes
    "activation": "GELU",       # GELU matches original implementation
    "normalization": None,      # no normalization layers in original
    # training / input
    "input_dim": {"C": 62, "T": 5},  # DE/PSD default shape
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

        # choose activation
        act_fn = getattr(nn, cfg["activation"])() if hasattr(nn, cfg["activation"]) else nn.GELU()

        # network definition
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, w[0]),
            act_fn,
            nn.Dropout(p),
            nn.Linear(w[0], w[1]),
            act_fn,
            nn.Dropout(p),
            nn.Linear(w[1], out_dim),
        )

    def forward(self, x):
        return self.net(x)
