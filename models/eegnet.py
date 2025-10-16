# ==========================================
# Config block (tunable but defaults unchanged)
# ==========================================
CONFIG = {
    "dropout": 0.5,               # dropout probability
    "layer_widths": [8, 16, 16],  # convolutional channel sizes
    "kernel_sizes": [(1, 64), (1, 16)],  # temporal filter sizes
    "activation": "ELU",          # activation type
    "normalization": "BatchNorm"  # normalization type
}

# ==========================================
# Model definition
# ==========================================
import torch
import torch.nn as nn


class eegnet(nn.Module):
    def __init__(self, out_dim, C, T, cfg=CONFIG):
        super(eegnet, self).__init__()
        p = cfg["dropout"]
        w = cfg["layer_widths"]
        k = cfg["kernel_sizes"]

        # activation
        act_fn = getattr(nn, cfg["activation"])() if hasattr(nn, cfg["activation"]) else nn.ELU()

        # normalization factory
        def norm_layer(channels):
            if cfg["normalization"] == "BatchNorm":
                return nn.BatchNorm2d(channels)
            elif cfg["normalization"] == "LayerNorm":
                return nn.LayerNorm([channels, 1, 1])
            elif cfg["normalization"] == "GroupNorm":
                return nn.GroupNorm(4, channels)
            else:
                return nn.BatchNorm2d(channels)

        # core CNN
        self.net = nn.Sequential(
            nn.Conv2d(1, w[0], k[0]),
            norm_layer(w[0]),

            nn.Conv2d(w[0], w[1], (C, 1)),
            norm_layer(w[1]),
            act_fn,
            nn.AvgPool2d((1, 2)),
            nn.Dropout(p),

            nn.Conv2d(w[1], w[2], k[1]),
            norm_layer(w[2]),
            act_fn,
            nn.AvgPool2d((1, 2)),
            nn.Dropout2d(p)
        )

        # dynamic output dimension inference
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            feat_dim = self.net(dummy).view(1, -1).shape[1]

        self.out = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x
