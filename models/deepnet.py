# ==========================================
# Config block (tunable but defaults unchanged)
# ==========================================
CONFIG = {
    "dropout": 0.5,               # dropout probability
    "layer_widths": [25, 50, 100, 200],  # conv channel sizes
    "kernel_size": (1, 10),       # temporal convolution kernel
    "activation": "ELU",          # activation type
    "normalization": "BatchNorm"  # normalization type
}


# ==========================================
# Model definition
# ==========================================
import torch
import torch.nn as nn


class deepnet(nn.Module):
    def __init__(self, out_dim, C, T, cfg=CONFIG):
        super(deepnet, self).__init__()
        p = cfg["dropout"]
        w = cfg["layer_widths"]
        k = cfg["kernel_size"]

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

        # convolutional stack
        self.net = nn.Sequential(
            nn.Conv2d(1, w[0], k, (1, 1)),
            nn.Conv2d(w[0], w[0], (C, 1), (1, 1)),
            norm_layer(w[0]),
            act_fn,
            nn.MaxPool2d((1, 2)),
            nn.Dropout(p),

            nn.Conv2d(w[0], w[1], k),
            norm_layer(w[1]),
            act_fn,
            nn.MaxPool2d((1, 2)),
            nn.Dropout(p),

            nn.Conv2d(w[1], w[2], k),
            norm_layer(w[2]),
            act_fn,
            nn.MaxPool2d((1, 2)),
            nn.Dropout(p),

            nn.Conv2d(w[2], w[3], k),
            norm_layer(w[3]),
            act_fn,
            nn.MaxPool2d((1, 2)),
            nn.Dropout(p),
        )

        # dynamically infer flattened feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            feat_dim = self.net(dummy).view(1, -1).shape[1]

        self.out = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x
