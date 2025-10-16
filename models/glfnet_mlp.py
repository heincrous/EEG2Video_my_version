# ==========================================
# Config block (tunable but defaults unchanged)
# ==========================================
CONFIG = {
    "dropout": 0.0,               # dropout probability; 0 default
    "layer_widths": [256, 128],   # hidden layer sizes; [512, 256] default
    "activation": "ELU",          # "ELU", "GELU", "SiLU"; ELU default
    "normalization": "BatchNorm", # "BatchNorm", "LayerNorm", "GroupNorm"; BatchNorm default
}


# ==========================================
# Model definition
# ==========================================
import torch
import torch.nn as nn


class mlpnet(nn.Module):
    def __init__(self, out_dim, input_dim, cfg=CONFIG):
        super(mlpnet, self).__init__()
        w = cfg["layer_widths"]
        p = cfg["dropout"]

        # activation
        act_fn = getattr(nn, cfg["activation"])() if hasattr(nn, cfg["activation"]) else nn.GELU()

        # optional normalization factory
        def norm_layer(dim):
            if cfg["normalization"] == "LayerNorm":
                return nn.LayerNorm(dim)
            elif cfg["normalization"] == "BatchNorm1d":
                return nn.BatchNorm1d(dim)
            else:
                return nn.Identity()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, w[0]),
            norm_layer(w[0]),
            act_fn,
            nn.Dropout(p),

            nn.Linear(w[0], w[1]),
            norm_layer(w[1]),
            act_fn,
            nn.Dropout(p),

            nn.Linear(w[1], out_dim)
        )

    def forward(self, x):
        return self.net(x)


class glfnet_mlp(nn.Module):
    def __init__(self, out_dim, emb_dim, input_dim, cfg=CONFIG):
        super(glfnet_mlp, self).__init__()
        self.cfg = cfg

        # Global EEG branch
        self.globalnet = mlpnet(emb_dim, input_dim, cfg=self.cfg)

        # Occipital electrode subset
        self.occipital_index = list(range(50, 62))
        self.occipital_localnet = mlpnet(emb_dim, 12 * 5, cfg=self.cfg)

        # Fusion projection
        self.out = nn.Linear(emb_dim * 2, out_dim)

    def forward(self, x):
        global_feature = self.globalnet(x)
        occipital_x = x[:, self.occipital_index, :]
        occipital_feature = self.occipital_localnet(occipital_x)
        out = self.out(torch.cat((global_feature, occipital_feature), 1))
        return out
