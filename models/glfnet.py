# ==========================================
# Config block (tunable but defaults unchanged)
# ==========================================
CONFIG = {
    "dropout": 0.5,              # dropout probability
    "layer_width": 40,           # same as shallownet
    "activation": "ELU",         # activation function
    "normalization": "BatchNorm" # normalization type
}

# ==========================================
# Model definition
# ==========================================
import torch
import torch.nn as nn
from .shallownet import shallownet  # assumes same folder


class glfnet(nn.Module):
    def __init__(self, out_dim, emb_dim, C, T, cfg=CONFIG):
        super(glfnet, self).__init__()
        self.cfg = cfg

        # Global EEG branch
        self.globalnet = shallownet(emb_dim, C, T, cfg=self.cfg)

        # Occipital electrodes (channels 50–61 inclusive)
        self.occipital_index = list(range(50, 62))
        self.occipital_localnet = shallownet(emb_dim, 12, T, cfg=self.cfg)

        # Fusion projection
        self.out = nn.Linear(emb_dim * 2, out_dim)

    def forward(self, x):
        # Global representation
        global_feature = self.globalnet(x)
        global_feature = global_feature.view(x.size(0), -1)

        # Local (occipital) representation
        occipital_x = x[:, :, self.occipital_index, :]
        occipital_feature = self.occipital_localnet(occipital_x)

        # Concatenate and project
        out = self.out(torch.cat((global_feature, occipital_feature), 1))
        return out
