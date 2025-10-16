# ==========================================
# Config block (inherits from shallownet)
# ==========================================
import torch
import torch.nn as nn
from .shallownet import shallownet, CONFIG as SHALLOW_CONFIG  # import shallownet config
import copy

# copy to avoid accidental mutation
CONFIG = copy.deepcopy(SHALLOW_CONFIG)
CONFIG.update({
    "dropout": 0.5,              # override if needed
    "layer_width": 40            # override if needed
})


# ==========================================
# Model definition
# ==========================================
class glfnet(nn.Module):
    def __init__(self, out_dim, emb_dim, C, T, cfg=CONFIG):
        super(glfnet, self).__init__()
        self.cfg = cfg

        # Global EEG branch (full channel set)
        self.globalnet = shallownet(emb_dim, C, T, cfg=self.cfg)

        # Occipital subset (channels 50–61 inclusive)
        self.occipital_index = list(range(50, 62))
        self.occipital_localnet = shallownet(emb_dim, 12, T, cfg=self.cfg)

        # Fusion projection
        self.out = nn.Linear(emb_dim * 2, out_dim)

    def forward(self, x):
        global_feature = self.globalnet(x)
        global_feature = global_feature.view(x.size(0), -1)

        occipital_x = x[:, :, self.occipital_index, :]
        occipital_feature = self.occipital_localnet(occipital_x)

        out = self.out(torch.cat((global_feature, occipital_feature), 1))
        return out
