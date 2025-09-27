# ==========================================
# fusion_model.py
# Configurable Fusion model combining raw EEG, DE, and PSD encoders
# ==========================================

# === Third-party libraries ===
import torch
import torch.nn as nn

# === Repo imports ===
from .models import (
    shallownet, deepnet, eegnet, tsconv, conformer, glfnet,
    mlpnet, glfnet_mlp
)


# ==========================================
# Helper: build encoder from name
# ==========================================
def build_encoder(name, out_dim, emb_dim=None, C=None, T=None, input_dim=None):
    if name == "shallownet":
        return shallownet(out_dim, C, T)
    elif name == "deepnet":
        return deepnet(out_dim, C, T)
    elif name == "eegnet":
        return eegnet(out_dim, C, T)
    elif name == "tsconv":
        return tsconv(out_dim, C, T)
    elif name == "conformer":
        return conformer(emb_size=emb_dim, out_dim=out_dim)
    elif name == "glfnet":
        return glfnet(out_dim, emb_dim, C, T)
    elif name == "mlpnet":
        return mlpnet(out_dim, input_dim)
    elif name == "glfnet_mlp":
        return glfnet_mlp(out_dim, emb_dim, input_dim)
    else:
        raise ValueError(f"Unknown encoder name: {name}")


# ==========================================
# FusionNet definition
# ==========================================
class FusionNet(nn.Module):
    def __init__(self,
                 out_dim,
                 emb_dim,
                 C,
                 T,
                 de_dim,
                 psd_dim,
                 raw_model="glfnet",
                 de_model="glfnet_mlp",
                 psd_model="glfnet_mlp"):

        super(FusionNet, self).__init__()

        # Raw EEG encoder
        self.raw_encoder = build_encoder(
            raw_model, out_dim=emb_dim, emb_dim=emb_dim, C=C, T=T
        )

        # DE encoder
        self.de_encoder = build_encoder(
            de_model, out_dim=emb_dim, emb_dim=emb_dim, input_dim=de_dim
        )

        # PSD encoder
        self.psd_encoder = build_encoder(
            psd_model, out_dim=emb_dim, emb_dim=emb_dim, input_dim=psd_dim
        )

        # Fusion classifier
        self.classifier = nn.Linear(emb_dim * 3, out_dim)

    def forward(self, raw, de, psd):
        raw_feat = self.raw_encoder(raw)
        de_feat  = self.de_encoder(de)
        psd_feat = self.psd_encoder(psd)

        fused = torch.cat([raw_feat, de_feat, psd_feat], dim=1)
        return self.classifier(fused)
