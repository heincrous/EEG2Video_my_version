# ==========================================
# fusion_model.py
# Fusion model combining raw EEG, DE, and PSD encoders
# ==========================================

# === Third-party libraries ===
import torch
import torch.nn as nn

# === Repo imports ===
from models import glfnet, glfnet_mlp


# ==========================================
# FusionNet definition
# ==========================================
class FusionNet(nn.Module):
    def __init__(self, out_dim, emb_dim, C, T, de_dim, psd_dim):
        super(FusionNet, self).__init__()

        # Raw EEG encoder (windows/segments)
        self.raw_encoder = glfnet(out_dim=emb_dim, emb_dim=emb_dim, C=C, T=T)

        # DE encoder
        self.de_encoder = glfnet_mlp(out_dim=emb_dim, emb_dim=emb_dim, input_dim=de_dim)

        # PSD encoder
        self.psd_encoder = glfnet_mlp(out_dim=emb_dim, emb_dim=emb_dim, input_dim=psd_dim)

        # Fusion classifier
        self.classifier = nn.Linear(emb_dim * 3, out_dim)

    def forward(self, raw, de, psd):
        raw_feat = self.raw_encoder(raw)
        de_feat  = self.de_encoder(de)
        psd_feat = self.psd_encoder(psd)

        fused = torch.cat([raw_feat, de_feat, psd_feat], dim=1)
        return self.classifier(fused)
