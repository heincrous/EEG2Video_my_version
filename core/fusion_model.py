# ==========================================
# Flexible Fusion model (works with training script)
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
def build_encoder(name, out_dim, emb_dim=None, C=None, T=None, input_dim=None, feature_type=None):
    # enforce valid combinations
    if feature_type == "raw":
        if name not in ["shallownet", "deepnet", "eegnet", "tsconv", "conformer", "glfnet"]:
            raise ValueError(f"{name} is not valid for raw EEG. Use a conv-based model.")
    if feature_type in ["de", "psd"]:
        if name not in ["mlpnet", "glfnet_mlp"]:
            raise ValueError(f"{name} is not valid for {feature_type.upper()}. Use an MLP-based model.")

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
# FusionNet definition (flexible)
# ==========================================
class FusionNet(nn.Module):
    def __init__(self, encoder_cfgs, num_classes=40, emb_dim=128):
        """
        encoder_cfgs: dict mapping feature_name -> (model_name, kwargs)
          e.g. {
            "raw": ("glfnet", {"out_dim": emb_dim, "emb_dim": emb_dim, "C": 62, "T": 200}),
            "de":  ("glfnet_mlp", {"out_dim": emb_dim, "emb_dim": emb_dim, "input_dim": 310}),
          }
        """
        super().__init__()
        self.encoders = nn.ModuleDict()
        for feat_type, (model_name, kwargs) in encoder_cfgs.items():
            self.encoders[feat_type] = build_encoder(model_name, feature_type=feat_type, **kwargs)

        # infer feature dimension dynamically
        total_dim = sum([list(enc.modules())[-1].out_features for enc in self.encoders.values()])
        self.classifier = nn.Linear(total_dim, num_classes)

    def forward(self, inputs, return_feats=False):
        feats = []
        for name, encoder in self.encoders.items():
            if name not in inputs:
                continue
            feats.append(encoder(inputs[name]))
        fused = torch.cat(feats, dim=-1)
        if return_feats:
            return fused
        return self.classifier(fused)
