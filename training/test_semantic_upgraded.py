# ==========================================
# Semantic Predictor Evaluation (Block 7 Test, BLIP_EEG_bundle-based)
# ==========================================

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F

# === Repo imports ===
repo_root = "/content/EEG2Video_my_version"
sys.path.append(repo_root)
from core_files.models import (
    eegnet, shallownet, deepnet, tsconv, conformer, mlpnet,
    glfnet_mlp, glmnet
)

# === Wrappers ===
class WindowEncoderWrapper(torch.nn.Module):
    def __init__(self, base_encoder, out_dim, reduce="mean"):
        super().__init__()
        self.base = base_encoder
        self.reduce = reduce
    def forward(self, x):  # (B,W,C,T)
        B, W, C, T = x.shape
        x = x.view(B*W, 1, C, T)
        feats = self.base(x)
        feats = feats.view(B, W, -1)
        if self.reduce == "mean":
            return feats.mean(1)
        elif self.reduce == "none":
            return feats.view(B, -1)
        else:
            raise ValueError("reduce must be 'mean' or 'none'")

class ReshapeWrapper(torch.nn.Module):
    def __init__(self, base_model, n_tokens=77):
        super().__init__()
        self.base = base_model
        self.n_tokens = n_tokens
    def forward(self, x):
        out = self.base(x)
        return out.view(out.size(0), self.n_tokens, 768)

# === Cosine similarity across tokens ===
def tokenwise_cosine(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return float((a * b).sum(-1).mean())

# === Build class prototypes (blocks 0–5 BLIP embeddings) ===
def build_prototypes(blip_emb, loss_type):
    prototypes = {}
    for class_id in range(40):
        clips = blip_emb[:6, class_id]      # (6,5,77,768)
        clips = clips.reshape(-1,77,768)    # (30,77,768)
        proto = clips.mean(0)
        if "cosine" in loss_type:
            proto = proto / (np.linalg.norm(proto, axis=-1, keepdims=True) + 1e-8)
        prototypes[class_id] = proto
    return prototypes

# === Model builder (consistent with training) ===
def build_model(feature_type, encoder_type, output_dim, input_dim=None, window_reduce="mean"):
    if feature_type in ["DE","PSD"]:
        if encoder_type == "mlp":
            return mlpnet(out_dim=output_dim, input_dim=input_dim)
        elif encoder_type == "glfnet_mlp":
            return glfnet_mlp(out_dim=output_dim, emb_dim=256, input_dim=input_dim)

    elif feature_type == "segments":
        if encoder_type == "mlp":
            return mlpnet(out_dim=output_dim, input_dim=input_dim)
        elif encoder_type == "eegnet":
            return eegnet(out_dim=output_dim, C=62, T=400)
        elif encoder_type == "shallownet":
            return shallownet(out_dim=output_dim, C=62, T=400)
        elif encoder_type == "deepnet":
            return deepnet(out_dim=output_dim, C=62, T=400)
        elif encoder_type == "tsconv":
            return tsconv(out_dim=output_dim, C=62, T=400)
        elif encoder_type == "glmnet":
            return glmnet(out_dim=output_dim, emb_dim=256, C=62, T=400)

    elif feature_type == "windows":
        if encoder_type == "mlp":
            return mlpnet(out_dim=output_dim, input_dim=input_dim)
        elif encoder_type == "conformer":
            return conformer(out_dim=output_dim)

    raise ValueError(f"Invalid combination: {feature_type}, {encoder_type}")

# === Main ===
if __name__ == "__main__":
    bundle_path = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_EEG_bundle.npz"
    ckpt_root   = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"

    ckpts = [f for f in os.listdir(ckpt_root) if f.startswith("semantic_predictor_") and f.endswith(".pt")]
    for i, ck in enumerate(ckpts):
        print(f"[{i}] {ck}")
    choice    = int(input("\nEnter checkpoint index: ").strip())
    ckpt_file = ckpts[choice]
    tag       = ckpt_file.replace("semantic_predictor_", "").replace(".pt", "")
    ckpt_path = os.path.join(ckpt_root, ckpt_file)
    scaler_path = os.path.join(ckpt_root, f"scaler_{tag}.pkl")

    # Parse tag (feature, encoder, loss)
    parts = tag.split("_")
    feature_type = parts[0]
    loss_type    = parts[-1]

    if feature_type == "windows":
        window_reduce = parts[-2]
        encoder_type = "_".join(parts[1:-2])
    else:
        window_reduce = None
        encoder_type = "_".join(parts[1:-1])

    print(f"\n[Config] Feature={feature_type}, Encoder={encoder_type}, Loss={loss_type}, WindowReduce={window_reduce}")

    # Load scaler
    with open(scaler_path,"rb") as f:
        scaler = pickle.load(f)

    # Load bundle
    data = np.load(bundle_path, allow_pickle=True)
    blip_emb = data["BLIP_embeddings"]   # (7,40,5,77,768)
    eeg_dict = data["EEG_data"].item()

    # Build prototypes from train blocks
    prototypes = build_prototypes(blip_emb, loss_type)

    # Build model
    input_dim = scaler.mean_.shape[0]
    output_dim = 77*768
    base_model = build_model(feature_type, encoder_type, output_dim, input_dim, window_reduce)
    model = ReshapeWrapper(base_model, n_tokens=77).cuda()
    checkpoint = torch.load(ckpt_path,map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # === Evaluation on block 6 (test = block 7) ===
    correct_top1 = correct_top5 = total = 0
    mse_vals, cos_vals, token_cos_vals, corr_vals = [], [], [], []

    for subj, feats in eeg_dict.items():
        eeg = feats[f"EEG_{feature_type}"][6]   # (40,5,...)
        txt = blip_emb[6]                       # (40,5,77,768)

        for ci in range(40):
            for cj in range(5):
                eeg_flat = eeg[ci,cj].reshape(-1)
                eeg_scaled = scaler.transform([eeg_flat])[0]

                if feature_type == "windows":
                    eeg_tensor = torch.tensor(eeg_scaled.reshape(7,62,100),dtype=torch.float32).unsqueeze(0).cuda()
                elif feature_type == "segments":
                    eeg_tensor = torch.tensor(eeg_scaled.reshape(1,62,400),dtype=torch.float32).unsqueeze(0).cuda()
                elif feature_type in ["DE","PSD"]:
                    eeg_tensor = torch.tensor(eeg_scaled.reshape(62,5),dtype=torch.float32).unsqueeze(0).cuda()

                true_class = ci
                true_emb = txt[ci,cj]

                with torch.no_grad():
                    pred_emb = model(eeg_tensor).squeeze(0).cpu().numpy()

                if "cosine" in loss_type:
                    pred_emb = pred_emb / (np.linalg.norm(pred_emb,axis=-1,keepdims=True)+1e-8)

                # classification
                sims = {cid: tokenwise_cosine(pred_emb, proto) for cid,proto in prototypes.items()}
                ranked = sorted(sims.items(), key=lambda x:x[1], reverse=True)
                top1, top5 = ranked[0][0],[cid for cid,_ in ranked[:5]]
                correct_top1 += int(top1==true_class)
                correct_top5 += int(true_class in top5)
                total += 1

                # metrics
                mse_vals.append(np.mean((pred_emb-true_emb)**2))
                cos = np.dot(pred_emb.flatten(),true_emb.flatten()) / (
                    np.linalg.norm(pred_emb.flatten())*np.linalg.norm(true_emb.flatten())+1e-8)
                cos_vals.append(cos)
                a = pred_emb/(np.linalg.norm(pred_emb,axis=-1,keepdims=True)+1e-8)
                b = true_emb/(np.linalg.norm(true_emb,axis=-1,keepdims=True)+1e-8)
                token_cos_vals.append((a*b).sum(-1).mean())
                corr_vals.append(np.corrcoef(pred_emb.flatten(),true_emb.flatten())[0,1])

    print(f"\nTop-1 Accuracy: {correct_top1/total:.4f}")
    print(f"Top-5 Accuracy: {correct_top5/total:.4f}")
    print(f"Mean MSE: {np.mean(mse_vals):.6f}")
    print(f"Mean Cosine similarity: {np.mean(cos_vals):.4f}")
    print(f"Mean Token-wise cosine: {np.mean(token_cos_vals):.4f}")
    print(f"Mean Pearson correlation: {np.mean(corr_vals):.4f}")
