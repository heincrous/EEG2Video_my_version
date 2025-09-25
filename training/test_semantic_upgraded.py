# ==========================================
# Semantic Predictor Evaluation (Multi-bundle, Embedding + Classification Metrics)
# ==========================================

import os
import sys
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

# === Repo imports ===
repo_root = "/content/EEG2Video_my_version"
sys.path.append(repo_root)
from core_files.models import (
    eegnet, shallownet, deepnet, tsconv, conformer, mlpnet,
    glfnet, glfnet_mlp, glmnet   # <-- added glmnet
)

# === Wrapper for CNN-based encoders ===
class WindowEncoderWrapper(torch.nn.Module):
    def __init__(self, base_encoder, out_dim):
        super().__init__()
        self.base = base_encoder
    def forward(self, x):  # (B,7,62,100)
        return self.base(x)  # glmnet already returns (B,W,feat)

# === Reshape wrapper (flat -> (77,768)) ===
class ReshapeWrapper(torch.nn.Module):
    def __init__(self, base_model, n_tokens=77):
        super().__init__()
        self.base = base_model
        self.n_tokens = n_tokens
    def forward(self, x):
        out = self.base(x)
        return out.view(out.size(0), self.n_tokens, 768)

# === Build class prototypes (token-aware) ===
def build_class_prototypes(bundle_path, loss_type):
    data = np.load(bundle_path, allow_pickle=True)
    blip_embeddings = data["BLIP_embeddings"]  # (N,77,768)
    keys = data["keys"]

    class_groups = defaultdict(list)
    for i, key in enumerate(keys):
        class_id = int(next(p for p in key.split("/") if p.startswith("class")).replace("class", "").split("_")[0])
        class_groups[class_id].append(blip_embeddings[i])

    prototypes = {}
    for cid, embs in class_groups.items():
        avg_emb = np.mean(np.stack(embs), axis=0)  # (77,768)
        if "cosine" in loss_type:
            avg_emb = avg_emb / (np.linalg.norm(avg_emb, axis=-1, keepdims=True) + 1e-8)
        prototypes[cid] = avg_emb
    return prototypes

# === Model builder ===
def build_model(feature_type, encoder_type, output_dim, input_dim=None):
    if feature_type in ["DE", "PSD"]:
        if encoder_type == "mlp":
            return mlpnet(out_dim=output_dim, input_dim=input_dim)
        elif encoder_type == "glfnet":
            return glfnet(out_dim=output_dim, emb_dim=256, C=62, T=5)
        elif encoder_type == "glfnet_mlp":
            return glfnet_mlp(out_dim=output_dim, emb_dim=256, input_dim=input_dim)
        else:
            raise ValueError(f"Unknown encoder type for DE/PSD: {encoder_type}")
    elif feature_type == "windows":
        if encoder_type == "mlp":
            return mlpnet(out_dim=output_dim, input_dim=input_dim)
        elif encoder_type == "eegnet":
            return WindowEncoderWrapper(eegnet(out_dim=output_dim, C=62, T=100), out_dim=output_dim)
        elif encoder_type == "shallownet":
            return WindowEncoderWrapper(shallownet(out_dim=output_dim, C=62, T=100), out_dim=output_dim)
        elif encoder_type == "deepnet":
            return WindowEncoderWrapper(deepnet(out_dim=output_dim, C=62, T=100), out_dim=output_dim)
        elif encoder_type == "tsconv":
            return WindowEncoderWrapper(tsconv(out_dim=output_dim, C=62, T=100), out_dim=output_dim)
        elif encoder_type == "conformer":
            return conformer(out_dim=output_dim)
        elif encoder_type == "glmnet":   # <-- added glmnet support
            return WindowEncoderWrapper(glmnet(out_dim=output_dim, emb_dim=256, C=62, T=100), out_dim=output_dim)
        else:
            raise ValueError(f"Unknown encoder type for windows: {encoder_type}")
    else:
        raise ValueError(f"Invalid feature type: {feature_type}")

# === Cosine similarity across tokens ===
def tokenwise_cosine(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return float((a * b).sum(-1).mean())

# === Main ===
if __name__ == "__main__":
    bundle_root = "/content/drive/MyDrive/EEG2Video_data/processed/SubjectBundles"
    ckpt_root   = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"

    ckpts = [f for f in os.listdir(ckpt_root) if f.startswith("semantic_predictor_") and f.endswith(".pt")]
    for i, ck in enumerate(ckpts):
        print(f"[{i}] {ck}")
    choice    = int(input("\nEnter checkpoint index: ").strip())
    ckpt_file = ckpts[choice]
    tag       = ckpt_file.replace("semantic_predictor_", "").replace(".pt", "")
    ckpt_path = os.path.join(ckpt_root, ckpt_file)
    scaler_path = os.path.join(ckpt_root, f"scaler_{tag}.pkl")

    # Parse naming
    parts = tag.split("_")
    parts = parts[1:]
    feature_type = parts[0]
    loss_type    = parts[-1]
    encoder_type = "_".join(parts[1:-1])

    print(f"\n[Config] Feature={feature_type}, Encoder={encoder_type}, Loss={loss_type}")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    input_dim = scaler.mean_.shape[0]
    output_dim = 77 * 768
    base_model = build_model(feature_type, encoder_type, output_dim, input_dim)
    model = ReshapeWrapper(base_model, n_tokens=77).cuda()
    checkpoint = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # === Build prototypes from first train bundle ===
    train_bundles = sorted([f for f in os.listdir(bundle_root) if f.endswith("_train.npz")])
    proto_bundle = os.path.join(bundle_root, train_bundles[0])
    prototypes = build_class_prototypes(proto_bundle, loss_type)

    # === Collect test samples from ALL test bundles ===
    test_bundles = sorted([f for f in os.listdir(bundle_root) if f.endswith("_test.npz")])
    class_groups = defaultdict(list)

    test_samples = []
    for tb in test_bundles:
        data = np.load(os.path.join(bundle_root, tb), allow_pickle=True)
        eeg_data = data[f"EEG_{feature_type}"]
        blip_data = data["BLIP_embeddings"]
        keys = data["keys"]
        for i, key in enumerate(keys):
            class_id = int(next(p for p in key.split("/") if p.startswith("class")).replace("class","").split("_")[0])
            class_groups[class_id].append((eeg_data[i], blip_data[i]))
            test_samples.append((class_id, eeg_data[i], blip_data[i]))

    # Choose one random sample per class across ALL bundles
    chosen = {cid: random.choice(samples) for cid, samples in class_groups.items() if samples}

    # === Evaluation ===
    correct_top1 = correct_top5 = total = 0
    mse_vals, cos_vals, token_cos_vals, corr_vals = [], [], [], []

    for true_class, (eeg, true_emb) in chosen.items():
        eeg_flat = eeg.reshape(-1)
        eeg_scaled = scaler.transform([eeg_flat])[0]

        if feature_type == "windows":
            eeg_tensor = torch.tensor(eeg_scaled.reshape(7,62,100), dtype=torch.float32).unsqueeze(0).cuda()
        else:
            eeg_tensor = torch.tensor(eeg_scaled.reshape(62,5), dtype=torch.float32).unsqueeze(0).cuda()

        with torch.no_grad():
            pred_emb = model(eeg_tensor).squeeze(0).cpu().numpy()

        if "cosine" in loss_type:
            pred_emb = pred_emb / (np.linalg.norm(pred_emb, axis=-1, keepdims=True) + 1e-8)

        # === Classification via prototypes ===
        sims = {cid: tokenwise_cosine(pred_emb, proto) for cid, proto in prototypes.items()}
        ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        top1, top5 = ranked[0][0], [cid for cid, _ in ranked[:5]]
        correct_top1 += int(top1 == true_class)
        correct_top5 += int(true_class in top5)
        total += 1

        # === Embedding-level metrics ===
        mse_vals.append(np.mean((pred_emb - true_emb)**2))
        cos = np.dot(pred_emb.flatten(), true_emb.flatten()) / (
            np.linalg.norm(pred_emb.flatten()) * np.linalg.norm(true_emb.flatten()) + 1e-8
        )
        cos_vals.append(cos)

        a = pred_emb / (np.linalg.norm(pred_emb, axis=-1, keepdims=True) + 1e-8)
        b = true_emb / (np.linalg.norm(true_emb, axis=-1, keepdims=True) + 1e-8)
        token_cos_vals.append((a * b).sum(-1).mean())

        corr_vals.append(np.corrcoef(pred_emb.flatten(), true_emb.flatten())[0,1])

        print(f"True={true_class} | Pred@1={top1} | Top-5={top5} | "
              f"MSE={mse_vals[-1]:.6f}, Cos={cos_vals[-1]:.4f}, "
              f"TokenCos={token_cos_vals[-1]:.4f}, Corr={corr_vals[-1]:.4f}")

    print(f"\nTop-1 Accuracy: {correct_top1/total:.4f}")
    print(f"Top-5 Accuracy: {correct_top5/total:.4f}")
    print(f"Mean MSE: {np.mean(mse_vals):.6f}")
    print(f"Mean Cosine similarity: {np.mean(cos_vals):.4f}")
    print(f"Mean Token-wise cosine: {np.mean(token_cos_vals):.4f}")
    print(f"Mean Pearson correlation: {np.mean(corr_vals):.4f}")
