# ==========================================
# Semantic Predictor Evaluation (Row-aware, token-wise similarity)
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
from core_files.models import eegnet, shallownet, deepnet, tsconv, conformer, mlpnet


# === Wrapper for windowed encoders ===
class WindowEncoderWrapper(torch.nn.Module):
    def __init__(self, base_encoder, out_dim):
        super().__init__()
        self.base = base_encoder

    def forward(self, x):  # (B,7,62,100)
        B, W, C, T = x.shape
        x = x.view(B*W, 1, C, T)
        feats = self.base(x)
        feats = feats.view(B, W, -1)
        return feats.mean(1)  # (B, out_dim)


# === Build class prototypes (token-aware) ===
def build_class_prototypes(bundle_path, loss_type):
    data = np.load(bundle_path, allow_pickle=True)
    blip_embeddings = data["BLIP_embeddings"]  # (N,77,768)
    keys = data["keys"]

    class_groups = defaultdict(list)
    for i, key in enumerate(keys):
        parts = key.split("/")
        class_token = next(p for p in parts if p.startswith("class"))
        class_id = int(class_token.replace("class", "").split("_")[0])
        class_groups[class_id].append(blip_embeddings[i])  # keep full (77,768)

    prototypes = {}
    for cid, embs in class_groups.items():
        avg_emb = np.mean(np.stack(embs), axis=0)  # (77,768)
        if loss_type == "cosine":
            # normalize each row separately
            norms = np.linalg.norm(avg_emb, axis=-1, keepdims=True) + 1e-8
            avg_emb = avg_emb / norms
        prototypes[cid] = avg_emb
    return prototypes


# === Model builder ===
def build_model(feature_type, encoder_type, output_dim, input_dim=None):
    if feature_type in ["DE", "PSD"]:
        return mlpnet(out_dim=output_dim, input_dim=input_dim)
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
            return WindowEncoderWrapper(conformer(out_dim=output_dim), out_dim=output_dim)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    else:
        raise ValueError(f"Invalid feature type: {feature_type}")


# === Cosine similarity across tokens ===
def tokenwise_cosine(a, b):
    # a, b: (77,768)
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return float((a * b).sum(-1).mean())  # average across 77 tokens


# === Main ===
if __name__ == "__main__":
    bundle_root = "/content/drive/MyDrive/EEG2Video_data/processed/SubjectBundles"
    eeg_root    = "/content/drive/MyDrive/EEG2Video_data/processed"
    ckpt_root   = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"

    ckpts = [f for f in os.listdir(ckpt_root) if f.startswith("semantic_predictor_") and f.endswith(".pt")]
    for i, ck in enumerate(ckpts):
        print(f"[{i}] {ck}")
    choice    = int(input("\nEnter checkpoint index: ").strip())
    ckpt_file = ckpts[choice]
    tag       = ckpt_file.replace("semantic_predictor_", "").replace(".pt", "")
    ckpt_path = os.path.join(ckpt_root, ckpt_file)
    scaler_path = os.path.join(ckpt_root, f"scaler_{tag}.pkl")

    parts = tag.split("_")
    feature_type = parts[-3]
    encoder_type = parts[-2]
    loss_type    = parts[-1]

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    input_dim = scaler.mean_.shape[0]
    output_dim = 77 * 768
    model = build_model(feature_type, encoder_type, output_dim, input_dim).cuda()
    checkpoint = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Build prototypes
    train_bundles = sorted([f for f in os.listdir(bundle_root) if f.endswith("_train.npz")])
    proto_bundle = os.path.join(bundle_root, train_bundles[0])
    prototypes = build_class_prototypes(proto_bundle, loss_type)

    # Pick test samples
    test_list_path = os.path.join(eeg_root, f"EEG_{feature_type}", "test_list.txt")
    with open(test_list_path, "r") as f:
        test_lines = [line.strip() for line in f if line.strip()]

    class_groups = defaultdict(list)
    for rel_path in test_lines:
        parts = rel_path.split("/")
        class_token = next(p for p in parts if p.startswith("class"))
        class_id = int(class_token.replace("class", "").split("_")[0])
        class_groups[class_id].append(rel_path)

    chosen = [random.choice(v) for v in class_groups.values() if v]

    # Evaluation
    correct_top1 = correct_top5 = total = 0
    for rel_path in chosen:
        eeg_path = os.path.join(eeg_root, f"EEG_{feature_type}", rel_path)
        true_class = int(next(p for p in rel_path.split("/") if p.startswith("class")).replace("class","").split("_")[0])

        eeg = np.load(eeg_path)
        eeg_flat = eeg.reshape(-1)
        eeg_scaled = scaler.transform([eeg_flat])[0]

        if feature_type == "windows":
            eeg_tensor = torch.tensor(eeg_scaled.reshape(7,62,100), dtype=torch.float32).unsqueeze(0).cuda()
        else:
            eeg_tensor = torch.tensor(eeg_scaled.reshape(62,5), dtype=torch.float32).unsqueeze(0).cuda()

        with torch.no_grad():
            pred_emb = model(eeg_tensor).squeeze(0).cpu().numpy().reshape(77,768)

        if loss_type == "cosine":
            pred_emb = pred_emb / (np.linalg.norm(pred_emb, axis=-1, keepdims=True) + 1e-8)

        sims = {cid: tokenwise_cosine(pred_emb, proto) for cid, proto in prototypes.items()}
        ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)

        top1, top5 = ranked[0][0], [cid for cid,_ in ranked[:5]]
        correct_top1 += int(top1 == true_class)
        correct_top5 += int(true_class in top5)
        total += 1

        print(f"True={true_class} | Pred@1={top1} | Top-5={top5} | {'Correct' if top1==true_class else 'Wrong'}")

    print(f"\nTop-1 Accuracy: {correct_top1/total:.4f}")
    print(f"Top-5 Accuracy: {correct_top5/total:.4f}")
