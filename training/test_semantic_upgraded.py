# ==========================================
# Semantic Predictor Upgraded Evaluation
# Prototype Classification (1 Sample Per Class, Top-1 and Top-5)
# ==========================================

import os
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from collections import defaultdict

# === Repo imports ===
from core_files.models import eegnet, shallownet, deepnet, tsconv, conformer, mlpnet

# === Wrapper for windowed encoders ===
class WindowEncoderWrapper(torch.nn.Module):
    def __init__(self, base_encoder, out_dim):
        super().__init__()
        self.base = base_encoder

    def forward(self, x):  # x: (B,7,62,100)
        B, W, C, T = x.shape
        x = x.view(B * W, 1, C, T)
        feats = self.base(x)   # (B*W, out_dim)
        feats = feats.view(B, W, -1)
        return feats.mean(1)


# ==========================================
# Cosine similarity helper
# ==========================================
def cosine_similarity(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum().item()


# ==========================================
# Build class prototypes from one training bundle
# ==========================================
def build_class_prototypes(bundle_path):
    data = np.load(bundle_path, allow_pickle=True)
    blip_embeddings = data["BLIP_embeddings"]  # (N,77,768)
    keys = data["keys"]

    class_groups = defaultdict(list)
    for i, key in enumerate(keys):
        parts = key.split("/")
        class_token = next(p for p in parts if p.startswith("class"))
        class_id = int(class_token.replace("class", "").split("_")[0])
        emb = blip_embeddings[i].reshape(-1)
        class_groups[class_id].append(emb)

    prototypes = {cid: np.mean(np.stack(embs), axis=0) for cid, embs in class_groups.items()}
    return prototypes


# ==========================================
# Model builder
# ==========================================
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


# ==========================================
# Main evaluation
# ==========================================
if __name__ == "__main__":
    bundle_root = "/content/drive/MyDrive/EEG2Video_data/processed/SubjectBundles"
    eeg_root    = "/content/drive/MyDrive/EEG2Video_data/processed"
    ckpt_root   = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"
    log_dir     = "/content/drive/MyDrive/EEG2Video_outputs/semantic_eval"
    os.makedirs(log_dir, exist_ok=True)

    ckpts = [f for f in os.listdir(ckpt_root) if f.startswith("semantic_predictor_") and f.endswith(".pt")]
    print("\nAvailable checkpoints:")
    for i, ck in enumerate(ckpts):
        print(f"[{i}] {ck}")

    choice    = int(input("\nEnter checkpoint index: ").strip())
    ckpt_file = ckpts[choice]
    tag       = ckpt_file.replace("semantic_predictor_", "").replace(".pt", "")
    ckpt_path = os.path.join(ckpt_root, ckpt_file)
    scaler_path = os.path.join(ckpt_root, f"scaler_{tag}.pkl")

    # Parse tag
    parts = tag.split("_")
    feature_type = parts[-3]     # DE / PSD / windows
    encoder_type = parts[-2]     # mlp / eegnet / shallownet / deepnet / tsconv / conformer

    print(f"\nLoading checkpoint: {ckpt_file}")
    print(f"Loading scaler: scaler_{tag}.pkl")
    print(f"Detected feature type: {feature_type}, encoder: {encoder_type}")

    # Load scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Build model
    input_dim = scaler.mean_.shape[0]
    output_dim = 77 * 768
    model = build_model(feature_type, encoder_type, output_dim, input_dim).cuda()
    checkpoint = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Prototypes
    train_bundles = sorted([f for f in os.listdir(bundle_root) if f.endswith("_train.npz")])
    proto_bundle = os.path.join(bundle_root, train_bundles[0])
    print(f"\nBuilding BLIP prototypes from: {train_bundles[0]}")
    prototypes = build_class_prototypes(proto_bundle)

    # Test list
    test_list_path = os.path.join(eeg_root, f"EEG_{feature_type}", "test_list.txt")
    with open(test_list_path, "r") as f:
        test_lines = [line.strip() for line in f if line.strip()]

    class_groups = defaultdict(list)
    for rel_path in test_lines:
        parts = rel_path.split("/")
        class_token = next(p for p in parts if p.startswith("class"))
        class_id = int(class_token.replace("class", "").split("_")[0])
        class_groups[class_id].append(rel_path)

    chosen = [random.choice(class_groups[cid]) for cid in sorted(class_groups.keys()) if class_groups[cid]]
    print(f"\nEvaluating {len(chosen)} samples (1 per class)")

    correct_top1 = correct_top5 = total = 0
    per_class_results = {}

    for rel_path in chosen:
        eeg_path = os.path.join(eeg_root, f"EEG_{feature_type}", rel_path)
        parts = rel_path.split("/")
        class_token = next(p for p in parts if p.startswith("class"))
        true_class = int(class_token.replace("class", "").split("_")[0])

        eeg = np.load(eeg_path)

        if feature_type == "windows":
            eeg_flat = eeg.reshape(-1)
            eeg_scaled = scaler.transform([eeg_flat])[0]
            eeg_tensor = torch.tensor(eeg_scaled.reshape(7,62,100), dtype=torch.float32).unsqueeze(0).cuda()
        elif feature_type in ["DE","PSD"]:
            eeg_flat = eeg.reshape(-1)
            eeg_scaled = scaler.transform([eeg_flat])[0]
            eeg_tensor = torch.tensor(eeg_scaled.reshape(62,5), dtype=torch.float32).unsqueeze(0).cuda()
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")

        with torch.no_grad():
            pred_emb = model(eeg_tensor).squeeze(0).cpu().numpy()

        sims = {cid: cosine_similarity(torch.tensor(pred_emb), torch.tensor(proto)) for cid, proto in prototypes.items()}
        ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        top1_class = ranked[0][0]
        top5_classes = [cid for cid, _ in ranked[:5]]

        t1 = (top1_class == true_class)
        t5 = (true_class in top5_classes)
        per_class_results[true_class] = (t1, t5)
        correct_top1 += int(t1)
        correct_top5 += int(t5)
        total += 1

        print(f"{rel_path} | True={true_class} Pred@1={top1_class} Top-5={top5_classes} | {'Correct' if t1 else 'Wrong'} (Top-1)")

    acc_top1 = correct_top1 / total if total > 0 else 0.0
    acc_top5 = correct_top5 / total if total > 0 else 0.0
    print(f"\n=== Classification Accuracy ===")
    print(f"Top-1 Accuracy: {acc_top1:.4f} ({correct_top1}/{total})")
    print(f"Top-5 Accuracy: {acc_top5:.4f} ({correct_top5}/{total})")

    # Save log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"class_eval_{tag}_{timestamp}.txt")
    with open(log_file, "w") as f:
        f.write(f"Checkpoint: {ckpt_file}\n")
        f.write(f"Feature type: {feature_type}\n")
        f.write(f"Encoder: {encoder_type}\n")
        f.write(f"Prototype source: {train_bundles[0]}\n")
        f.write(f"Samples tested: {total}\n")
        f.write(f"Top-1 Accuracy: {acc_top1:.4f}\n")
        f.write(f"Top-5 Accuracy: {acc_top5:.4f}\n\n")
        f.write("=== Per-class results ===\n")
        for cid in sorted(per_class_results.keys()):
            t1, t5 = per_class_results[cid]
            f.write(f"Class {cid:02d}: Top-1={'Correct' if t1 else 'Wrong'}, Top-5={'Correct' if t5 else 'Wrong'}\n")

    print(f"\nResults saved to: {log_file}")
