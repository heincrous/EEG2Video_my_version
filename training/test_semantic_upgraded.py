# ==========================================
# Semantic Predictor Upgraded Evaluation (Prototype Classification, 1 Sample Per Class)
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
from train_semantic import SemanticPredictor


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
    keys = data["keys"]                        # ["BlockX/classYY_clipZZ.npy", ...]

    # Group embeddings by class
    class_groups = defaultdict(list)
    for i, key in enumerate(keys):
        class_id = int(key.split("/")[1].replace("class", ""))
        emb = blip_embeddings[i].reshape(-1)  # flatten
        class_groups[class_id].append(emb)

    # Average per class
    prototypes = {}
    for class_id, embs in class_groups.items():
        prototypes[class_id] = np.mean(np.stack(embs), axis=0)

    return prototypes


# ==========================================
# Main evaluation
# ==========================================
if __name__ == "__main__":
    # === Paths ===
    bundle_root = "/content/drive/MyDrive/EEG2Video_data/processed/SubjectBundles"
    eeg_root    = "/content/drive/MyDrive/EEG2Video_data/processed"
    ckpt_root   = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"
    log_dir     = "/content/drive/MyDrive/EEG2Video_outputs/semantic_eval"
    os.makedirs(log_dir, exist_ok=True)

    # === Checkpoint selection ===
    ckpts = [f for f in os.listdir(ckpt_root) if f.startswith("semantic_predictor_") and f.endswith(".pt")]
    print("\nAvailable checkpoints:")
    for i, ck in enumerate(ckpts):
        print(f"[{i}] {ck}")

    choice    = int(input("\nEnter checkpoint index: ").strip())
    ckpt_file = ckpts[choice]
    tag       = ckpt_file.replace("semantic_predictor_", "").replace(".pt", "")
    ckpt_path = os.path.join(ckpt_root, ckpt_file)
    scaler_path = os.path.join(ckpt_root, f"scaler_{tag}.pkl")

    # Detect feature type
    if tag.endswith("_DE"):
        feature_type = "DE"
    elif tag.endswith("_PSD"):
        feature_type = "PSD"
    elif tag.endswith("_windows"):
        feature_type = "windows"
    else:
        raise ValueError(f"Cannot infer feature type from checkpoint tag: {tag}")

    print(f"\nLoading checkpoint: {ckpt_file}")
    print(f"Loading scaler: scaler_{tag}.pkl")
    print(f"Detected feature type: {feature_type}")

    # === Load scaler ===
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # === Load model ===
    input_dim = scaler.mean_.shape[0]
    model = SemanticPredictor(input_dim).cuda()
    checkpoint = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # === Build prototypes from one training bundle ===
    train_bundles = sorted([f for f in os.listdir(bundle_root) if f.endswith("_train.npz")])
    if not train_bundles:
        raise RuntimeError("No training bundles found.")
    proto_bundle = os.path.join(bundle_root, train_bundles[0])
    print(f"\nBuilding BLIP prototypes from: {train_bundles[0]}")
    prototypes = build_class_prototypes(proto_bundle)

    # === Load test list ===
    test_list_path = os.path.join(eeg_root, f"EEG_{feature_type}", "test_list.txt")
    with open(test_list_path, "r") as f:
        test_lines = [line.strip() for line in f if line.strip()]

    # === Group test samples by class ===
    class_groups = defaultdict(list)
    for rel_path in test_lines:
        class_id = int(rel_path.split("/")[1].replace("class", "").split("_")[0])
        class_groups[class_id].append(rel_path)

    # === Select one sample per class (40 total) ===
    chosen = []
    for class_id in sorted(class_groups.keys()):
        if class_groups[class_id]:
            chosen.append(random.choice(class_groups[class_id]))

    print(f"\nEvaluating {len(chosen)} samples (1 per class)")

    correct = 0
    total = 0
    per_class_results = {}

    for rel_path in chosen:
        eeg_path = os.path.join(eeg_root, f"EEG_{feature_type}", rel_path)
        true_class = int(rel_path.split("/")[1].replace("class", "").split("_")[0])

        eeg = np.load(eeg_path).reshape(-1)
        eeg_scaled = scaler.transform([eeg])[0]
        eeg_tensor = torch.tensor(eeg_scaled, dtype=torch.float32).unsqueeze(0).cuda()

        with torch.no_grad():
            pred_emb = model(eeg_tensor).squeeze(0).cpu().numpy()

        # Compare to prototypes
        sims = {cid: cosine_similarity(torch.tensor(pred_emb), torch.tensor(proto)) for cid, proto in prototypes.items()}
        pred_class = max(sims, key=sims.get)

        correct_flag = (pred_class == true_class)
        per_class_results[true_class] = correct_flag

        if correct_flag:
            correct += 1
        total += 1

        print(f"{rel_path} | True={true_class} Pred={pred_class} | {'Correct' if correct_flag else 'Wrong'}")

    acc = correct / total if total > 0 else 0.0
    print(f"\n=== Classification Accuracy ===")
    print(f"Top-1 Accuracy: {acc:.4f} ({correct}/{total})")

    # === Save log ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"class_eval_{tag}_{timestamp}.txt")

    with open(log_file, "w") as f:
        f.write(f"Checkpoint: {ckpt_file}\n")
        f.write(f"Feature type: {feature_type}\n")
        f.write(f"Prototype source: {train_bundles[0]}\n")
        f.write(f"Samples tested: {total}\n")
        f.write(f"Top-1 Accuracy: {acc:.4f}\n\n")

        f.write("=== Per-class results ===\n")
        for cid in sorted(per_class_results.keys()):
            res = "Correct" if per_class_results[cid] else "Wrong"
            f.write(f"Class {cid:02d}: {res}\n")

    print(f"\nResults saved to: {log_file}")
