# ==========================================
# Inference: EEG → BLIP semantic embeddings
# ==========================================
import os, torch
import numpy as np
from einops import rearrange
from sklearn.preprocessing import StandardScaler
from train_semantic_predictor import SemanticPredictor, FEATURE_PATHS, run_device, CLASS_SUBSET, subject_name, FEATURE_TYPES

# ==========================================
# Config
# ==========================================
MODE = "negative"   # options: "predict", "negative"

# ==========================================
# Paths
# ==========================================
SEMANTIC_CKPT_DIR = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"
OUTPUT_DIR        = "/content/drive/MyDrive/EEG2Video_outputs/semantic_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. Select checkpoint
# ==========================================
ckpts = [f for f in os.listdir(SEMANTIC_CKPT_DIR) if f.endswith(".pt")]
print("Available semantic predictor checkpoints:")
for i, f in enumerate(ckpts):
    print(f"[{i}] {f}")

choice = int(input("Select checkpoint index: "))
ckpt_file = ckpts[choice]
ckpt_path = os.path.join(SEMANTIC_CKPT_DIR, ckpt_file)

print("Loading checkpoint:", ckpt_path)
ckpt = torch.load(ckpt_path, map_location=run_device)
state_dict = ckpt["state_dict"]

print("Feature types:", FEATURE_TYPES)
print("Subject:", subject_name)
print("Class subset:", CLASS_SUBSET)

# ==========================================
# 2. Load EEG features (train/test split like training)
# ==========================================
def load_features(subname, ft):
    path = os.path.join(FEATURE_PATHS[ft], subname)
    arr = np.load(path)  # (7,40,5,62,5) or (7,40,5,62,400)

    if ft in ["DE", "PSD"]:
        arr = arr.reshape(7, 40, 5, -1)       # (7,40,5,310)
        arr = arr.reshape(-1, arr.shape[-1])  # (1400,310)
    elif ft == "segments":
        arr = rearrange(arr, "a b c d (w t) -> (a b c w) (d t)", w=2, t=200)

    return arr

features_all = []
for ft in FEATURE_TYPES:
    arr = load_features(subject_name, ft)

    # truncate to valid length (7 blocks × 40 classes × 5 clips)
    valid_len = 40 * 5 * 7
    arr = arr[:valid_len]

    # build label array
    labels_block = np.repeat(np.arange(40), 5)   # 200 per block
    labels_all   = np.tile(labels_block, 7)      # 1400

    # apply class subset mask if needed
    if CLASS_SUBSET is not None:
        mask = np.isin(labels_all, CLASS_SUBSET)
        arr, labels_all = arr[mask], labels_all[mask]

    # flatten to 2D
    arr = arr.reshape(arr.shape[0], -1)
    features_all.append(arr)

# concatenate feature types (one feature type → no effect)
features_all = np.concatenate(features_all, axis=1)

# fit scaler on all samples (train+test, after masking)
scaler = StandardScaler().fit(features_all)
features_all = scaler.transform(features_all)

# compute block split AFTER masking
samples_per_block = (len(CLASS_SUBSET) if CLASS_SUBSET else 40) * 5
test_idx = np.arange(6 * samples_per_block, 7 * samples_per_block)
X_test = features_all[test_idx]

# ==========================================
# 3. Build model
# ==========================================
input_dim = X_test.shape[1]
model = SemanticPredictor(input_dim).to(run_device)
model.load_state_dict(state_dict)
model.eval()

# ==========================================
# 4. Run inference or negative
# ==========================================
if MODE == "predict":
    with torch.no_grad():
        eeg_tensor = torch.tensor(X_test, dtype=torch.float32).to(run_device)
        preds = model(eeg_tensor).cpu().numpy()

    preds = preds.reshape(-1, 77, 768)
    base_name = ckpt_file.replace(".pt", "")
    out_path  = os.path.join(OUTPUT_DIR, f"embeddings_{base_name}.npy")
    np.save(out_path, preds.astype(np.float32))
    print("Saved semantic embeddings to:", out_path)
    print("Shape:", preds.shape)

elif MODE == "negative":
    X_neg = X_test.mean(axis=0, keepdims=True)
    with torch.no_grad():
        neg_pred = model(torch.tensor(X_neg, dtype=torch.float32).to(run_device))
        neg_pred = neg_pred.cpu().numpy().reshape(1, 77, 768)

    neg_tag = ckpt_file.replace(".pt", "") + "_negative"
    out_path = os.path.join(OUTPUT_DIR, f"{neg_tag}.npy")
    np.save(out_path, neg_pred.astype(np.float32))
    print("Saved NEGATIVE semantic embedding to:", out_path)
    print("Shape:", neg_pred.shape)
