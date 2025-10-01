# ==========================================
# Inference: EEG → BLIP semantic embeddings
# ==========================================
import os, torch
import numpy as np
from einops import rearrange
from sklearn.preprocessing import StandardScaler
from train_semantic_predictor import SemanticPredictor, FEATURE_PATHS, run_device

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
# 1. List checkpoints
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

# deduce subject + features from filename
parts = ckpt_file.replace(".pt","").split("_")

# subject tag
subject_tag = next((p for p in parts if p.startswith("sub")), None)

# feature types
if subject_tag:
    subj_idx = parts.index(subject_tag)
    feature_types = parts[2:subj_idx]
else:
    feature_types = parts[2:-1] if "subset" in parts[-1] else parts[2:]

# class subset
if "subset" in ckpt_file:
    subset_str = ckpt_file.split("subset")[1].replace(".pt","")
    class_subset = [int(x) for x in subset_str.split("-")]
else:
    class_subset = None

print("Feature types:", feature_types)
print("Subject:", subject_tag)
print("Class subset:", class_subset)

# ==========================================
# 2. Load EEG features (block 7 = test set)
# ==========================================
def load_features(subname, ft):
    path = os.path.join(FEATURE_PATHS[ft], subname)
    arr = np.load(path)  # (7,40,5,62,5) for DE/PSD (2s), (7,40,5,62,400) for segments

    if ft in ["DE", "PSD"]:
        # Already 1 sample per 2s → just flatten
        arr = arr.reshape(7, 40, 5, -1)       # (7,40,5,310)
        arr = arr.reshape(-1, arr.shape[-1])  # (1400,310)

    elif ft == "segments":
        arr = rearrange(arr, "a b c d (w t) -> (a b c w) (d t)", w=2, t=200)

    return arr

# === Test block indices (block 6 only) ===
samples_per_block = (len(class_subset) if class_subset else 40) * 5
test_idx = np.arange(6 * samples_per_block, 7 * samples_per_block)

features_test = []
for ft in feature_types:
    arr = load_features(subject_tag + ".npy", ft)
    arr = arr[test_idx]  # take only test block
    flat = arr.reshape(len(test_idx), -1)

    # scaling (ideally load training scaler; here we fit on test block)
    scaler = StandardScaler().fit(flat)
    arr = scaler.transform(flat)

    features_test.append(arr)

X_test = np.concatenate(features_test, axis=1)

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

    # reshape to (N,77,768)
    preds = preds.reshape(-1, 77, 768)

    # save
    base_name = ckpt_file.replace(".pt","")
    out_path  = os.path.join(OUTPUT_DIR, f"embeddings_{base_name}.npy")
    np.save(out_path, preds.astype(np.float32))
    print("Saved semantic embeddings to:", out_path)
    print("Shape:", preds.shape)

elif MODE == "negative":
    # Compute mean EEG feature from the test set
    X_neg = X_test.mean(axis=0, keepdims=True)  # shape (1, feat_dim)

    with torch.no_grad():
        neg_pred = model(torch.tensor(X_neg, dtype=torch.float32).to(run_device))
        neg_pred = neg_pred.cpu().numpy().reshape(1, 77, 768)

    # save
    neg_tag = ckpt_file.replace(".pt", "") + "_negative"
    out_path = os.path.join(OUTPUT_DIR, f"{neg_tag}.npy")
    np.save(out_path, neg_pred.astype(np.float32))
    print("Saved NEGATIVE semantic embedding to:", out_path)
    print("Shape:", neg_pred.shape)
    

# ==========================================
# 5. Save outputs
# ==========================================
base_name = ckpt_file.replace(".pt","")
out_path  = os.path.join(OUTPUT_DIR, f"embeddings_{base_name}.npy")
np.save(out_path, preds)
print("Saved semantic embeddings to:", out_path)
print("Shape:", preds.shape)
