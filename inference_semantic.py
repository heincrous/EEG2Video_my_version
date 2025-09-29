# ==========================================
# Inference: EEG → BLIP semantic embeddings
# ==========================================
import os, torch 
# import joblib
import numpy as np
from einops import rearrange
from sklearn.preprocessing import StandardScaler
from train_semantic_predictor import SemanticPredictor, FEATURE_PATHS, run_device


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

# feature types = everything between "semantic_predictor" and the subject tag
if subject_tag:
    subj_idx = parts.index(subject_tag)
    feature_types = parts[2:subj_idx]
else:
    feature_types = parts[2:-1] if "subset" in parts[-1] else parts[2:]

# class subset (if any)
if "subset" in ckpt_file:
    subset_str = ckpt_file.split("subset")[1].replace(".pt","")
    class_subset = [int(x) for x in subset_str.split("-")]
else:
    class_subset = None

print("Feature types:", feature_types)
print("Subject:", subject_tag)
print("Class subset:", class_subset)


# ==========================================
# 2. Load scalers
# ==========================================
# scalers = {}
# for ft in feature_types:
#     scaler_file = f"scaler_{ft}_{subject_tag}"
#     if class_subset is not None:
#         scaler_file += "_subset" + "-".join(str(c) for c in class_subset)
#     scaler_file += ".pkl"
#     scaler_path = os.path.join(SEMANTIC_CKPT_DIR, scaler_file)
#     scalers[ft] = joblib.load(scaler_path)
#     print("Loaded scaler:", scaler_file)


# ==========================================
# 3. Load EEG features (block 7 = test set)
# ==========================================
def load_features(subname, ft):
    path = os.path.join(FEATURE_PATHS[ft], subname)
    arr = np.load(path)
    if ft in ["DE","PSD"]:
        arr = arr.reshape(-1, 62*5)
    elif ft == "segments":
        arr = rearrange(arr, "a b c d (w t) -> (a b c w) (d t)", w=2, t=200)
    return arr

samples_per_block = (len(class_subset) if class_subset else 40) * 5 * 2
test_idx = np.arange(6 * samples_per_block, 7 * samples_per_block)

# features_test = []
# for ft in feature_types:
#     arr = load_features(subject_tag + ".npy", ft)
#     arr = scalers[ft].transform(arr[test_idx].reshape(len(test_idx), -1))
#     features_test.append(arr)
# X_test = np.concatenate(features_test, axis=1)

features_test = []
for ft in feature_types:
    arr = load_features(subject_tag + ".npy", ft)
    arr = arr[test_idx]  # take only test block
    flat = arr.reshape(len(test_idx), -1)
    scaler = StandardScaler().fit(flat)  # fit on test block itself
    arr = scaler.transform(flat)
    features_test.append(arr)
X_test = np.concatenate(features_test, axis=1)


# ==========================================
# 4. Build model
# ==========================================
input_dim = X_test.shape[1]
model = SemanticPredictor(input_dim).to(run_device)
model.load_state_dict(state_dict, strict=False)
model.eval()


# ==========================================
# 5. Run inference
# ==========================================
with torch.no_grad():
    eeg_tensor = torch.tensor(X_test, dtype=torch.float32).to(run_device)
    preds = model(eeg_tensor).cpu().numpy()

# reshape to (N,77,768)
preds = preds.reshape(-1, 77, 768)


# ==========================================
# 6. Save outputs
# ==========================================
base_name = ckpt_file.replace(".pt","")
out_path  = os.path.join(OUTPUT_DIR, f"embeddings_{base_name}.npy")
np.save(out_path, preds)
print("Saved semantic embeddings to:", out_path)
print("Shape:", preds.shape)
