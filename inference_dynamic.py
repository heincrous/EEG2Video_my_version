# ==========================================
# Inference: Dynamic Predictor → DANA latents
# ==========================================
import os, joblib
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from train_dynamic_predictor import make_encoder, FusionNet, FEATURE_PATHS, DYNPRED_CKPT_DIR, run_device
from inference_seq2seq import OUTPUT_DIR as SEQ2SEQ_OUT
from core.add_noise import Diffusion  # your DANA implementation


# ==========================================
# Paths
# ==========================================
OUTPUT_DIR = "/content/drive/MyDrive/EEG2Video_outputs/dana_latents"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 1. List checkpoints
# ==========================================
ckpts = [f for f in os.listdir(DYNPRED_CKPT_DIR) if f.endswith(".pt")]
print("Available dynamic predictor checkpoints:")
for i, f in enumerate(ckpts):
    print(f"[{i}] {f}")

choice = int(input("Select checkpoint index: "))
ckpt_file = ckpts[choice]
ckpt_path = os.path.join(DYNPRED_CKPT_DIR, ckpt_file)
print("Loading checkpoint:", ckpt_path)
state_dict = torch.load(ckpt_path, map_location=run_device)["state_dict"]

# parse filename: dynpredictor_DE_sub1_subset1-10-12...
parts = ckpt_file.replace(".pt","").split("_")
feature_types = parts[1:-1] if "subset" in parts[-1] else parts[1:-0]
subject_tag   = [p for p in parts if p.startswith("sub")][0]

if "subset" in ckpt_file:
    subset_str = ckpt_file.split("subset")[1].replace(".pt","")
    class_subset = [int(x) for x in subset_str.split("-")]
else:
    class_subset = None

print("Subject:", subject_tag)
print("Feature types:", feature_types)
print("Class subset:", class_subset)


# ==========================================
# 2. Load Seq2Seq latents
# ==========================================
latents_file = f"latents_seq2seq_{subject_tag}"
if class_subset:
    latents_file += "_subset" + "-".join(str(c) for c in class_subset)
latents_file += ".npy"

latents_path = os.path.join(SEQ2SEQ_OUT, latents_file)
latents = np.load(latents_path)  # (N,6,4,36,64)
print("Loaded Seq2Seq latents:", latents.shape)


# ==========================================
# 3. Build dynamic predictor
# ==========================================
if len(feature_types) > 1:
    encoders, emb_dims = {}, {}
    for ft in feature_types:
        enc, dim = make_encoder(ft, return_logits=False)
        encoders[ft] = enc
        emb_dims[ft] = dim
    model = FusionNet(encoders, emb_dims)
else:
    ft = feature_types[0]
    model = make_encoder(ft, return_logits=True)

model.to(run_device)
model.load_state_dict(state_dict, strict=True)
model.eval()


# ==========================================
# 4. Run prediction → OPS scores
# ==========================================
# For simplicity assume dynamic predictor outputs logits for test samples
# Collect probabilities of class=1
ops_scores = []
with torch.no_grad():
    # TODO: load EEG test features for this subject (block 7, scaled with correct scalers)
    # Example placeholder: X_test = torch.tensor(...).to(run_device)
    for batch in test_loader:
        X, _, _ = batch
        if isinstance(X, dict):
            X = {ft: X[ft].to(run_device) for ft in X}
        else:
            X = X.to(run_device)
        y_hat = model(X)
        probs = torch.softmax(y_hat, dim=-1)[:,1]  # OPS scores
        ops_scores.append(probs.cpu().numpy())
ops_scores = np.concatenate(ops_scores)
print("OPS scores:", ops_scores.shape)


# ==========================================
# 5. Apply DANA to Seq2Seq latents
# ==========================================
diffusion = Diffusion(time_steps=500)
noisy_latents = []
for i in range(latents.shape[0]):
    dyn_beta = 0.3 if ops_scores[i] >= 0.5 else 0.2
    lat = torch.from_numpy(latents[i:i+1]).float().to(run_device)
    out = diffusion.forward(lat, dynamic_beta=dyn_beta)
    noisy_latents.append(out.cpu().numpy())

noisy_latents = np.concatenate(noisy_latents, axis=0)
print("Noisy latents:", noisy_latents.shape)


# ==========================================
# 6. Save outputs
# ==========================================
base_name = ckpt_file.replace(".pt","")
out_path  = os.path.join(OUTPUT_DIR, f"latents_add_noise_{base_name}.npy")
np.save(out_path, noisy_latents)
print("Saved noisy latents:", out_path)
