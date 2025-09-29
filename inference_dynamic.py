# ==========================================
# Inference: Dynamic Predictor → DANA latents
# ==========================================
import os, numpy as np, torch
from einops import rearrange
from sklearn.preprocessing import StandardScaler
from train_dynamic_predictor import make_encoder, DYNPRED_CKPT_DIR, FEATURE_PATHS, run_device
from core.add_noise import Diffusion

# ==========================================
# Paths
# ==========================================
SEQ2SEQ_OUT   = "/content/drive/MyDrive/EEG2Video_outputs/seq2seq_latents"
OUTPUT_DIR    = "/content/drive/MyDrive/EEG2Video_outputs/dana_latents"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. Select checkpoint
# ==========================================
ckpts = [f for f in os.listdir(DYNPRED_CKPT_DIR) if f.endswith(".pt")]
print("Available dynamic predictor checkpoints:")
for i, f in enumerate(ckpts):
    print(f"[{i}] {f}")
choice    = int(input("Select checkpoint index: "))
ckpt_file = ckpts[choice]
ckpt_path = os.path.join(DYNPRED_CKPT_DIR, ckpt_file)
print("Loading checkpoint:", ckpt_path)

state_dict = torch.load(ckpt_path, map_location=run_device)["state_dict"]

# deduce subject + subset
parts       = ckpt_file.replace(".pt","").split("_")
subject_tag = next((p for p in parts if p.startswith("sub")), None)
if "subset" in ckpt_file:
    subset_str   = ckpt_file.split("subset")[1].replace(".pt","")
    class_subset = [int(x) for x in subset_str.split("-")]
else:
    class_subset = None

print("Subject:", subject_tag)
print("Class subset:", class_subset)

# ==========================================
# 2. Load EEG windows (test block only)
# ==========================================
win_path = os.path.join(FEATURE_PATHS["windows"], subject_tag + ".npy")
windows  = np.load(win_path)  # (7,40,5,3,62,200)

# flatten to (N,62,200)
windows  = rearrange(windows, "g p d w c t -> (g p d w) c t")
labels   = np.tile(np.arange(40).repeat(5), 7)
labels   = np.repeat(labels, 3)  # 3 windows per clip

if class_subset is not None:
    mask   = np.isin(labels, class_subset)
    windows, labels = windows[mask], labels[mask]

samples_per_block = (len(class_subset) if class_subset else 40) * 5 * 3
test_idx = np.arange(6*samples_per_block, 7*samples_per_block)

X_test   = windows[test_idx]
y_test   = labels[test_idx]

# scale
scaler   = StandardScaler().fit(X_test.reshape(len(X_test), -1))
X_test   = scaler.transform(X_test.reshape(len(X_test), -1)).reshape(-1,1,62,200)
X_test   = torch.tensor(X_test, dtype=torch.float32).to(run_device)

# ==========================================
# 3. Build model
# ==========================================
model = make_encoder("windows", return_logits=True).to(run_device)
model.load_state_dict(state_dict, strict=True)
model.eval()

# ==========================================
# 4. Run inference (logits → clip-level)
# ==========================================
with torch.no_grad():
    logits = []
    for i in range(0, len(X_test), 64):
        batch = X_test[i:i+64]
        out   = model(batch)  # (B,2)
        logits.append(out.cpu())
logits = torch.cat(logits, dim=0)  # (N,2)

# average 3 windows per clip
num_clips    = logits.shape[0] // 3
logits_clip  = logits.view(num_clips, 3, -1).mean(dim=1)
print("Clip-level logits:", logits_clip.shape)

# ==========================================
# 5. Decide dynamic beta per clip
# ==========================================
probs    = torch.softmax(logits_clip, dim=-1)[:,1]  # probability of "fast"
betas    = torch.where(probs > 0.5, 0.3, 0.2)       # example mapping

# ==========================================
# 6. Load Seq2Seq latents
# ==========================================
lat_files = [f for f in os.listdir(SEQ2SEQ_OUT) if subject_tag in f]
print("Available seq2seq latents:")
for i, f in enumerate(lat_files):
    print(f"[{i}] {f}")
lat_choice  = int(input("Select latents index: "))
lat_path    = os.path.join(SEQ2SEQ_OUT, lat_files[lat_choice])
latents     = np.load(lat_path)  # (num_clips, 6,4,36,64)
latents     = torch.tensor(latents, dtype=torch.float32).to(run_device)

# ==========================================
# 7. Add noise using DANA
# ==========================================
diffusion = Diffusion(time_steps=500)
out_all   = []
for i in range(num_clips):
    out = diffusion.forward(latents[i:i+1], betas[i].item())
    out_all.append(out.cpu())
out_all = torch.cat(out_all, dim=0)

# ==========================================
# 8. Save outputs
# ==========================================
base_name = ckpt_file.replace(".pt","")
out_path  = os.path.join(OUTPUT_DIR, f"dana_{base_name}.pt")
torch.save(out_all, out_path)
print("Saved noisy latents to:", out_path, out_all.shape)
