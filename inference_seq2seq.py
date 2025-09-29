# ==========================================
# Inference: EEG → Video Latents (Seq2Seq)
# ==========================================
import os
# import joblib
import numpy as np
import torch
from einops import rearrange
from train_seq2seq import MyTransformer, EEG_DIR, SEQ2SEQ_CKPT_DIR, run_device


# ==========================================
# Paths
# ==========================================
OUTPUT_DIR = "/content/drive/MyDrive/EEG2Video_outputs/seq2seq_latents"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 1. List checkpoints
# ==========================================
ckpts = [f for f in os.listdir(SEQ2SEQ_CKPT_DIR) if f.endswith(".pt")]
print("Available Seq2Seq checkpoints:")
for i, f in enumerate(ckpts):
    print(f"[{i}] {f}")

choice = int(input("Select checkpoint index: "))
ckpt_file = ckpts[choice]
ckpt_path = os.path.join(SEQ2SEQ_CKPT_DIR, ckpt_file)

print("Loading checkpoint:", ckpt_path)
state_dict = torch.load(ckpt_path, map_location=run_device)["state_dict"]

# deduce subject + subset from filename
parts = ckpt_file.replace(".pt","").split("_")
subject_tag = next((p for p in parts if p.startswith("sub")), None)

if "subset" in ckpt_file:
    subset_str = ckpt_file.split("subset")[1].replace(".pt","")
    class_subset = [int(x) for x in subset_str.split("-")]
else:
    class_subset = None

print("Subject:", subject_tag)
print("Class subset:", class_subset)


# ==========================================
# 2. Load scaler
# ==========================================
scaler_file = f"scaler_{subject_tag}"
if class_subset is not None:
    scaler_file += "_subset" + "-".join(str(c) for c in class_subset)
scaler_file += ".pkl"

scaler_path = os.path.join(SEQ2SEQ_CKPT_DIR, scaler_file)
scaler = joblib.load(scaler_path)
print("Loaded scaler:", scaler_file)


# ==========================================
# 3. Load EEG (test block only)
# ==========================================
eeg_path = os.path.join(EEG_DIR, subject_tag + ".npy")
eegdata  = np.load(eeg_path)  # (7,40,5,7,62,100)
EEG = rearrange(eegdata, "g p d w c l -> (g p d) w c l")

labels_block = np.repeat(np.arange(40), 5)
labels_all   = np.tile(labels_block, 7)

if class_subset is not None:
    mask = np.isin(labels_all, class_subset)
    EEG, labels_all = EEG[mask], labels_all[mask]

samples_per_block = (len(class_subset) if class_subset else 40) * 5
test_idx = np.arange(6 * samples_per_block, 7 * samples_per_block)

EEG_test = EEG[test_idx]

# scale with saved scaler
b, w, c, l = EEG_test.shape
EEG_test_2d = EEG_test.reshape(b, -1)
EEG_test_scaled = scaler.transform(EEG_test_2d).reshape(b, w, c, l)
EEG_test = torch.from_numpy(EEG_test_scaled).float().to(run_device)


# ==========================================
# 4. Build model
# ==========================================
model = MyTransformer().to(run_device)
model.load_state_dict(state_dict, strict=True)
model.eval()


# ==========================================
# 5. Run inference
# ==========================================
all_latents = []
with torch.no_grad():
    for i in range(0, len(EEG_test), 16):
        eeg_batch = EEG_test[i:i+16]
        b = eeg_batch.shape[0]

        # FIX: use 7 dummy frames instead of 1
        padded_video = torch.zeros((b, 7, 4, 36, 64), device=run_device)

        latents = model(eeg_batch, padded_video)  # (B,7,4,36,64)
        latents = latents[:, 1:, :, :, :]         # drop dummy frame
        all_latents.append(latents.cpu().numpy())

all_latents = np.concatenate(all_latents, axis=0)
print("Predicted latents shape:", all_latents.shape)


# ==========================================
# 6. Save outputs
# ==========================================
base_name = ckpt_file.replace(".pt","")
out_path  = os.path.join(OUTPUT_DIR, f"latents_{base_name}.npy")
np.save(out_path, all_latents)
print("Saved latents to:", out_path)
print("Shape:", all_latents.shape)

