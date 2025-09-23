import os, sys, random, torch, imageio, joblib
import numpy as np
from einops import rearrange
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from diffusers import AutoencoderKL

# === PATHS ===
drive_root   = "/content/drive/MyDrive/EEG2Video_data/processed"
output_dir   = "/content/drive/MyDrive/EEG2Video_outputs/test_seq2seq"
vae_path     = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4/vae"
caption_root = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text"
latent_scaler_path = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_latent_scaler.pkl"
eeg_scaler_path    = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_eeg_scaler.pkl"

eeg_test_list  = os.path.join(drive_root, "EEG_windows/test_list.txt")
vid_test_list  = os.path.join(drive_root, "Video_latents/test_list_dup.txt")
text_test_list = os.path.join(caption_root, "test_list.txt")

os.makedirs(output_dir, exist_ok=True)

# === IMPORT MODEL ===
from train_seq2seq import MyTransformer
checkpoint_path = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_checkpoint.pt"
model = MyTransformer(pred_frames=24).cuda()
ckpt = torch.load(checkpoint_path, map_location="cuda")
model.load_state_dict(ckpt["state_dict"])
model.eval()

# === LOAD VAE ===
vae = AutoencoderKL.from_pretrained(vae_path).cuda()
vae.eval()

# === LOAD SCALERS ===
latent_scaler = joblib.load(latent_scaler_path)
eeg_scaler    = joblib.load(eeg_scaler_path)

# === LOAD TEST LISTS ===
with open(eeg_test_list) as f:
    eeg_files = [l.strip() for l in f]
with open(vid_test_list) as f:
    vid_files = [l.strip() for l in f]
with open(text_test_list) as f:
    txt_files = [os.path.join(caption_root, l.strip()) for l in f]

# === Pick 2 EEGs for collapse sanity check ===
idx1, idx2 = random.sample(range(len(eeg_files)), 2)
eeg1 = np.load(os.path.join(drive_root, "EEG_windows", eeg_files[idx1]))
eeg2 = np.load(os.path.join(drive_root, "EEG_windows", eeg_files[idx2]))

# Normalize with EEG scaler
b, c, t = eeg1.shape
eeg1 = eeg_scaler.transform(eeg1.reshape(-1, t)).reshape(b, c, t)
b, c, t = eeg2.shape
eeg2 = eeg_scaler.transform(eeg2.reshape(-1, t)).reshape(b, c, t)

eeg1 = torch.tensor(eeg1, dtype=torch.float32).unsqueeze(0).cuda()
eeg2 = torch.tensor(eeg2, dtype=torch.float32).unsqueeze(0).cuda()

# === Inference (autoregressive forward) ===
with torch.no_grad():
    pred1 = model(eeg1, torch.zeros((1,24,4,36,64), device="cuda"))
    pred2 = model(eeg2, torch.zeros((1,24,4,36,64), device="cuda"))

print("MSE between two predicted latents:", F.mse_loss(pred1, pred2).item())

# === Continue as before with one EEG for decoding ===
idx = idx1
eeg_path = os.path.join(drive_root, "EEG_windows", eeg_files[idx])
vid_path = os.path.join(drive_root, "Video_latents", vid_files[idx])
txt_path = txt_files[idx % len(txt_files)]

video = np.load(vid_path) # [24,4,36,64]
with open(txt_path, "r") as f:
    caption_text = f.read().strip()

video = torch.tensor(video, dtype=torch.float32).unsqueeze(0).cuda()

with torch.no_grad():
    pred = model(eeg1, torch.zeros((1,24,4,36,64), device="cuda"))

print("\nFinal caption used:", caption_text)
