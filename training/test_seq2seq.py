import os, sys, random, torch, imageio
import numpy as np
from einops import rearrange
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer

# === PATHS ===
drive_root = "/content/drive/MyDrive/EEG2Video_data/processed"
output_dir = "/content/drive/MyDrive/EEG2Video_outputs/test_seq2seq"
vae_path   = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4/vae"
caption_root = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text"

eeg_test_list = os.path.join(drive_root, "EEG_windows/test_list.txt")
vid_test_list = os.path.join(drive_root, "Video_latents/test_list_dup.txt")
text_test_list = os.path.join(caption_root, "test_list.txt")

os.makedirs(output_dir, exist_ok=True)

# === IMPORT YOUR MODEL ===
from seq2seq_model import Seq2SeqEEG2Video  # assumes you saved model definition here

# === LOAD MODEL ===
checkpoint_path = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_model.pt"
model = Seq2SeqEEG2Video().cuda()
ckpt = torch.load(checkpoint_path, map_location="cuda")
model.load_state_dict(ckpt["state_dict"])
model.eval()

# === LOAD VAE ===
vae = AutoencoderKL.from_pretrained(vae_path).cuda()
vae.eval()

# === PICK RANDOM TEST SAMPLE ===
with open(eeg_test_list) as f:
    eeg_files = [l.strip() for l in f]
with open(vid_test_list) as f:
    vid_files = [l.strip() for l in f]
with open(text_test_list) as f:
    txt_files = [os.path.join(caption_root, l.strip()) for l in f]

idx = random.randint(0, len(eeg_files)-1)
eeg_path = os.path.join(drive_root, "EEG_windows", eeg_files[idx])
vid_path = os.path.join(drive_root, "Video_latents", vid_files[idx])
txt_path = txt_files[idx % len(txt_files)]  # align index to caption list

# === LOAD DATA ===
eeg = np.load(eeg_path)   # [7,62,100]
video = np.load(vid_path) # [F,4,36,64]

with open(txt_path, "r") as f:
    caption_text = f.read().strip()

print("Chosen EEG:", eeg_path)
print("Chosen video:", vid_path)
print("Chosen caption file:", txt_path)
print("Caption text:", caption_text)

# === PREP TENSORS ===
eeg = torch.tensor(eeg, dtype=torch.float32).unsqueeze(0).cuda()   # [1,7,62,100]
video = torch.tensor(video, dtype=torch.float32).unsqueeze(0).cuda() # [1,F,4,36,64]

# === PREDICTED VIDEO ===
b, f, c, h, w = video.shape
zero_frame = torch.zeros((b,1,c,h,w), device=video.device)
full_video = torch.cat([zero_frame, video], dim=1)

with torch.no_grad():
    pred = model(eeg, full_video)[:, :-1]  # [1,F,4,36,64]

# === RANDOM BASELINE ===
rand_latent = torch.randn_like(video)

# === SHUFFLED BASELINE ===
shuffle_idx = random.randint(0, len(vid_files)-1)
while shuffle_idx == idx:
    shuffle_idx = random.randint(0, len(vid_files)-1)
shuffle_latent = np.load(os.path.join(drive_root, "Video_latents", vid_files[shuffle_idx]))
shuffle_latent = torch.tensor(shuffle_latent, dtype=torch.float32).unsqueeze(0).cuda()

# === METRICS ===
def mse_ssim(pred, target):
    pred_np = pred.detach().cpu().numpy().reshape(-1)
    tgt_np  = target.detach().cpu().numpy().reshape(-1)
    mse_val = mean_squared_error(tgt_np, pred_np)
    ssim_val = ssim(
        target.detach().cpu().numpy()[0,0].transpose(1,2,0),
        pred.detach().cpu().numpy()[0,0].transpose(1,2,0),
        channel_axis=-1, data_range=1.0
    )
    return mse_val, ssim_val

print("\n--- METRICS ---")
print("Model vs GT:", mse_ssim(pred, video))
print("Random vs GT:", mse_ssim(rand_latent, video))
print("Shuffled vs GT:", mse_ssim(shuffle_latent, video))

# === DECODE & SAVE MP4s ===
def decode_and_save(latents, name):
    # latents: [1,F,4,36,64]
    latents = latents.squeeze(0) / 0.18215
    latents = rearrange(latents, "f c h w -> f c h w")
    with torch.no_grad():
        frames = []
        for i in range(latents.shape[0]):
            frame = vae.decode(latents[i:i+1]).sample
            frame = (frame.clamp(-1,1)+1)/2
            frame = (frame * 255).cpu().numpy().astype(np.uint8)
            frame = rearrange(frame, "b c h w -> h w c b")[...,0]
            frames.append(frame)
    frames = np.stack(frames)
    mp4_name = os.path.splitext(os.path.basename(vid_path))[0] + f"_{frames.shape[0]}f_{name}.mp4"
    mp4_path = os.path.join(output_dir, mp4_name)
    writer = imageio.get_writer(mp4_path, fps=24, codec="libx264")
    for f in frames:
        writer.append_data(f)
    writer.close()
    print("Saved:", mp4_path)

decode_and_save(video, "groundtruth")
decode_and_save(pred, "predicted")
decode_and_save(rand_latent, "random")
decode_and_save(shuffle_latent, "shuffled")

print("\nFinal caption used:", caption_text)
