import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from decord import VideoReader, cpu
from diffusers import AutoencoderKL

# Import your Seq2Seq model (adjust if class name differs)
from my_autoregressive_transformer import Seq2SeqModel

# Paths
EEG_PATH = "/content/drive/MyDrive/Data/Raw/SEED-DV/EEG/sub1.npy"
VIDEO_PATH = "/content/drive/MyDrive/Data/Raw/SEED-DV/Video/1st_10min.mp4"
CKPT_DIR = "/content/drive/MyDrive/EEG2Video_checkpoints/"
os.makedirs(CKPT_DIR, exist_ok=True)

# Hyperparams
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 1
lr = 1e-4

# 1. Load EEG (block 1 only, slice for speed)
eeg_data = np.load(EEG_PATH)
block1 = eeg_data[0]
eeg_segment = torch.tensor(block1[:, :512], dtype=torch.float32).unsqueeze(0).to(device)

# 2. Load video frames (few frames from 1st_10min.mp4)
vr = VideoReader(VIDEO_PATH, ctx=cpu(0))
frames = [torch.tensor(vr[i].asnumpy()).permute(2,0,1).unsqueeze(0).float()/255.0 for i in range(0,12,2)]
video_tensor = torch.cat(frames).to(device)

# 3. Encode frames into latents with pretrained VAE
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
with torch.no_grad():
    video_tensor = torch.nn.functional.interpolate(video_tensor, (256,256))
    latents = vae.encode(video_tensor).latent_dist.sample() * 0.18215

print("EEG segment:", eeg_segment.shape)
print("Video tensor:", video_tensor.shape)
print("Latents:", latents.shape)

# 4. Init Seq2Seq
model = Seq2SeqModel()  # adjust args if needed
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 5. One tiny training loop
for epoch in range(epochs):
    optimizer.zero_grad()
    pred_latents = model(eeg_segment)
    loss = criterion(pred_latents, latents)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss {loss.item():.4f}")

# 6. Save checkpoint
save_path = os.path.join(CKPT_DIR, "seq2seq_sub1_block1.pt")
torch.save(model.state_dict(), save_path)
print(f"Checkpoint saved at {save_path}")

