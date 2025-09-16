import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from decord import VideoReader, cpu
from diffusers import AutoencoderKL

# Paths
EEG_PATH = "/content/drive/MyDrive/Data/Raw/EEG/sub1.npy"
VIDEO_PATH = "/content/drive/MyDrive/Data/Raw/Video/1st_10min.mp4"
CKPT_DIR = "/content/drive/MyDrive/EEG2Video_checkpoints/"
os.makedirs(CKPT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load EEG (block 1 only, first 512 samples)
eeg_data = np.load(EEG_PATH)  # shape (7, 62, 104000)
block1 = eeg_data[0]
eeg_segment = torch.tensor(block1[:, :512], dtype=torch.float32).unsqueeze(0).to(device)  # [1,62,512]

# 2. Load video (grab first few frames)
vr = VideoReader(VIDEO_PATH, ctx=cpu(0))
frames = [torch.tensor(vr[i].asnumpy()).permute(2,0,1).unsqueeze(0).float()/255.0 for i in range(0,12,2)]
video_tensor = torch.cat(frames).to(device)  # [6,3,H,W]

# 3. Encode frames into latents with pretrained VAE
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
with torch.no_grad():
    video_tensor = torch.nn.functional.interpolate(video_tensor, (256,256))
    latents = vae.encode(video_tensor).latent_dist.sample() * 0.18215  # [6,4,32,32]

# --- Seq2Seq Model (minimal) ---
class EEG2Latent(nn.Module):
    def __init__(self, eeg_dim=62*512, latent_dim=4*32*32, hidden=512, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(eeg_dim, hidden)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(hidden, latent_dim)

    def forward(self, x):
        # x: [B, C, T]
        b, c, t = x.shape
        x = x.view(b, -1)  # flatten [B, C*T]
        x = self.embed(x).unsqueeze(0)  # [1,B,H]
        h = self.transformer(x)  # [1,B,H]
        out = self.out(h).squeeze(0)  # [B,latent_dim]
        return out

model = EEG2Latent().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# 4. Prepare target latents
target = latents.view(latents.size(0), -1)  # [6,4096]

# 5. Training loop (tiny demo: few steps)
for step in range(10):
    optimizer.zero_grad()
    pred = model(eeg_segment.repeat(target.size(0),1,1))  # repeat EEG per frame
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()
    print(f"Step {step} | Loss: {loss.item():.4f}")

# 6. Save checkpoint
torch.save(model.state_dict(), os.path.join(CKPT_DIR, "seq2seq_demo.pt"))
print("Checkpoint saved to", CKPT_DIR)
