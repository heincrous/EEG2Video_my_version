import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu
from diffusers import AutoencoderKL

# ----------------------------------------------------------------
# Paths
EEG_PATH = "/content/drive/MyDrive/Data/Raw/EEG/sub1.npy"
VIDEO_PATH = "/content/drive/MyDrive/Data/Raw/Video/1st_10min.mp4"
CKPT_DIR = "/content/drive/MyDrive/EEG2Video_checkpoints/"
os.makedirs(CKPT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Dataset
class EEGVideoDataset(Dataset):
    def __init__(self, eeg_path, video_path, num_samples=100, frames_per_clip=6):
        self.eeg_data = np.load(eeg_path)  # shape (7, 62, 104000)
        self.block = self.eeg_data[0]      # just block 1 for now
        self.video = VideoReader(video_path, ctx=cpu(0))
        self.num_samples = num_samples
        self.frames_per_clip = frames_per_clip

        # Preload VAE for frame → latents
        self.vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae"
        ).to(device)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # EEG: random 512-sample segment
        start = np.random.randint(0, self.block.shape[1] - 512)
        eeg_segment = self.block[:, start:start+512]
        eeg_tensor = torch.tensor(eeg_segment, dtype=torch.float32)

        # Video: evenly spaced frames
        frame_ids = np.linspace(0, len(self.video)-1, self.frames_per_clip, dtype=int)
        frames = [
            torch.tensor(self.video[i].asnumpy()).permute(2,0,1).float()/255.0
            for i in frame_ids
        ]
        video_tensor = torch.stack(frames).to(device)

        with torch.no_grad():
            video_tensor = torch.nn.functional.interpolate(video_tensor, (256,256))
            latents = self.vae.encode(video_tensor).latent_dist.sample() * 0.18215

        return eeg_tensor, latents
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Seq2Seq model
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
# ----------------------------------------------------------------

if __name__ == "__main__":
    # Dataset and dataloader
    dataset = EEGVideoDataset(EEG_PATH, VIDEO_PATH, num_samples=50, frames_per_clip=6)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Model, optimizer, loss
    model = EEG2Latent().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(2):  # run a couple of epochs for testing
        for step, (eeg, latents) in enumerate(loader):
            eeg = eeg.to(device)              # [B,62,512]
            latents = latents.to(device)      # [B,F,4,32,32]
            target = latents.view(latents.size(0)*latents.size(1), -1)  # [B*F,4096]
            eeg = eeg.unsqueeze(1).repeat(1, latents.size(1), 1, 1)     # [B,F,62,512]
            eeg = eeg.view(-1, 62, 512)                                # [B*F,62,512]

            optimizer.zero_grad()
            pred = model(eeg)                  # [B*F,4096]
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            if step % 5 == 0:
                print(f"Epoch {epoch} Step {step} | Loss: {loss.item():.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(CKPT_DIR, "seq2seq_demo.pt")
    torch.save(model.state_dict(), ckpt_path)
    print("Checkpoint saved to", ckpt_path)
