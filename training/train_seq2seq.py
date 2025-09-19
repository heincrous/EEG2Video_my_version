import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Import our fixed wrapper (Seq2SeqModel returns 9216 latents instead of 13 classes)
from models_original.seq2seq import Seq2SeqModel


# Dataset: EEG features + video latents
class SeedDVEEGDataset(torch.utils.data.Dataset):
    def __init__(self, eeg_dir, latent_dir):
        super().__init__()
        self.eeg_files = sorted([os.path.join(eeg_dir, f) for f in os.listdir(eeg_dir) if f.endswith(".npy")])
        self.latent_files = sorted([os.path.join(latent_dir, f) for f in os.listdir(latent_dir) if f.endswith(".npy")])
        assert len(self.eeg_files) == len(self.latent_files), "Mismatch between EEG and latents"

    def __len__(self):
        return len(self.eeg_files)

    def __getitem__(self, idx):
        # Load EEG DE features: shape (7,40,5,62,5)
        eeg = np.load(self.eeg_files[idx])
        # Load video latents: shape (7,4,36,64)
        latents = np.load(self.latent_files[idx])

        # Reshape EEG: collapse blocks × concepts × clips → sequence length
        # Each timestep has 62 channels × 5 bands = 310 features
        eeg = eeg.reshape(-1, 62 * 5)              # [1400, 310]

        # Reshape latents: each clip → flattened vector (9216 dims)
        latents = latents.reshape(-1, 9216)        # [7, 9216]

        # Convert to tensors
        eeg = torch.from_numpy(eeg).float()
        latents = torch.from_numpy(latents).float()

        return eeg, latents


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "/content/drive/MyDrive/EEG2Video_checkpoints/"
    os.makedirs(save_dir, exist_ok=True)

    # Model
    model = Seq2SeqModel().to(device)

    # Dataset & loader
    eeg_dir = "/content/drive/MyDrive/Data/Processed/EEG_DE_1per2s/"
    latent_dir = "/content/drive/MyDrive/Data/Processed/Video_latents/"
    dataset = SeedDVEEGDataset(eeg_dir, latent_dir)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # smaller batch for VRAM

    # Loss & optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(2):  # start small for testing
        for eeg, target in dataloader:
            eeg, target = eeg.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(eeg, target)   # expects [B, T, D]

            # Ensure target matches output shape
            target_flat = target.view(output.shape)

            loss = criterion(output, target_flat)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(save_dir, "seq2seq_real.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved at {ckpt_path}")


if __name__ == "__main__":
    main()
