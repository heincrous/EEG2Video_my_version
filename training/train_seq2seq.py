import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Import Seq2Seq model (keep __init__.py for this style)
from models_original.seq2seq import Seq2SeqModel


# Dataset: EEG features + per-clip video latents
class SeedDVEEGDataset(torch.utils.data.Dataset):
    def __init__(self, eeg_file, latent_root):
        super().__init__()
        # Load one subject's EEG features (7,40,5,62,5)
        self.eeg_data = np.load(eeg_file)
        self.latent_root = latent_root

        # Build index map, only keep entries that have a latent file
        self.index_map = []
        for b in range(7):
            for c in range(40):
                for i in range(5):
                    latent_path = os.path.join(
                        self.latent_root,
                        f"block{b+1:02d}",
                        f"class{c:02d}",
                        f"clip{i}.npy"
                    )
                    if os.path.exists(latent_path):
                        self.index_map.append((b, c, i))

        print(f"Found {len(self.index_map)} aligned EEG-video pairs")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        b, c, i = self.index_map[idx]

        # EEG clip: keep as (5,62,5) instead of flattening
        eeg_clip = self.eeg_data[b, c, i]  # shape (5,62,5)

        # Video latents: (6,4,36,64) → flatten to (6,9216)
        latent_path = os.path.join(
            self.latent_root,
            f"block{b+1:02d}",
            f"class{c:02d}",
            f"clip{i}.npy"
        )
        latents = np.load(latent_path).reshape(6, -1)

        return torch.from_numpy(eeg_clip).float(), torch.from_numpy(latents).float()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "/content/drive/MyDrive/EEG2Video_checkpoints/"
    os.makedirs(save_dir, exist_ok=True)

    # Model
    model = Seq2SeqModel().to(device)

    # Dataset & loader
    eeg_file = "/content/drive/MyDrive/Data/Processed/EEG_DE_1per2s/sub1.npy"
    latent_root = "/content/drive/MyDrive/Data/Processed/Video_latents_per_clip/"
    dataset = SeedDVEEGDataset(eeg_file, latent_root)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Loss & optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(2):  # quick test run
        for eeg, target in dataloader:
            eeg, target = eeg.to(device), target.to(device)  # eeg: [B,5,62,5], target: [B,6,9216]

            optimizer.zero_grad()
            output = model(eeg, target)   # model must return [B,6,9216]

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(save_dir, "seq2seq_subset.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved at {ckpt_path}")


if __name__ == "__main__":
    main()
