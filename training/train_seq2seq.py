import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Import Seq2Seq model
from models_original.seq2seq import Seq2SeqModel


# Dataset: EEG time-window features + per-clip video latents
class SeedDVEEGDataset(torch.utils.data.Dataset):
    def __init__(self, eeg_files, latent_root, max_blocks=7):
        super().__init__()
        self.eeg_data_list = [np.load(f) for f in eeg_files]  # one per subject
        self.latent_root = latent_root
        self.max_blocks = min(max_blocks, self.eeg_data_list[0].shape[0])

        # Build index map: (subject_idx, block, class, clip)
        self.index_map = []
        for s, eeg_data in enumerate(self.eeg_data_list):
            for b in range(self.max_blocks):    # restrict to available blocks
                for c in range(40):            # classes
                    for i in range(5):         # clips
                        latent_path = os.path.join(
                            self.latent_root,
                            f"block{b+1:02d}",
                            f"class{c:02d}",
                            f"clip{i}.npy"
                        )
                        if os.path.exists(latent_path):
                            self.index_map.append((s, b, c, i))

        print(f"Found {len(self.index_map)} aligned EEG-video pairs "
              f"(subjects={len(self.eeg_data_list)}, blocks={self.max_blocks})")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        s, b, c, i = self.index_map[idx]

        # EEG clip: shape (F,62,100), where F=4 windows per clip
        eeg_clip = self.eeg_data_list[s][b, c, i]  # [4,62,100]

        # Video latents: (6,4,36,64) → flatten to (6,9216)
        latent_path = os.path.join(
            self.latent_root,
            f"block{b+1:02d}",
            f"class{c:02d}",
            f"clip{i}.npy"
        )
        latents = np.load(latent_path).reshape(6, -1)  # [6,9216]

        return torch.from_numpy(eeg_clip).float(), torch.from_numpy(latents).float()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "/content/drive/MyDrive/EEG2Video_checkpoints/"
    os.makedirs(save_dir, exist_ok=True)

    # Model
    model = Seq2SeqModel().to(device)

    # Dataset & loader
    eeg_files = [
        f"/content/drive/MyDrive/Data/Processed/EEG_timewindows_100/sub{i}.npy"
        for i in range(1, 11)  # subjects 1–10
    ]
    latent_root = "/content/drive/MyDrive/Data/Processed/Video_latents_per_clip/"
    dataset = SeedDVEEGDataset(eeg_files, latent_root, max_blocks=4)  # only blocks 1–4
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    # Loss & optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(50):  # quick test run
        for eeg, target in dataloader:
            # eeg: [B,F,62,100], target: [B,6,9216]
            eeg, target = eeg.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(eeg, target)   # model must return [B,6,9216]

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(save_dir, "seq2seq_sub1to10_blocks1to4.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved at {ckpt_path}")


if __name__ == "__main__":
    main()
