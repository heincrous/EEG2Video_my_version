import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import Seq2Seq model from authors' code (wrapper defined in models_original/seq2seq/__init__.py)
from models_original.seq2seq import Seq2SeqModel


# Dummy dataset with correct shapes
class DummyEEGDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=100, frames=7, n_channels=62, timesteps=100):
        super().__init__()
        # EEG input: [B, 7, 62, 100]
        self.data = torch.randn(n_samples, frames, n_channels, timesteps)
        # Target latents: [B, 7, 4, 36, 64]
        self.targets = torch.randn(n_samples, frames, 4, 36, 64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "/content/drive/MyDrive/EEG2Video_checkpoints/"
    os.makedirs(save_dir, exist_ok=True)

    # Model
    model = Seq2SeqModel().to(device)

    # Dataset & loader
    dataset = DummyEEGDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Loss & optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(2):  # small for testing
        for eeg, target in dataloader:
            eeg, target = eeg.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(eeg, target)

            # If model returns a tuple, keep first element
            if isinstance(output, tuple):
                output = output[0]

            # Flatten target to match output shape
            loss = criterion(output, target.view(output.shape))
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(save_dir, "seq2seq_dummy.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved at {ckpt_path}")


if __name__ == "__main__":
    main()
