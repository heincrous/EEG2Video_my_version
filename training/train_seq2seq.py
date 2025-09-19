import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import Seq2Seq model (wrapper points to myTransformer)
from models_original.seq2seq import Seq2SeqModel

# Dummy dataset for classification (13 classes, like authors’ txtpredictor)
class DummyEEGDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=100, frames=7, n_channels=62, timesteps=100, n_classes=13):
        super().__init__()
        self.data = torch.randn(n_samples, frames, n_channels, timesteps)  # [B,7,62,100]
        self.targets = torch.randint(0, n_classes, (n_samples,))           # [B] class labels

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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(2):  # small for testing
        for eeg, target in dataloader:
            eeg, target = eeg.to(device), target.to(device)

            optimizer.zero_grad()
            # Pass eeg as src, and a dummy tensor as tgt to satisfy forward()
            dummy_tgt = torch.zeros_like(eeg)[:, :, :4, :4]  # small filler
            output = model(eeg, dummy_tgt)

            if isinstance(output, tuple):
                output = output[0]

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(save_dir, "seq2seq_dummy.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved at {ckpt_path}")

if __name__ == "__main__":
    main()