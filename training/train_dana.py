import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------------------
# Paths
EEG_PATH = "/content/drive/MyDrive/Data/Raw/EEG/sub1.npy"
CKPT_DIR = "/content/drive/MyDrive/EEG2Video_checkpoints/"
os.makedirs(CKPT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Dataset (EEG -> semantic embedding stub)
class EEGSemanticDataset(Dataset):
    def __init__(self, eeg_path, num_samples=100):
        self.eeg_data = np.load(eeg_path)  # shape (7, 62, 104000)
        self.block = self.eeg_data[0]      # just block 1 for now
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # EEG: random 512-sample segment
        start = np.random.randint(0, self.block.shape[1] - 512)
        eeg_segment = self.block[:, start:start+512]
        eeg_tensor = torch.tensor(eeg_segment, dtype=torch.float32)

        # Semantic embedding (stub): [77,768]
        target = torch.randn(77, 768, dtype=torch.float32)

        return eeg_tensor, target
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Seq2Seq model
class EEG2Semantic(nn.Module):
    def __init__(self, eeg_dim=62*512, embed_dim=77*768, hidden=512, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(eeg_dim, hidden)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(hidden, embed_dim)

    def forward(self, x):
        # x: [B, C, T]
        b, c, t = x.shape
        x = x.view(b, -1)           # [B, C*T]
        x = self.embed(x).unsqueeze(0)  # [1,B,H]
        h = self.transformer(x)     # [1,B,H]
        out = self.out(h).squeeze(0)  # [B, embed_dim]
        return out.view(b, 77, 768)   # reshape to [B,77,768]
# ----------------------------------------------------------------

if __name__ == "__main__":
    # Dataset and dataloader
    dataset = EEGSemanticDataset(EEG_PATH, num_samples=50)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Model, optimizer, loss
    model = EEG2Semantic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(2):  # run a couple of epochs for testing
        for step, (eeg, target) in enumerate(loader):
            eeg = eeg.to(device)        # [B,62,512]
            target = target.to(device)  # [B,77,768]

            optimizer.zero_grad()
            pred = model(eeg)           # [B,77,768]
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            if step % 5 == 0:
                print(f"Epoch {epoch} Step {step} | Loss: {loss.item():.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(CKPT_DIR, "seq2seq_semantic.pt")
    torch.save(model.state_dict(), ckpt_path)
    print("Checkpoint saved to", ckpt_path)
