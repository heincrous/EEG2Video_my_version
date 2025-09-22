import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------------
# Semantic Predictor (must match training definition)
# -------------------------------------------------------------------------
class SemanticPredictor(nn.Module):
    def __init__(self):
        super(SemanticPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(310, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 77 * 768)
        )
    def forward(self, eeg):
        return self.mlp(eeg)

# -------------------------------------------------------------------------
# Dataset wrapper (for test set)
# -------------------------------------------------------------------------
class EEGTextDataset(Dataset):
    def __init__(self, eeg_list_path, text_list_path, base_dir, max_samples=None):
        self.eeg_base = os.path.join(base_dir, "EEG_features")
        self.text_base = os.path.join(base_dir, "BLIP_embeddings")

        with open(eeg_list_path, 'r') as f:
            self.eeg_files = [line.strip() for line in f.readlines()]
        with open(text_list_path, 'r') as f:
            self.text_files = [line.strip() for line in f.readlines()]

        assert len(self.eeg_files) == len(self.text_files), "Mismatch between EEG and text file counts"

        if max_samples is not None:
            self.eeg_files = self.eeg_files[:max_samples]
            self.text_files = self.text_files[:max_samples]

        # Fit scaler on EEG test data
        eeg_all = []
        for eeg_f in self.eeg_files:
            abs_eeg_path = os.path.join(self.eeg_base, eeg_f)
            eeg_all.append(np.load(abs_eeg_path).reshape(-1))
        eeg_all = np.vstack(eeg_all)
        self.scaler = StandardScaler().fit(eeg_all)

    def __len__(self):
        return len(self.eeg_files)

    def __getitem__(self, idx):
        eeg_path = os.path.join(self.eeg_base, self.eeg_files[idx])
        txt_path = os.path.join(self.text_base, self.text_files[idx])

        eeg = np.load(eeg_path).reshape(-1)
        txt = np.load(txt_path).reshape(-1)

        eeg = self.scaler.transform([eeg])[0]
        return torch.tensor(eeg, dtype=torch.float32), torch.tensor(txt, dtype=torch.float32)

# -------------------------------------------------------------------------
# Cosine similarity helper
# -------------------------------------------------------------------------
def cosine_similarity(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum().item()

# -------------------------------------------------------------------------
# Main evaluation
# -------------------------------------------------------------------------
if __name__ == "__main__":
    drive_root = "/content/drive/MyDrive/EEG2Video_data/processed"
    ckpt_path = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_predictor.pt"

    eeg_test_list  = os.path.join(drive_root, "EEG_features/test_list.txt")
    text_test_list = os.path.join(drive_root, "BLIP_embeddings/test_list_dup.txt")

    # Load dataset (limit samples for speed)
    test_dataset = EEGTextDataset(eeg_test_list, text_test_list, base_dir=drive_root, max_samples=50)

    # Load model + checkpoint
    model = SemanticPredictor().cuda()
    checkpoint = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Evaluate a few samples
    print("\n=== Final Evaluation ===")
    for idx in [0, 1, 2]:  # first 3 test samples
        eeg, txt = test_dataset[idx]
        eeg = eeg.unsqueeze(0).cuda()
        txt = txt.cuda()

        with torch.no_grad():
            pred = model(eeg).squeeze(0)

        mse = F.mse_loss(pred, txt).item()
        cos = cosine_similarity(pred, txt)

        print(f"Sample {idx}: MSE={mse:.6f}, Cosine={cos:.6f}")
