import os
import torch
import numpy as np
from train_semantic_predictor import SemanticPredictor

# ---------------- Paths ----------------
SEMANTIC_CKPT = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_predictor.pt"
TEST_EEG_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/Split_4train1test/test/EEG_features/DE_1per2s"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Load model ----------------
model = SemanticPredictor(input_dim=310).to(device)
state_dict = torch.load(SEMANTIC_CKPT, map_location=device)["state_dict"]
model.load_state_dict(state_dict)
model.eval()
print("Semantic predictor loaded successfully.")

# ---------------- Sample EEG ----------------
# Pick one subject/block/clip
subj = sorted(os.listdir(TEST_EEG_DIR))[0]
block_dir = os.path.join(TEST_EEG_DIR, subj)
block = sorted(os.listdir(block_dir))[0]
eeg_file = os.path.join(block_dir, block, sorted(os.listdir(os.path.join(block_dir, block)))[0])

eeg = np.load(eeg_file).reshape(-1)  # shape (310,)
eeg_tensor = torch.tensor(eeg, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 310]

# ---------------- Forward pass ----------------
with torch.no_grad():
    embedding = model(eeg_tensor)

print("Embedding shape:", embedding.shape)  # should be [1, 77*768] = [1, 59136]
