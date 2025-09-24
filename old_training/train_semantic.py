import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle

# -------------------------------------------------------------------------
# Semantic Predictor (MLP)
# -------------------------------------------------------------------------
class SemanticPredictor(nn.Module):
    def __init__(self):
        super(SemanticPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(310, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 77 * 768)
        )

    def forward(self, eeg):
        return self.mlp(eeg)


# -------------------------------------------------------------------------
# Dataset wrapper
# -------------------------------------------------------------------------
class EEGTextDataset(Dataset):
    def __init__(self, npz_files, scaler=None, fit_scaler=False, max_samples=None):
        eeg_all, text_all = [], []

        for f in npz_files:
            data = np.load(f, allow_pickle=True)
            eeg = data["EEG_DE"]                 # (N, 62, 5)
            text = data["BLIP_embeddings"]       # (N, 77, 768)

            eeg = eeg.reshape(eeg.shape[0], -1)   # (N, 310)
            text = text.reshape(text.shape[0], -1) # (N, 77*768)

            eeg_all.append(eeg)
            text_all.append(text)

        self.eeg = np.vstack(eeg_all)
        self.text = np.vstack(text_all)

        if max_samples:
            self.eeg = self.eeg[:max_samples]
            self.text = self.text[:max_samples]

        if fit_scaler:
            self.scaler = StandardScaler().fit(self.eeg)
        else:
            self.scaler = scaler

        self.eeg = self.scaler.transform(self.eeg)

    def __len__(self):
        return self.eeg.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.eeg[idx], dtype=torch.float32),
            torch.tensor(self.text[idx], dtype=torch.float32),
        )


# -------------------------------------------------------------------------
# Training Loop
# -------------------------------------------------------------------------
if __name__ == "__main__":
    bundle_dir = "/content/drive/MyDrive/EEG2Video_data/processed/SubjectBundles"
    save_root = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_predictor"
    os.makedirs(save_root, exist_ok=True)

    all_bundles = sorted([f for f in os.listdir(bundle_dir) if f.endswith("_train.npz")])
    subjects = [f.replace("_train.npz", "") for f in all_bundles]

    print("\nAvailable subjects:")
    for idx, subj in enumerate(subjects):
        print(f"{idx}: {subj}")

    choice = input("\nEnter subject indices (comma separated), 'all', or 'check': ").strip()
    num_epochs = int(input("\nEnter number of epochs: ") or 50)

    # --- Dry run ---
    if choice.lower() == "check":
        test_file = os.path.join(bundle_dir, all_bundles[0])
        dataset = EEGTextDataset([test_file], fit_scaler=True, max_samples=10)
        eeg, txt = dataset[0]
        print("Dry run OK")
        print("EEG shape:", eeg.shape, "Text shape:", txt.shape)
        exit()

    # --- Select files ---
    if choice.lower() == "all":
        selected_files = [os.path.join(bundle_dir, f) for f in all_bundles]
        tag = "all"
    else:
        selected_idx = [int(c.strip()) for c in choice.split(",") if c.strip().isdigit()]
        selected_files = [os.path.join(bundle_dir, all_bundles[i]) for i in selected_idx]
        tag = "_".join([subjects[i] for i in selected_idx])

    # --- Build dataset ---
    dataset = EEGTextDataset(selected_files, fit_scaler=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

    # --- Model ---
    model = SemanticPredictor().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(dataloader))

    # --- Training ---
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for eeg, text in dataloader:
                eeg, text = eeg.cuda(non_blocking=True), text.cuda(non_blocking=True)

                optimizer.zero_grad()
                pred = model(eeg)
                loss = F.mse_loss(pred, text)
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})
                pbar.update(1)

        print(f"Epoch {epoch+1}: avg_loss={total_loss/len(dataloader):.6f}")

    # --- Save ---
    ckpt_path = os.path.join(save_root, f"semantic_predictor_{tag}.pt")
    scaler_path = os.path.join(save_root, f"scaler_{tag}.pkl")

    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(dataset.scaler, f)

    print(f"Model saved to: {ckpt_path}")
    print(f"Scaler saved to: {scaler_path}")