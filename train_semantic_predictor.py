# ==========================================
# EEG → CLIP Semantic Predictor
# (Mean EEG per class + All CLIPs per class + Cleanup + 10-epoch Evaluation + Full Inference)
# ==========================================
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import preprocessing
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import os, random, glob


# ==========================================
# Model
# ==========================================
class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
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


# ==========================================
# Dataset
# ==========================================
class Dataset:
    def __init__(self, eeg, text):
        self.eeg = eeg
        self.text = text
    def __len__(self):
        return len(self.eeg)
    def __getitem__(self, i):
        return self.eeg[i], self.text[i]


# ==========================================
# GT label (7×40)
# ==========================================
GT_label = np.array([
 [23,22,9,6,18,14,5,36,25,19,28,35,3,16,24,40,15,27,38,33,34,4,39,17,1,26,20,29,13,32,37,2,11,12,30,31,8,21,7,10],
 [27,33,22,28,31,12,38,4,18,17,35,39,40,5,24,32,15,13,2,16,34,25,19,30,23,3,8,29,7,20,11,14,37,6,21,1,10,36,26,9],
 [15,36,31,1,34,3,37,12,4,5,21,24,14,16,39,20,28,29,18,32,2,27,8,19,13,10,30,40,17,26,11,9,33,25,35,7,38,22,23,6],
 [16,28,23,1,39,10,35,14,19,27,37,31,5,18,11,25,29,13,20,24,7,34,26,4,40,12,8,22,21,30,17,2,38,9,3,36,33,6,32,15],
 [18,29,7,35,22,19,12,36,8,15,28,1,34,23,20,13,37,9,16,30,2,33,27,21,14,38,10,17,31,3,24,39,11,32,4,25,40,5,26,6],
 [29,16,1,22,34,39,24,10,8,35,27,31,23,17,2,15,25,40,3,36,26,6,14,37,9,12,19,30,5,28,32,4,13,18,21,20,7,11,33,38],
 [38,34,40,10,28,7,1,37,22,9,16,5,12,36,20,30,6,15,35,2,31,26,18,24,8,3,23,19,14,13,21,4,25,11,32,17,39,29,33,27]
])
chosed_label = [1, 10, 12, 16, 19, 23, 25, 31, 34, 39]


# ==========================================
# Utility
# ==========================================
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def cleanup_old_files(ckpt_dir, embed_dir, subset_tag):
    print("Checking for existing files to delete...")
    ckpt_pattern = os.path.join(ckpt_dir, f"eeg2text_{subset_tag}.pt")
    embed_pattern = os.path.join(embed_dir, f"pred_embeddings_{subset_tag}*.npy")
    for path in glob.glob(ckpt_pattern):
        os.remove(path)
        print(f"🧹 Deleted checkpoint: {path}")
    for path in glob.glob(embed_pattern):
        os.remove(path)
        print(f"🧹 Deleted embedding: {path}")
    print("✅ Cleanup complete.\n")


def compute_cosine_metrics(preds, trues, n_classes=10, n_clips=5):
    preds = preds.reshape(n_classes, n_clips, -1)
    trues = trues.reshape(n_classes, n_clips, -1)
    preds_norm = preds / np.linalg.norm(preds, axis=2, keepdims=True)
    trues_norm = trues / np.linalg.norm(trues, axis=2, keepdims=True)
    mean_cos = np.mean([cosine_similarity(p, t).diagonal().mean() for p, t in zip(preds_norm, trues_norm)])
    within = np.mean([np.mean(cosine_similarity(preds_norm[c])) for c in range(n_classes)])
    flat = preds_norm.reshape(n_classes * n_clips, -1)
    cos = cosine_similarity(flat)
    labels = np.repeat(np.arange(n_classes), n_clips)
    between = cos[labels[:, None] != labels[None, :]].mean()
    return mean_cos, within, between


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    seed_everything(114514)
    device = "cuda:0"

    eeg_path  = "/content/drive/MyDrive/EEG2Video_data/processed/DE_1per2s_authors/sub1.npy"
    clip_path = "/content/drive/MyDrive/EEG2Video_data/processed/CLIP_embeddings_authors/CLIP_embeddings_full.npy"
    ckpt_dir  = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"
    embed_dir = "/content/drive/MyDrive/EEG2Video_outputs/semantic_embeddings"
    subset_tag = "sub1_subset10"

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(embed_dir, exist_ok=True)
    cleanup_old_files(ckpt_dir, embed_dir, subset_tag)

    eegdata = np.load(eeg_path)     # (7, 40, 5, 310)
    clipdata = np.load(clip_path)   # (7, 40, 5, 77, 768)

    # === Train set: average EEG per class, pair with all 5 CLIPs ===
    eeg, text = [], []
    for blk in range(6):
        for lbl in chosed_label:
            idx = np.where(GT_label[blk] == lbl)[0][0]
            eeg_mean = eegdata[blk, idx].mean(axis=0)
            for c in range(5):
                eeg.append(eeg_mean)
                text.append(clipdata[blk, idx, c].reshape(-1))
    eeg, text = np.array(eeg), np.array(text)
    scaler = preprocessing.StandardScaler()
    eeg = scaler.fit_transform(eeg)
    dataset = Dataset(eeg, text)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # === Test set: all EEG clips and CLIPs ===
    test_block = 6
    eeg_test, text_test = [], []
    for lbl in chosed_label:
        idx = np.where(GT_label[test_block] == lbl)[0][0]
        for c in range(5):
            eeg_test.append(eegdata[test_block, idx, c])
            text_test.append(clipdata[test_block, idx, c].reshape(-1))
    eeg_test, text_test = np.array(eeg_test), np.array(text_test)
    eeg_test = scaler.transform(eeg_test)

    # === Train ===
    model = CLIP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200 * len(dataloader))

    for epoch in tqdm(range(1, 51)):
        model.train()
        total_loss = 0
        for eeg_b, txt_b in dataloader:
            eeg_b, txt_b = eeg_b.float().to(device), txt_b.float().to(device)
            optimizer.zero_grad()
            out = model(eeg_b)
            loss = F.mse_loss(out, txt_b)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(eeg_test).float().to(device)).cpu().numpy()
            mean_cos, within, between = compute_cosine_metrics(preds, text_test)
            print(f"[Epoch {epoch}] Loss={total_loss/len(dataloader):.4f} | Cos={mean_cos:.4f} | Within={within:.4f} | Between={between:.4f}")

    # === Save model ===
    torch.save({'state_dict': model.state_dict()}, os.path.join(ckpt_dir, f"eeg2text_{subset_tag}.pt"))
    print(f"\n✅ Model saved to {ckpt_dir}")

    # === Save final predictions ===
    with torch.no_grad():
        preds = model(torch.tensor(eeg_test).float().to(device)).cpu().numpy()
    np.save(os.path.join(embed_dir, f"pred_embeddings_{subset_tag}.npy"), preds.reshape(len(chosed_label)*5, 77, 768))
    print(f"✅ Final embeddings saved to {embed_dir}")
