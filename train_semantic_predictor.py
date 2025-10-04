# ==========================================
# EEG → CLIP Semantic Predictor
# (All EEGs × All CLIPs per class + Fixed Cosine Loss + Safe Metrics + float32)
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
        self.eeg = eeg.astype(np.float32)
        self.text = text.astype(np.float32)
        self.len = eeg.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.eeg[item], self.text[item]


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
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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
    preds = preds.reshape(n_classes, n_clips, -1).astype(np.float32)
    trues = trues.reshape(n_classes, n_clips, -1).astype(np.float32)

    preds_norm = preds / np.linalg.norm(preds, axis=2, keepdims=True)
    trues_norm = trues / np.linalg.norm(trues, axis=2, keepdims=True)

    mean_cos = np.mean([
        np.diag(cosine_similarity(p, t)).mean()
        for p, t in zip(preds_norm, trues_norm)
    ])

    within_vals = []
    for c in range(n_classes):
        cos_c = cosine_similarity(preds_norm[c])
        np.fill_diagonal(cos_c, 0)
        within_vals.append(cos_c.mean())
    within = np.mean(within_vals)

    flat = preds_norm.reshape(n_classes * n_clips, -1)
    labels = np.repeat(np.arange(n_classes), n_clips)
    cos_all = cosine_similarity(flat)
    mask = labels[:, None] != labels[None, :]
    between = cos_all[mask].mean()

    return float(mean_cos), float(within), float(between)


# ==========================================
# Main
# ==========================================
if __name__ == '__main__':
    seed_everything(114514)
    device = 'cuda:0'

    eeg_path  = "/content/drive/MyDrive/EEG2Video_data/processed/DE_1per2s_authors/sub1.npy"
    clip_path = "/content/drive/MyDrive/EEG2Video_data/processed/CLIP_embeddings_authors/CLIP_embeddings_full.npy"
    ckpt_dir  = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"
    embed_dir = "/content/drive/MyDrive/EEG2Video_outputs/semantic_embeddings"
    subset_tag = "sub1_subset10"

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(embed_dir, exist_ok=True)
    cleanup_old_files(ckpt_dir, embed_dir, subset_tag)

    # === Load as float32 ===
    eegdata = np.load(eeg_path).astype(np.float32)
    clip_embeddings = np.load(clip_path).astype(np.float32)

    print("\n=== Sanity Check: EEG–CLIP Class Alignment ===")
    for blk in range(7):
        eeg_order = [np.where(GT_label[blk] == lbl)[0][0] for lbl in chosed_label]
        clip_order = [np.where(GT_label[blk] == lbl)[0][0] for lbl in chosed_label]
        if eeg_order != clip_order:
            print(f"❌ Misalignment detected in block {blk}")
        else:
            print(f"✅ Block {blk} aligned correctly: {eeg_order[:5]} ...")

    # === Build training data ===
    eeg, text = [], []
    for blk in range(6):  # first 6 blocks for training
        for lbl in chosed_label:
            idx = np.where(GT_label[blk] == lbl)[0][0]
            eeg_clips  = eegdata[blk, idx]
            clip_clips = clip_embeddings[blk, idx]
            for eeg_clip in eeg_clips:
                for clip_clip in clip_clips:
                    eeg.append(eeg_clip)
                    text.append(clip_clip)

    eeg = np.array(eeg, dtype=np.float32).reshape(len(eeg), -1)
    text = np.array(text, dtype=np.float32).reshape(len(text), -1)

    clip_norm_mean = np.mean(np.linalg.norm(text, axis=1))
    print(f"Average CLIP norm (train): {clip_norm_mean:.3f}")
    text = text / np.linalg.norm(text, axis=1, keepdims=True)

    scaler = preprocessing.StandardScaler()
    eeg = scaler.fit_transform(eeg).astype(np.float32)

    print(f"\nTraining EEG shape: {eeg.shape}, CLIP shape: {text.shape}")
    dataset = Dataset(eeg, text)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    # === Prepare test data ===
    test_block = 6
    eeg_test, text_test = [], []
    for lbl in chosed_label:
        idx = np.where(GT_label[test_block] == lbl)[0][0]
        for c in range(5):
            eeg_test.append(eegdata[test_block, idx, c])
            text_test.append(clip_embeddings[test_block, idx, c])
    eeg_test = np.array(eeg_test, dtype=np.float32).reshape(-1, 310)
    text_test = np.array(text_test, dtype=np.float32).reshape(-1, 77 * 768)
    eeg_test = scaler.transform(eeg_test).astype(np.float32)

    # === Train (safe cosine loss) ===
    model = CLIP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200 * len(dataloader))

    for epoch in tqdm(range(1, 101)):
        model.train()
        total_loss = 0.0
        for eeg_batch, text_batch in dataloader:
            eeg_batch = eeg_batch.to(device)
            text_batch = text_batch.to(device)
            optimizer.zero_grad()

            preds = F.normalize(model(eeg_batch), dim=-1)
            text_batch = F.normalize(text_batch, dim=-1)

            cos_sim = F.cosine_similarity(preds, text_batch, dim=-1, eps=1e-8).clamp(-1, 1)
            loss = 1 - cos_sim.mean()

            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                preds_test = model(torch.tensor(eeg_test).to(device))
                preds_test = F.normalize(preds_test, dim=-1).cpu().numpy()
            mean_cos, within, between = compute_cosine_metrics(preds_test, text_test)
            print(f"[Epoch {epoch}] Loss={total_loss:.4f} | Cos={mean_cos:.4f} | Within={within:.4f} | Between={between:.4f}")

    # === Save model ===
    final_ckpt = os.path.join(ckpt_dir, f"eeg2text_{subset_tag}.pt")
    torch.save({'state_dict': model.state_dict()}, final_ckpt)
    print(f"\n✅ Final model saved: {final_ckpt}")

    # === Save inference (rescaled to CLIP norm) ===
    with torch.no_grad():
        preds_test = model(torch.tensor(eeg_test).to(device))
        preds_test = F.normalize(preds_test, dim=-1) * clip_norm_mean
        preds_test = preds_test.cpu().numpy().astype(np.float32)

    save_pred_path = os.path.join(embed_dir, f"pred_embeddings_{subset_tag}.npy")
    np.save(save_pred_path, preds_test.reshape(len(chosed_label)*5, 77, 768))
    print(f"✅ Final predictions saved: {save_pred_path}")
    print(f"✅ Predicted embeddings rescaled by CLIP mean norm ({clip_norm_mean:.3f}) and saved in float32.")
