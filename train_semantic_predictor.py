# ==========================================
# EEG → CLIP Semantic Predictor (Self-Aligning)
# ==========================================
import os, random, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from einops import rearrange
from tqdm import tqdm


# ==========================================
# Model
# ==========================================
class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(310, 10000),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(10000, 77 * 768),
        )

    def forward(self, eeg):
        out = self.mlp(eeg)
        out = F.normalize(out, dim=-1)  # stay on CLIP hypersphere
        return out


# ==========================================
# Dataset
# ==========================================
class Dataset:
    def __init__(self, eeg, text):
        self.eeg = eeg
        self.text = text

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        return self.eeg[idx], self.text[idx]


# ==========================================
# Config
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"

GT_label = np.array([
 [23,22,9,6,18,14,5,36,25,19,28,35,3,16,24,40,15,27,38,33,34,4,39,17,1,26,20,29,13,32,37,2,11,12,30,31,8,21,7,10],
 [27,33,22,28,31,12,38,4,18,17,35,39,40,5,24,32,15,13,2,16,34,25,19,30,23,3,8,29,7,20,11,14,37,6,21,1,10,36,26,9],
 [15,36,31,1,34,3,37,12,4,5,21,24,14,16,39,20,28,29,18,32,2,27,8,19,13,10,30,40,17,26,11,9,33,25,35,7,38,22,23,6],
 [16,28,23,1,39,10,35,14,19,27,37,31,5,18,11,25,29,13,20,24,7,34,26,4,40,12,8,22,21,30,17,2,38,9,3,36,33,6,32,15],
 [18,29,7,35,22,19,12,36,8,15,28,1,34,23,20,13,37,9,16,30,2,33,27,21,14,38,10,17,31,3,24,39,11,32,4,25,40,5,26,6],
 [29,16,1,22,34,39,24,10,8,35,27,31,23,17,2,15,25,40,3,36,26,6,14,37,9,12,19,30,5,28,32,4,13,18,21,20,7,11,33,38],
 [38,34,40,10,28,7,1,37,22,9,16,5,12,36,20,30,6,15,35,2,31,26,18,24,8,3,23,19,14,13,21,4,25,11,32,17,39,29,33,27]
])
chosed_label = [1, 3, 5, 11, 12, 13, 23, 27, 30, 38]


# ==========================================
# Seed + Cleanup
# ==========================================
def seed_everything(seed=114514):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_everything()

ckpt_dir = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"
emb_dir  = "/content/drive/MyDrive/EEG2Video_outputs/semantic_embeddings"
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(emb_dir, exist_ok=True)

for d in [ckpt_dir, emb_dir]:
    for f in os.listdir(d):
        if "sub1_subset10" in f or "eeg2text_10_classes" in f:
            os.remove(os.path.join(d, f))
            print(f"🧹 Deleted: {f}")


# ==========================================
# Load EEG and CLIP
# ==========================================
eegdata = np.load("/content/drive/MyDrive/EEG2Video_data/processed/DE_1per2s_authors/sub1.npy")
clip_embeddings = np.load("/content/drive/MyDrive/EEG2Video_data/processed/CLIP_embeddings_authors/CLIP_embeddings_full.npy")
print("EEG:", eegdata.shape, "CLIP:", clip_embeddings.shape)

EEG, Text = [], []
for blk in range(6):
    eeg_idx = [list(GT_label[blk]).index(lbl) for lbl in chosed_label]
    EEG.append(eegdata[blk][eeg_idx])
    text_blk = torch.from_numpy(clip_embeddings[blk])
    text_idx = [list(GT_label[blk]).index(lbl) for lbl in chosed_label]
    text_blk = text_blk[text_idx, :][:, ::5].repeat_interleave(5, dim=1)
    text_blk = text_blk.reshape(len(chosed_label), 5, -1)
    Text.append(text_blk)

EEG = np.stack(EEG, axis=0)
EEG = rearrange(EEG, "a b c e f -> (a b c) (e f)")
Text = torch.cat(Text, dim=0).reshape(-1, Text[0].shape[-1])
print("Train EEG:", EEG.shape, "Train Text:", Text.shape)

# Normalize EEG
scaler = preprocessing.StandardScaler()
scaler.fit(EEG)
EEG = scaler.transform(EEG)

dataset = Dataset(EEG, Text)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ==========================================
# Setup model + optimizer
# ==========================================
model = CLIP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200 * len(loader))

# ==========================================
# Helper functions
# ==========================================
def compute_cosine(preds, trues):
    preds = preds / np.linalg.norm(preds, axis=1, keepdims=True)
    trues = trues / np.linalg.norm(trues, axis=1, keepdims=True)
    return np.mean(np.diag(cosine_similarity(preds, trues)))

def class_cosine(preds, trues):
    n_class, n_clip = 10, 5
    preds = preds.reshape(n_class, n_clip, -1)
    trues = trues.reshape(n_class, n_clip, -1)
    within, between = [], []
    for i in range(n_class):
        cos_within = cosine_similarity(preds[i], preds[i]).mean()
        for j in range(n_class):
            if i != j:
                cos_between = cosine_similarity(preds[i], preds[j]).mean()
                between.append(cos_between)
        within.append(cos_within)
    return np.mean(within), np.mean(between)


# ==========================================
# Training
# ==========================================
num_epochs = 200
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    for eeg_batch, text_batch in loader:
        eeg_batch = eeg_batch.float().to(device)
        text_batch = text_batch.float().to(device)
        text_batch = F.normalize(text_batch, dim=-1)

        optimizer.zero_grad()
        pred = model(eeg_batch)
        cos_loss = 1 - F.cosine_similarity(pred, text_batch, dim=1).mean()
        mse_loss = F.mse_loss(pred, text_batch)
        loss = mse_loss + 0.1 * cos_loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    # evaluate on training batch subset every 10 epochs
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            eeg_test = torch.tensor(EEG).float().to(device)
            preds = model(eeg_test).cpu().numpy()
            trues = Text.numpy()
            mean_cos = compute_cosine(preds, trues)
            w, b = class_cosine(preds, trues)
        print(f"[Epoch {epoch:03d}] loss={total_loss:.4f} | cos={mean_cos:.4f} | within={w:.4f} | between={b:.4f}")

# ==========================================
# Save model and embeddings
# ==========================================
ckpt_path = os.path.join(ckpt_dir, "eeg2text_10_classes.pt")
torch.save({"state_dict": model.state_dict()}, ckpt_path)
print(f"✅ Model saved to {ckpt_path}")

# final inference embeddings
model.eval()
with torch.no_grad():
    eeg_test = torch.tensor(EEG).float().to(device)
    preds = model(eeg_test).cpu().numpy().reshape(len(chosed_label) * 6 * 5, 77, 768)

emb_path = os.path.join(emb_dir, "pred_embeddings_sub1_subset10.npy")
np.save(emb_path, preds)
print(f"✅ Final embeddings saved to {emb_path}")
