import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import preprocessing
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
from sklearn.metrics.pairwise import cosine_similarity
import os, random


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
class Dataset():
    def __init__(self, eeg, text):
        self.eeg = eeg
        self.text = text
        self.len = eeg.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.eeg[item], self.text[item]


# ==========================================
# Ground-truth labels
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
chosed_label = [1, 3, 5, 11, 12, 13, 23, 27, 30, 38]


# ==========================================
# Seed
# ==========================================
def seed_everything(seed=114514):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()
device = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# Cleanup old files
# ==========================================
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
# Data loading and alignment
# ==========================================
eegdata = np.load('/content/drive/MyDrive/EEG2Video_data/processed/DE_1per2s_authors/sub1.npy')
clip_embeddings = np.load('/content/drive/MyDrive/EEG2Video_data/processed/CLIP_embeddings_authors/CLIP_embeddings_full.npy')

eeg = []
Text = []

for blk in range(6):
    indices = [np.where(GT_label[blk] == lbl)[0][0] for lbl in chosed_label]
    chosed_eeg = eegdata[blk, indices]
    eeg.append(chosed_eeg)

    text = clip_embeddings[blk, indices]
    text = text.reshape(len(chosed_label) * 5, -1)
    Text.append(text)

eeg = np.stack(eeg, axis=0)
eeg = rearrange(eeg, 'a b c e f -> (a b c) (e f)')
Text = np.concatenate(Text, axis=0)

# Normalize EEG
normalize = preprocessing.StandardScaler()
normalize.fit(eeg)
eeg = normalize.transform(eeg)
print("Train EEG:", eeg.shape, "Train Text:", Text.shape)

dataset = Dataset(eeg, Text)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# ==========================================
# Training
# ==========================================
model = CLIP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200 * len(dataloader))
num_epochs = 200

# Prepare test data
test_block = 6
indices = [np.where(GT_label[test_block] == lbl)[0][0] for lbl in chosed_label]
eeg_test = eegdata[test_block, indices]
eeg_test = rearrange(eeg_test, 'b c e f -> (b c) (e f)')
eeg_test = normalize.transform(eeg_test)
text_test = clip_embeddings[test_block, indices].reshape(len(chosed_label) * 5, -1)
true_flat = text_test

def class_cosine(preds):
    n_class, n_clip = 10, 5
    preds = preds.reshape(n_class, n_clip, -1)
    within, between = [], []
    for i in range(n_class):
        cos_within = cosine_similarity(preds[i], preds[i]).mean()
        within.append(cos_within)
        for j in range(n_class):
            if i != j:
                cos_between = cosine_similarity(preds[i], preds[j]).mean()
                between.append(cos_between)
    return np.mean(within), np.mean(between)

for epoch in tqdm(range(1, num_epochs + 1)):
    model.train()
    epoch_loss = 0.0
    for eeg_batch, text_batch in dataloader:
        eeg_batch = eeg_batch.float().to(device)
        text_batch = text_batch.float().to(device)
        optimizer.zero_grad()
        preds = model(eeg_batch)
        loss = F.mse_loss(preds, text_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(eeg_test).float().to(device)).cpu().numpy()
        pred_norm = preds / np.linalg.norm(preds, axis=1, keepdims=True)
        true_norm = true_flat / np.linalg.norm(true_flat, axis=1, keepdims=True)
        mean_cos = np.mean(np.diag(cosine_similarity(pred_norm, true_norm)))
        within, between = class_cosine(pred_norm)
        print(f"[Epoch {epoch:03d}] Loss={epoch_loss:.4f} | EEG→CLIP={mean_cos:.4f} | Within={within:.4f} | Between={between:.4f} | Δ={within-between:.4f}")

# ==========================================
# Save final model and embeddings
# ==========================================
model.eval()
with torch.no_grad():
    preds = model(torch.tensor(eeg_test).float().to(device)).cpu().numpy()
preds = preds.reshape(len(chosed_label)*5, 77, 768)
np.save(os.path.join(emb_dir, "pred_embeddings_sub1_subset10.npy"), preds)
torch.save({'state_dict': model.state_dict()}, os.path.join(ckpt_dir, "eeg2text_10_classes.pt"))
print("✅ Training complete and outputs saved.")
