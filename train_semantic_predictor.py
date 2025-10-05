# ==========================================
# EEG → CLIP Semantic Predictor
# (All EEG×All CLIP Training + Single-Pass Inference + Cleanup)
# ==========================================
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import preprocessing
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
import os, glob, random


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
        self.len = eeg.shape[0]
    def __len__(self):
        return self.len
    def __getitem__(self, item):
        return self.eeg[item], self.text[item]


# ==========================================
# Label order (authors)
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
# Utility setup
# ==========================================
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(114514)
device = 'cuda:0'


def cleanup():
    ckpt_dir = '/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints'
    out_dir  = '/content/drive/MyDrive/EEG2Video_outputs/semantic_embeddings'
    for d in [ckpt_dir, out_dir]:
        os.makedirs(d, exist_ok=True)
        for f in glob.glob(os.path.join(d, '*allpairs*')) + glob.glob(os.path.join(d, '*classlevel*')):
            os.remove(f)
            print(f"🧹 Deleted old file: {f}")
    return ckpt_dir, out_dir


# ==========================================
# Main
# ==========================================
if __name__ == '__main__':
    ckpt_path, out_dir = cleanup()

    eegdata = np.load('/content/drive/MyDrive/EEG2Video_data/processed/DE_1per2s_authors/sub1.npy')
    clipdata = np.load('/content/drive/MyDrive/EEG2Video_data/processed/CLIP_embeddings_authors/CLIP_embeddings_full.npy')

    print("EEG data shape:", eegdata.shape)
    print("CLIP embedding shape:", clipdata.shape)

    # ==========================================
    # Build training pairs (all EEG×all CLIP per class per block)
    # ==========================================
    EEG_list, Text_list = [], []
    for i in range(6):  # training blocks
        indices = [list(GT_label[i]).index(e) for e in chosed_label]
        eeg_block = eegdata[i][indices]       # (10, 5, 62, 5)
        clip_block = clipdata[i][indices]     # (10, 5, 77, 768)

        eeg_pairs, clip_pairs = [], []
        for c in range(len(indices)):
            eeg_c = eeg_block[c]
            clip_c = clip_block[c]
            eeg_c = np.repeat(eeg_c, 5, axis=0)     # repeat each EEG 5×
            clip_c = np.tile(clip_c, (5, 1, 1))     # all CLIPs
            eeg_pairs.append(eeg_c)
            clip_pairs.append(clip_c)

        eeg_pairs = np.concatenate(eeg_pairs, 0)
        clip_pairs = np.concatenate(clip_pairs, 0)

        eeg_pairs = torch.from_numpy(eeg_pairs)
        clip_pairs = torch.from_numpy(clip_pairs)
        eeg_pairs = rearrange(eeg_pairs, 'a b c -> a (b c)')
        clip_pairs = rearrange(clip_pairs, 'a b c -> a (b c)')

        EEG_list.append(eeg_pairs)
        Text_list.append(clip_pairs)

    EEG = torch.cat(EEG_list, 0)
    Text = torch.cat(Text_list, 0)
    print(f"Training EEG shape: {EEG.shape}, Text shape: {Text.shape}")

    # Normalization
    scaler = preprocessing.StandardScaler()
    scaler.fit(EEG)
    EEG = torch.from_numpy(scaler.transform(EEG)).float()

    # ==========================================
    # Model + Training
    # ==========================================
    model = CLIP().to(device)
    dataset = Dataset(EEG, Text)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200 * len(dataloader))

    for epoch in tqdm(range(50)):
        model.train()
        epoch_loss = 0
        for eeg_batch, text_batch in dataloader:
            eeg_batch = eeg_batch.to(device)
            text_batch = text_batch.float().to(device)
            optimizer.zero_grad()
            pred = model(eeg_batch)
            loss = F.mse_loss(pred, text_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.6f}")

    ckpt_file = os.path.join(ckpt_path, 'semantic_predictor_allpairs_sub1.pt')
    torch.save({'state_dict': model.state_dict()}, ckpt_file)
    print("✅ Model saved to:", ckpt_file)


    # ==========================================
    # Inference (7th block: one per EEG)
    # ==========================================
    print("\n=== Running inference on test block (7th) ===")
    test_indices = [list(GT_label[6]).index(e) for e in chosed_label]
    eeg_test = eegdata[6][test_indices]   # (10, 5, 62, 5)
    eeg_test = rearrange(torch.from_numpy(eeg_test), 'a b c d -> (a b) (c d)')  # 50×310
    eeg_test = torch.from_numpy(scaler.transform(eeg_test)).float().to(device)

    model.eval()
    with torch.no_grad():
        preds = model(eeg_test).cpu().numpy()

    preds = preds.reshape(10 * 5, 77, 768)  # (50, 77, 768)
    out_file = os.path.join(out_dir, 'pred_embeddings_sub1_allpairs.npy')
    np.save(out_file, preds)
    print(f"✅ Saved predicted embeddings: {preds.shape}")
