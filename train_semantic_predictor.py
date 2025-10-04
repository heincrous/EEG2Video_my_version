# ==========================================
# EEG → CLIP Semantic Predictor
# (Class-level CLIP text embeddings + Cleanup)
# ==========================================
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import preprocessing
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
from transformers import CLIPTokenizer, CLIPTextModel
import os, random, glob


# ==========================================
# Cleanup old checkpoints and embeddings
# ==========================================
def cleanup():
    ckpt_dir = '/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints'
    out_dir  = '/content/drive/MyDrive/EEG2Video_outputs/semantic_embeddings'
    for folder in [ckpt_dir, out_dir]:
        os.makedirs(folder, exist_ok=True)
        old_files = glob.glob(os.path.join(folder, '*classlevel*'))
        for f in old_files:
            os.remove(f)
            print(f"🧹 Deleted old file: {f}")
cleanup()


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
def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
seed_everything(114514)
device = 'cuda:0'


# ==========================================
# Class prompts → CLIP embeddings
# ==========================================
PRETRAINED_SD_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"

class_prompts = {
    1:  "a video of a cat",
    10: "a video of a shark swimming in the sea",
    12: "a video of flowers blooming",
    16: "a person dancing on stage",
    19: "a close-up of a human face",
    23: "a city skyline with tall buildings",
    25: "cars driving on a road",
    31: "a pizza being cooked in an oven",
    34: "a person playing guitar",
    39: "an airplane flying in the sky"
}

tokenizer = CLIPTokenizer.from_pretrained(PRETRAINED_SD_PATH, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(PRETRAINED_SD_PATH, subfolder="text_encoder").to(device)
text_encoder.eval()

class_embs = []
for lbl in chosed_label:
    tokens = tokenizer([class_prompts[lbl]], padding="max_length", max_length=77, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = text_encoder(tokens.input_ids)[0].cpu().numpy()  # (1,77,768)
    class_embs.append(emb)
class_embs = np.concatenate(class_embs, axis=0)  # (10,77,768)
print("Created CLIP text embeddings:", class_embs.shape)


# ==========================================
# Main
# ==========================================
if __name__ == '__main__':
    eegdata = np.load('/content/drive/MyDrive/EEG2Video_data/processed/DE_1per2s_authors/sub1.npy')
    print("EEG data shape:", eegdata.shape)

    eeg = []
    for i in range(6):  # first 6 blocks for training
        indices = [list(GT_label[i]).index(element) for element in chosed_label]
        eeg.append(eegdata[i][indices, :])
    eeg = np.stack(eeg, axis=0)
    eeg = torch.from_numpy(eeg)
    eeg = rearrange(eeg, 'a b c d e -> (a b c) (d e)')

    # Build class-level CLIP text embedding targets
    Text = []
    for _ in range(6):  # same number of blocks
        text = np.repeat(class_embs[:, None, :, :], repeats=5, axis=1)
        text = torch.from_numpy(text)
        text = rearrange(text, 'a b c d -> (a b) (c d)')
        Text.append(text)
    Text = torch.cat(Text, dim=0)
    print("Target text shape:", Text.shape)

    # Normalization
    normalize = preprocessing.StandardScaler()
    normalize.fit(eeg)
    eeg = normalize.transform(eeg)
    print("EEG normalized shape:", eeg.shape)

    dataset = Dataset(eeg, Text)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CLIP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25 * len(dataloader))

    # ==========================================
    # Training
    # ==========================================
    for epoch in tqdm(range(25)):
        model.train()
        epoch_loss = 0
        for eeg_batch, text_batch in dataloader:
            eeg_batch = eeg_batch.float().to(device)
            text_batch = text_batch.float().to(device)
            optimizer.zero_grad()
            pred = model(eeg_batch)
            loss = F.mse_loss(pred, text_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.6f}")

    # ==========================================
    # Save model and predicted embeddings
    # ==========================================
    ckpt_path = '/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints/eeg2text_classlevel.pt'
    torch.save({'state_dict': model.state_dict()}, ckpt_path)
    print(f"Model saved to: {ckpt_path}")

    # ==========================================
    # Inference on 7th block (test)
    # ==========================================
    print("\n=== Running inference on test block (7th) ===")

    test_indices = [list(GT_label[6]).index(element) for element in chosed_label]
    eeg_test = eegdata[6][test_indices, :]  # shape (10, 5, 62, 5)
    eeg_test = rearrange(torch.from_numpy(eeg_test), 'a b c d -> (a b) (c d)')  # flatten to (50, 310)
    eeg_test = normalize.transform(eeg_test)
    eeg_test = torch.from_numpy(eeg_test).float().to(device)

    model.eval()
    with torch.no_grad():
        pred_embeddings = model(eeg_test).cpu().numpy()

    out_dir = '/content/drive/MyDrive/EEG2Video_outputs/semantic_embeddings'
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'pred_embeddings_sub1_classlevel.npy'), pred_embeddings)

    print(f"Saved predicted embeddings: {pred_embeddings.shape}")
