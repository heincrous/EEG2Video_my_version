import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import preprocessing
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
import os

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
            # nn.BatchNorm1d(50000),
            nn.ReLU(),
            # nn.Linear(10000, 10000),
            # nn.ReLU(),
            nn.Linear(10000, 77 * 768)
        )

    def forward(self, eeg):
        eeg_embeddings = self.mlp(eeg)
          # shape: (batch_size)
        return eeg_embeddings


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

class Dataset():
    def __init__(self, eeg, text):


        self.eeg = eeg
        self.text = text
        self.len = eeg.shape[0]


    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.eeg[item], self.text[item]

GT_label = np.array([[23, 22, 9, 6, 18,       14, 5, 36, 25, 19,      28, 35, 3, 16, 24,      40, 15, 27, 38, 33,
             34, 4, 39, 17, 1,       26, 20, 29, 13, 32,     37, 2, 11, 12, 30,      31, 8, 21, 7, 10, ],
            [27, 33, 22, 28, 31,     12, 38, 4, 18, 17,      35, 39, 40, 5, 24,      32, 15, 13, 2, 16,
 	         34, 25, 19, 30, 23,     3, 8, 29, 7, 20,        11, 14, 37, 6, 21,      1, 10, 36, 26, 9, ],
            [15, 36, 31, 1, 34,      3, 37, 12, 4, 5,        21, 24, 14, 16, 39,     20, 28, 29, 18, 32,
             2, 27, 8, 19, 13,       10, 30, 40, 17, 26,     11, 9, 33, 25, 35,      7, 38, 22, 23, 6,],
            [16, 28, 23, 1, 39,      10, 35, 14, 19, 27,     37, 31, 5, 18, 11,      25, 29, 13, 20, 24,
            7, 34, 26, 4, 40 ,       12, 8, 22, 21, 30,      17, 2, 38, 9,  3 ,      36, 33, 6, 32, 15,],
            [18, 29, 7, 35, 22  ,    19, 12, 36, 8, 15,      28, 1, 34, 23, 20 ,     13, 37, 9, 16, 30  ,
             2, 33, 27, 21, 14 ,     38, 10, 17, 31, 3,      24, 39, 11, 32, 4,      25, 40, 5, 26, 6 ,],
            [29, 16, 1, 22, 34,      39, 24, 10, 8, 35,      27, 31, 23, 17, 2,      15, 25, 40, 3, 36,
             26, 6, 14, 37, 9,       12, 19, 30, 5, 28,      32, 4, 13, 18, 21,      20, 7, 11, 33, 38],
            [38, 34, 40, 10, 28,     7, 1, 37, 22, 9,        16, 5, 12, 36, 20,      30, 6, 15, 35, 2,
             31, 26, 18, 24, 8,      3, 23, 19, 14, 13,      21, 4, 25, 11, 32,      17, 39, 29, 33, 27]
            ])
chosed_label = [1, 3, 5, 11, 12, 13, 23, 27, 30, 38]
# chosed_label = [i for i in range(1,41)]
labels = np.zeros((40, 5, 62, 5))
for i in range(40):
    labels[i]=i
import random
def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')
seed_everything(114514)
device='cuda:0'
import datetime
def get_time():
    current_time = datetime.datetime.now()

    # 加8小时并取mod 24
    new_hour = (current_time.hour + 8) % 24

    # 构建新的时间，替换小时部分
    new_time = current_time.replace(hour=new_hour)

    # 格式化输出
    formatted_time = new_time.strftime("%m-%d_%H-%M")
    return formatted_time

if __name__ == '__main__':
    eegdata = np.load('/content/drive/MyDrive/EEG2Video_data/processed/DE_1per2s_authors/sub1.npy')
    
    print(eegdata.shape)
    EEG = []
    eeg=[]
    for i in range(6):
        indices = [list(GT_label[i]).index(element) for element in chosed_label]
        chosed_eeg = eegdata[i][indices,:]
        eeg.append(chosed_eeg)
        EEG.append(labels)
    EEG = np.stack(EEG, axis=0)
    eeg = np.stack(eeg,axis=0)
    EEG = torch.from_numpy(EEG)
    eeg = torch.from_numpy(eeg)
    EEG = rearrange(EEG, 'a b c e f -> (a b c) (e f)')
    eeg = rearrange(eeg, 'a b c e f -> (a b c) (e f)')

    clip_embeddings = np.load('/content/drive/MyDrive/EEG2Video_data/processed/CLIP_embeddings_authors/CLIP_embeddings_full.npy')  # shape [7,40,5,77,768]
    Text = []

    # use 6 blocks for training (block 7 reserved for testing)
    for i in range(6):
        text = torch.from_numpy(clip_embeddings[i])  # [40,5,77,768]
        indices = [list(GT_label[i]).index(lbl) for lbl in chosed_label]

        # === exact authors’ subsample + repeat scheme ===
        text = text[indices, :][:, ::5].repeat_interleave(5, dim=1)

        # flatten last two dims (77*768)
        text = text.reshape(len(chosed_label), 5, -1)
        Text.append(text)

    Text = torch.cat(Text, dim=0)  # [6*40,5,77*768]
    Text = Text.reshape(-1, Text.shape[-1])  # [6*40*5, 77*768]

    model = CLIP()
    model=model.to(device)
    
    ## Normalization
    normalize = preprocessing.StandardScaler()
    normalize.fit(eeg)
    eeg = normalize.transform(eeg)
    EEG = normalize.transform(EEG) 
    print(eeg.shape)
    dataset = Dataset(eeg, Text)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200 * len(dataloader))

    for epoch in tqdm(range(50)):
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(dataloader):
            eeg, text = batch
            eeg = eeg.float().to(device)
            text_embeddings = text.float().to(device)
            optimizer.zero_grad()
            eeg_embeddings = model(eeg)

            loss = F.mse_loss(eeg_embeddings, text_embeddings)

            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        print(epoch_loss)

    model_dict = model.state_dict()

    # ==========================================
    # Inference on held-out test block (block 7)
    # ==========================================
    test_block = 6  # 0-indexed, so this is the 7th block
    indices = [list(GT_label[test_block]).index(lbl) for lbl in chosed_label]

    # load test EEG and CLIP embeddings
    eeg_test = eegdata[test_block][indices, :]
    clip_embeddings = np.load('/content/drive/MyDrive/EEG2Video_data/processed/CLIP_embeddings_authors/CLIP_embeddings_full.npy')
    text_test = torch.from_numpy(clip_embeddings[test_block][indices]).reshape(len(chosed_label), 5, -1)
    text_test = text_test.reshape(-1, text_test.shape[-1])

    # reshape EEG like training
    eeg_test = torch.from_numpy(eeg_test)
    eeg_test = rearrange(eeg_test, 'b c e f -> (b c) (e f)')

    # normalize with same scaler
    eeg_test = normalize.transform(eeg_test)

    # move to device and infer
    model.eval()

    with torch.no_grad():
        preds = model(torch.tensor(eeg_test).float().to(device)).cpu().numpy()

    # reshape from [50, 77*768] → [50, 77, 768]
    preds = preds.reshape(len(chosed_label) * 5, 77, 768)  # 10 classes × 5 clips = 50

    # save predictions
    save_pred_path = "/content/drive/MyDrive/EEG2Video_outputs/semantic_embeddings"
    os.makedirs(save_pred_path, exist_ok=True)
    np.save(os.path.join(save_pred_path, "pred_embeddings_sub1_subset10.npy"), preds)
    print(f"Saved predicted embeddings: {preds.shape}")

    # cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity

    true = text_test.numpy()   # shape (50, 77*768)
    preds_flat = preds.reshape(preds.shape[0], -1)  # flatten to (50, 77*768)
    true_flat  = true.reshape(true.shape[0], -1)

    pred_norm = preds_flat / np.linalg.norm(preds_flat, axis=1, keepdims=True)
    true_norm = true_flat  / np.linalg.norm(true_flat,  axis=1, keepdims=True)

    mean_cos = np.mean(np.diag(cosine_similarity(pred_norm, true_norm)))
    print(f"\n=== Test Block (Block 7) Cosine Similarity: {mean_cos:.4f} ===")

    path = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"
    os.makedirs(path, exist_ok=True)

    save_path = os.path.join(path, "eeg2text_10_classes.pt")
    torch.save({'state_dict': model_dict}, save_path)

    print(f"Model saved to {save_path}")

    

