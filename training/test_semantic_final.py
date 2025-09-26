# ==========================================
# evaluate_semantic_predictor_block7.py
# ==========================================
import os, torch, numpy as np
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel

# === Semantic Predictor ===
class SemanticPredictor(torch.nn.Module):
    def __init__(self, in_dim, out_shape=(77,768)):
        super().__init__()
        out_dim = out_shape[0] * out_shape[1]
        layers = [
            torch.nn.Linear(in_dim, 10000),
            torch.nn.ReLU(),
        ]
        for _ in range(3):
            layers.append(torch.nn.Linear(10000, 10000))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(10000, out_dim))
        self.mlp = torch.nn.Sequential(*layers)
        self.out_shape = out_shape

    def forward(self, x):
        return self.mlp(x).view(-1, *self.out_shape)

# === Load Stable Diffusion CLIP text encoder ===
device = "cuda" if torch.cuda.is_available() else "cpu"
sd_path = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"

clip_model = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").to(device)
tokenizer   = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")

@torch.no_grad()
def encode_texts(texts):
    inputs = tokenizer(texts, padding=True, return_tensors="pt", truncation=True).to(device)
    outputs = clip_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # pooled [N,768]

# === Top-k accuracy evaluation ===
def evaluate_topk(predictor, eeg_feats, labels, blip_embeds, k=5):
    predictor.eval()
    correct = 0
    total = len(labels)

    # build reference bank from block 7 BLIP embeddings
    all_refs = torch.tensor(blip_embeds, dtype=torch.float32).to(device)  # [Nref, 77,768]
    all_refs = all_refs.mean(dim=1)  # pool tokens -> [Nref,768]

    with torch.no_grad():
        for x, y in zip(eeg_feats, labels):
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            pred = predictor(x).mean(dim=1)  # [1,768]
            sims = F.cosine_similarity(pred, all_refs)  # [Nref]
            # top-k indices
            topk_idx = sims.topk(k).indices.cpu().numpy()
            if y in topk_idx:  # ground-truth class index in top-k
                correct += 1
    return correct/total

# === Example usage ===
if __name__ == "__main__":
    # load predictor
    predictor = SemanticPredictor(in_dim=310)  # adjust in_dim to match training
    ckpt_path = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints/semantic_predictor_sub1_de_best.pt"
    predictor.load_state_dict(torch.load(ckpt_path, map_location=device))
    predictor.to(device)

    # load EEG features (sub1.npy) and select block 7
    eeg_all = np.load("/content/drive/MyDrive/EEG2Video_data/processed/sub1.npy")  # [7,40,5,62,5] for DE
    eeg_block7 = eeg_all[6]  # block index 6 = 7th block -> shape [40,5,62,5]
    eeg_feats = eeg_block7.reshape(-1, 62*5)  # [200,310]
    labels = np.repeat(np.arange(40), 5)      # [200]

    # load BLIP embeddings and select block 7
    blip_all = np.load("/content/drive/MyDrive/EEG2Video_data/processed/BLIP_embeddings/BLIP_embeddings.npy")  # [7,40,5,77,768]
    blip_block7 = blip_all[6].reshape(-1, 77,768)  # [200,77,768]

    # run eval
    acc1 = evaluate_topk(predictor, eeg_feats, labels, blip_block7, k=1)
    acc5 = evaluate_topk(predictor, eeg_feats, labels, blip_block7, k=5)

    print(f"Block 7 Top-1 Accuracy: {acc1*100:.2f}%")
    print(f"Block 7 Top-5 Accuracy: {acc5*100:.2f}%")
