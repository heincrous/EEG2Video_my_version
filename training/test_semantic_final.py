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
    return outputs.last_hidden_state.mean(dim=1)  # pooled embedding [N,768]

# === Build caption bank per class ===
def build_caption_bank(caption_dir, num_classes=40):
    class_embeds = {}
    for c in range(num_classes):
        cap_path = os.path.join(caption_dir, f"class{c:02d}_captions.txt")
        with open(cap_path,"r") as f:
            caps = [line.strip() for line in f if line.strip()]
        embeds = encode_texts(caps)
        class_embeds[c] = embeds
    return class_embeds

# === Top-k accuracy evaluation ===
def evaluate_topk(predictor, eeg_feats, labels, caption_bank, k=5):
    predictor.eval()
    correct = 0
    total = len(labels)

    all_refs = []
    all_ref_classes = []
    for c, embeds in caption_bank.items():
        all_refs.append(embeds)
        all_ref_classes.extend([c]*embeds.size(0))
    all_refs = torch.cat(all_refs, dim=0)   # [Nref, 768]
    all_ref_classes = torch.tensor(all_ref_classes, device=device)

    with torch.no_grad():
        for x, y in zip(eeg_feats, labels):
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            pred = predictor(x).mean(dim=1)  # average tokens -> [1,768]
            sims = F.cosine_similarity(pred, all_refs)  # [Nref]
            topk_idx = sims.topk(k).indices
            topk_classes = all_ref_classes[topk_idx]
            if y in topk_classes:
                correct += 1
    return correct/total

# === Example usage ===
if __name__ == "__main__":
    # load predictor
    predictor = SemanticPredictor(in_dim=310)  # adjust in_dim to match training
    ckpt_path = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints/semantic_predictor.pt"
    predictor.load_state_dict(torch.load(ckpt_path, map_location=device))
    predictor.to(device)

    # load EEG features + labels
    eeg_feats = np.load("/content/drive/MyDrive/EEG2Video_data/test_eeg_feats.npy")   # shape [N, in_dim]
    labels    = np.load("/content/drive/MyDrive/EEG2Video_data/test_labels.npy")      # shape [N]

    # build caption bank
    caption_bank = build_caption_bank("/content/drive/MyDrive/EEG2Video_data/BLIP_captions", num_classes=40)

    # run eval
    acc1 = evaluate_topk(predictor, eeg_feats, labels, caption_bank, k=1)
    acc5 = evaluate_topk(predictor, eeg_feats, labels, caption_bank, k=5)

    print(f"Top-1 Accuracy: {acc1*100:.2f}%")
    print(f"Top-5 Accuracy: {acc5*100:.2f}%")
