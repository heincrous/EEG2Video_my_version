# ==========================================
# Semantic Predictor Evaluation (Block 7 Test)
# ==========================================

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter

# === Repo imports ===
repo_root = "/content/EEG2Video_my_version"
sys.path.append(repo_root)
from core_files.models import shallownet, deepnet, eegnet, tsconv, conformer

# === Semantic Predictor ===
class SemanticPredictor(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=[1024,2048,1024], out_dim=77*768):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev,h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev,out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

# === Similarity helpers ===
def tokenwise_cosine(a,b):
    a = a/(np.linalg.norm(a,axis=-1,keepdims=True)+1e-8)
    b = b/(np.linalg.norm(b,axis=-1,keepdims=True)+1e-8)
    return float((a*b).sum(-1).mean())

def build_prototypes(blip_emb):
    protos = {}
    for class_id in range(40):
        clips = blip_emb[:6,class_id]       # blocks 0–5
        clips = clips.reshape(-1,77,768)    # (30,77,768)
        protos[class_id] = clips.mean(0)
    return protos

# === Main ===
if __name__=="__main__":
    bundle_path = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_EEG_bundle.npz"
    ckpt_dir    = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"

    # Pick checkpoint
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    for i,ck in enumerate(ckpts): print(f"[{i}] {ck}")
    choice = int(input("\nSelect checkpoint index: ").strip())
    ckpt_file = ckpts[choice]
    ckpt_path = os.path.join(ckpt_dir, ckpt_file)

    # Parse filename
    parts = ckpt_file.replace(".pt","").split("_")
    feature_type, encoder_choice = parts[1], parts[2]
    print(f"\n[Config] Feature={feature_type}, Encoder={encoder_choice}")

    # Load data
    data = np.load(bundle_path, allow_pickle=True)
    blip_emb = data["BLIP_embeddings"]   # (7,40,5,77,768)
    eeg_dict = data["EEG_data"].item()

    # Build encoder+predictor
    encoders = {
        "shallownet": shallownet,
        "deepnet": deepnet,
        "eegnet": eegnet,
        "tsconv": tsconv,
        "conformer": conformer
    }
    C,T = (62,100) if feature_type=="windows" else (62,400)
    encoder = encoders[encoder_choice](out_dim=512,C=C,T=T).cuda()
    predictor = SemanticPredictor(input_dim=512).cuda()

    # Load weights
    ckpt = torch.load(ckpt_path,map_location="cuda")
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    predictor.load_state_dict(ckpt["semantic_state_dict"])
    encoder.eval(); predictor.eval()

    # Build BLIP prototypes
    prototypes = build_prototypes(blip_emb)

    # Eval loop (Block 6 = 7th block = test)
    correct1=correct5=total=0
    mse_vals=[]; cos_vals=[]; token_cos_vals=[]
    pred_classes=[]; true_classes=[]

    with tqdm(total=len(eeg_dict)*40*5, desc="Evaluating") as pbar:
        for subj,feats in eeg_dict.items():
            eeg = feats[f"EEG_{feature_type}"][6]   # block 6 test
            txt = blip_emb[6]
            for ci in range(40):
                for cj in range(5):
                    if feature_type=="windows":
                        # (7,62,100) → run each window → mean
                        windows = torch.tensor(eeg[ci,cj],dtype=torch.float32).unsqueeze(1).cuda() # (7,1,62,100)
                        with torch.no_grad():
                            feats = encoder(windows)        # (7,512)
                            pred = predictor(feats.mean(0,keepdim=True)) # (1,59136)
                            pred_emb = pred.view(77,768).cpu().numpy()
                    else: # segments
                        x = torch.tensor(eeg[ci,cj],dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda() # (1,1,62,400)
                        with torch.no_grad():
                            feats = encoder(x)              # (1,512)
                            pred = predictor(feats)         # (1,59136)
                            pred_emb = pred.view(77,768).cpu().numpy()

                    true_emb = txt[ci,cj]

                    # Classify
                    sims={cid:tokenwise_cosine(pred_emb,proto) for cid,proto in prototypes.items()}
                    ranked=sorted(sims.items(), key=lambda x:x[1], reverse=True)
                    top1,top5 = ranked[0][0],[cid for cid,_ in ranked[:5]]
                    correct1 += int(top1==ci); correct5 += int(ci in top5); total+=1
                    pred_classes.append(top1); true_classes.append(ci)

                    # Metrics
                    mse_vals.append(np.mean((pred_emb-true_emb)**2))
                    cos=np.dot(pred_emb.flatten(),true_emb.flatten())/(np.linalg.norm(pred_emb.flatten())*np.linalg.norm(true_emb.flatten())+1e-8)
                    cos_vals.append(cos)
                    a=pred_emb/(np.linalg.norm(pred_emb,axis=-1,keepdims=True)+1e-8)
                    b=true_emb/(np.linalg.norm(true_emb,axis=-1,keepdims=True)+1e-8)
                    token_cos_vals.append((a*b).sum(-1).mean())
                    pbar.update(1)

    print(f"\nTop-1 Acc: {correct1/total:.4f}, Top-5 Acc: {correct5/total:.4f}")
    print(f"Mean MSE: {np.mean(mse_vals):.6f}")
    print(f"Mean Cosine: {np.mean(cos_vals):.4f}")
    print(f"Mean Token-wise cosine: {np.mean(token_cos_vals):.4f}")

    # Per-class distributions
    pc,tc = Counter(pred_classes), Counter(true_classes)
    print("\nPredicted class distribution:")
    for cls in range(40):
        print(f"Class {cls}: {pc.get(cls,0)} ({pc.get(cls,0)/total:.2%})")
    print("\nTrue class distribution:")
    for cls in range(40):
        print(f"Class {cls}: {tc.get(cls,0)} ({tc.get(cls,0)/total:.2%})")
