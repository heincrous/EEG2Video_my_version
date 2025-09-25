# ==========================================
# test_semantic_final.py
# ==========================================
import os, sys, json, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

# === Repo imports ===
repo_root = "/content/EEG2Video_my_version"
sys.path.append(repo_root)
from core_files.models import shallownet, deepnet, eegnet, tsconv, conformer, glfnet_mlp

# -------------------------------------------------
# Fusion model (frozen encoders, load classifier)
# -------------------------------------------------
class FusionModel(nn.Module):
    def __init__(self, encoders, num_classes=40):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        total_dim = sum([list(e.modules())[-1].out_features for e in encoders.values()])
        self.classifier = nn.Linear(total_dim, num_classes)
        self.total_dim = total_dim
    def forward(self, inputs, return_feats=False):
        feats = [enc(inputs[name]) for name, enc in self.encoders.items()]
        fused = torch.cat(feats, dim=-1)
        if return_feats:
            return fused
        return self.classifier(fused)

# -------------------------------------------------
# Semantic predictor
# -------------------------------------------------
class SemanticPredictor(nn.Module):
    def __init__(self, input_dim, hidden=[512,512], out_dim=768, dropout=0.3):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self.hidden = hidden
    def forward(self, x): return self.net(x)

# -------------------------------------------------
# Feature loaders
# -------------------------------------------------
def load_subject_features(subj_name, feat_types):
    base = "/content/drive/MyDrive/EEG2Video_data/processed"
    feats = {}
    for ftype in feat_types:
        if ftype == "de":
            feats[ftype] = np.load(os.path.join(base,"EEG_DE_1per1s",f"{subj_name}.npy")).reshape(7,40,10,62,5)
        elif ftype == "psd":
            feats[ftype] = np.load(os.path.join(base,"EEG_PSD_1per1s",f"{subj_name}.npy")).reshape(7,40,10,62,5)
        elif ftype == "windows":
            feats[ftype] = np.load(os.path.join(base,"EEG_windows",f"{subj_name}.npy"))
        elif ftype == "segments":
            feats[ftype] = np.load(os.path.join(base,"EEG_segments",f"{subj_name}.npy"))
        elif ftype == "combo":
            de = np.load(os.path.join(base,"EEG_DE_1per1s",f"{subj_name}.npy")).reshape(7,40,10,62,5)
            psd = np.load(os.path.join(base,"EEG_PSD_1per1s",f"{subj_name}.npy")).reshape(7,40,10,62,5)
            feats[ftype] = np.concatenate([de, psd], axis=-1)
    return feats

def preprocess_for_fusion(batch_dict, feat_types):
    processed = {}
    for ft in feat_types:
        x = batch_dict[ft]
        if ft in ["de","psd"]:
            x = x.reshape(x.shape[0], 62, 5)
        elif ft == "combo":
            x = x.reshape(x.shape[0], 62, 10)
        elif ft == "windows":
            if x.ndim == 4: x = x.mean(1)
            x = x.unsqueeze(1)
        elif ft == "segments":
            x = x.unsqueeze(1)
        processed[ft] = x
    return processed

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    proto_dir = "/content/drive/MyDrive/EEG2Video_checkpoints/prototype_checkpoints"
    candidates = [f for f in sorted(os.listdir(proto_dir)) if f.endswith(".pt")]
    if not candidates:
        print("No checkpoints found."); return

    for i,f in enumerate(candidates): print(f"{i}: {f}")
    choice = int(input("Select checkpoint index: ").strip())
    ckpt_path = os.path.join(proto_dir, candidates[choice])
    subj_name = candidates[choice].replace("prototype_checkpoint_","").replace(".pt","")

    # Load fusion config
    fusion_cfg = json.load(open(f"/content/drive/MyDrive/EEG2Video_checkpoints/fusion_checkpoints/fusion_config_{subj_name}.json"))
    feat_types = fusion_cfg["features"]
    encoders = {}
    for ftype, enc_name in fusion_cfg["encoders"].items():
        if enc_name == "glfnet_mlp":
            input_dim = 62*5 if ftype in ["de","psd"] else 62*10
            encoders[ftype] = glfnet_mlp(out_dim=128, emb_dim=64, input_dim=input_dim).to(device)
        elif enc_name == "shallownet":
            encoders[ftype] = shallownet(out_dim=128,C=62,T=100 if ftype=="windows" else 400).to(device)
        elif enc_name == "deepnet":
            encoders[ftype] = deepnet(out_dim=128,C=62,T=400).to(device)
        elif enc_name == "eegnet":
            encoders[ftype] = eegnet(out_dim=128,C=62,T=100 if ftype=="windows" else 400).to(device)
        elif enc_name == "tsconv":
            encoders[ftype] = tsconv(out_dim=128,C=62,T=100 if ftype=="windows" else 400).to(device)
        elif enc_name == "conformer":
            encoders[ftype] = conformer(out_dim=128,C=62,T=100 if ftype=="windows" else 400).to(device)

    fusion = FusionModel(encoders, num_classes=40).to(device)
    fusion.load_state_dict(torch.load(f"/content/drive/MyDrive/EEG2Video_checkpoints/fusion_checkpoints/fusion_checkpoint_{subj_name}.pt", map_location=device))
    fusion.eval()

    # Load semantic predictor checkpoint
    state = torch.load(ckpt_path, map_location=device)
    predictor = SemanticPredictor(input_dim=fusion.total_dim, hidden=state["hidden"]).to(device)
    predictor.load_state_dict(state["predictor"])
    fusion.classifier.load_state_dict(state["classifier"])
    predictor.eval(); fusion.classifier.eval()
    scaler = joblib.load(ckpt_path.replace(".pt","_scaler.pkl"))

    # Load EEG + BLIP embeddings
    eeg_feats = load_subject_features(subj_name, feat_types)
    text_emb = np.load("/content/drive/MyDrive/EEG2Video_data/processed/BLIP_embeddings/BLIP_embeddings.npy")  # (7,40,5,77*768)

    print("\n=== Semantic Predictor Validation ===")
    print("Checkpoint:", ckpt_path)
    print("Predictor hidden:", state["hidden"])

    trial_count = min([eeg_feats[f].shape[2] for f in feat_types])
    Xs, Ys = {ft: [] for ft in feat_types}, []
    for b in range(7):
        for c in range(40):
            for k in range(trial_count):
                for ft in feat_types:
                    Xs[ft].append(eeg_feats[ft][b,c,k])
                Ys.append(text_emb[b,c,k])
    Ys = torch.tensor(np.array(Ys),dtype=torch.float32).to(device)

    Xs = {ft: torch.tensor(np.array(Xs[ft]),dtype=torch.float32) for ft in feat_types}
    proc = preprocess_for_fusion(Xs, feat_types)
    with torch.no_grad():
        Feat = fusion({ft: proc[ft].to(device) for ft in feat_types}, return_feats=True).cpu().numpy()
    Feat = torch.tensor(scaler.transform(Feat),dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = predictor(Feat)

    print("Prediction variance:", preds.var(dim=0).mean().item())
    cos = F.cosine_similarity(preds, Ys, dim=-1).mean().item()
    print("Mean cosine(pred, BLIP target):", cos)

if __name__=="__main__":
    main()
