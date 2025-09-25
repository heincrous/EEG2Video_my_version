# ==========================================
# test_semantic_upgraded.py
# ==========================================
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

# === Repo imports ===
repo_root = "/content/EEG2Video_my_version"
sys.path.append(repo_root)
from core_files.models import shallownet, deepnet, eegnet, tsconv, conformer, glfnet_mlp

# -------------------------------------------------
# Fusion model (frozen)
# -------------------------------------------------
class FusionModel(nn.Module):
    def __init__(self, encoders, num_classes=40):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        total_dim = sum([list(e.modules())[-1].out_features for e in encoders.values()])
        self.classifier = nn.Linear(total_dim, num_classes)
        self.total_dim = total_dim

    def forward(self, inputs, return_feats=False):
        feats = []
        for name, enc in self.encoders.items():
            feats.append(enc(inputs[name]))
        fused = torch.cat(feats, dim=-1)
        if return_feats:
            return fused
        return self.classifier(fused)

# -------------------------------------------------
# Semantic predictor
# -------------------------------------------------
class SemanticPredictor(nn.Module):
    def __init__(self, input_dim, hidden=[512,512], out_dim=77*768):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self.hidden = hidden
        print(f"[SemanticPredictor] Built with hidden={hidden}, out_dim={out_dim}")

    def forward(self, x):
        return self.net(x)

# -------------------------------------------------
# Load subject EEG features
# -------------------------------------------------
def load_subject_features(subj_name, feat_types):
    base = "/content/drive/MyDrive/EEG2Video_data/processed"
    feats = {}
    for ftype in feat_types:
        if ftype == "de":
            arr = np.load(os.path.join(base,"EEG_DE_1per1s",f"{subj_name}.npy"))
            feats[ftype] = arr.reshape(7,40,10,62,5)
        elif ftype == "psd":
            arr = np.load(os.path.join(base,"EEG_PSD_1per1s",f"{subj_name}.npy"))
            feats[ftype] = arr.reshape(7,40,10,62,5)
        elif ftype == "windows":
            feats[ftype] = np.load(os.path.join(base,"EEG_windows",f"{subj_name}.npy"))
        elif ftype == "segments":
            feats[ftype] = np.load(os.path.join(base,"EEG_segments",f"{subj_name}.npy"))
        elif ftype == "combo":
            de = np.load(os.path.join(base,"EEG_DE_1per1s",f"{subj_name}.npy")).reshape(7,40,10,62,5)
            psd = np.load(os.path.join(base,"EEG_PSD_1per1s",f"{subj_name}.npy")).reshape(7,40,10,62,5)
            feats[ftype] = np.concatenate([de, psd], axis=-1)
    return feats

# -------------------------------------------------
# Preprocess for fusion
# -------------------------------------------------
def preprocess_for_fusion(batch_dict, feat_types):
    processed = {}
    for ft in feat_types:
        x = batch_dict[ft]
        if ft in ["de","psd"]:
            x = x.reshape(x.shape[0], 62, 5)
        elif ft == "combo":
            x = x.reshape(x.shape[0], 62, 10)
        elif ft == "windows":
            if x.ndim == 4:
                x = x.mean(1)
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
    if not os.path.isdir(proto_dir):
        print("No prototype_checkpoints directory found.")
        return
    candidates = [f for f in sorted(os.listdir(proto_dir)) if f.endswith(".pt")]
    if not candidates:
        print("No checkpoints found in prototype_checkpoints.")
        return

    print("Available checkpoints:")
    for i, f in enumerate(candidates):
        print(f"{i}: {f}")
    choice = int(input("Select checkpoint index: ").strip())

    fname = candidates[choice]
    ckpt_path = os.path.join(proto_dir, fname)
    subj_name = fname.replace("prototype_checkpoint_","").replace(".pt","")
    print(f"Selected subject: {subj_name}")

    # load fusion config + checkpoint
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

    # load state
    state = torch.load(ckpt_path, map_location=device)

    # use saved hidden config
    hidden_cfg = state.get("hidden", [512,512])

    predictor = SemanticPredictor(input_dim=fusion.total_dim, hidden=hidden_cfg).to(device)
    predictor.load_state_dict(state["predictor"])
    fusion.classifier.load_state_dict(state["classifier"])
    predictor.eval(); fusion.classifier.eval()

    scaler = joblib.load(ckpt_path.replace(".pt","_scaler.pkl"))

    eeg_feats = load_subject_features(subj_name, feat_types)
    text_emb = np.load("/content/drive/MyDrive/EEG2Video_data/processed/BLIP_embeddings/BLIP_embeddings.npy").reshape(-1,77*768)
    text_tensor = torch.tensor(text_emb, dtype=torch.float32).to(device)

    print("\n=== BLIP Embeddings Check ===")
    blip_var = text_tensor.var(dim=0).mean().item()
    cos_blip = F.cosine_similarity(text_tensor[0].unsqueeze(0), text_tensor[1:], dim=-1).mean().item()
    print("BLIP variance:", blip_var)
    print("Mean cosine similarity between BLIP embeddings:", cos_blip)

    trial_count = min([eeg_feats[f].shape[2] for f in feat_types])
    Xs, Ys = {ft: [] for ft in feat_types}, []
    for c in range(40):
        for k in range(trial_count):
            for ft in feat_types:
                Xs[ft].append(eeg_feats[ft][0,c,k])
            Ys.append(text_emb[c*trial_count+k])
    Xs = {ft: torch.tensor(np.array(Xs[ft]),dtype=torch.float32) for ft in feat_types}
    Ys = torch.tensor(np.array(Ys),dtype=torch.float32).to(device)

    proc = preprocess_for_fusion(Xs, feat_types)
    with torch.no_grad():
        Feat_raw = fusion({ft: proc[ft].to(device) for ft in feat_types}, return_feats=True).cpu()

    print("\n=== Fusion Features Check ===")
    print("Fusion variance:", Feat_raw.var(dim=0).mean().item())
    cos_feats = F.cosine_similarity(Feat_raw[0].unsqueeze(0), Feat_raw[1:], dim=-1)
    print("Mean cosine similarity between fusion features:", cos_feats.mean().item())

    Feat = torch.tensor(scaler.transform(Feat_raw.numpy()),dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = predictor(Feat)

    print("\n=== Semantic Predictor Check ===")
    print("Prediction variance:", pred.var(dim=0).mean().item())
    cos_preds = F.cosine_similarity(pred[0].unsqueeze(0), pred[1:], dim=-1)
    print("Mean cosine similarity between predictions:", cos_preds.mean().item())
    cos_gt = F.cosine_similarity(pred, Ys, dim=-1)
    print("Mean cosine(pred, target):", cos_gt.mean().item())

if __name__=="__main__":
    main()
