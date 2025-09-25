import os, sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import joblib

# === Repo imports ===
repo_root = "/content/EEG2Video_my_version"
sys.path.append(repo_root)
from core_files.models import shallownet, deepnet, eegnet, tsconv, conformer, glfnet_mlp

# -------------------------------------------------
# Fusion model (frozen for feature extraction)
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
# Loader helper
# -------------------------------------------------
def load_fusion(subj_name, device):
    ckpt_dir = "/content/drive/MyDrive/EEG2Video_checkpoints/fusion_checkpoints"
    ckpt_path = os.path.join(ckpt_dir, f"fusion_checkpoint_{subj_name}.pt")
    cfg_path  = os.path.join(ckpt_dir, f"fusion_config_{subj_name}.json")

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    feat_types = cfg["features"]
    encoders = {}
    for ftype, enc_name in cfg["encoders"].items():
        if enc_name == "glfnet_mlp":
            if ftype in ["de", "psd"]:
                input_dim = 62*5
            elif ftype == "combo":
                input_dim = 62*10
            else:
                raise ValueError(f"Unsupported {ftype} for MLP")
            encoders[ftype] = glfnet_mlp(out_dim=128, emb_dim=64, input_dim=input_dim).to(device)
        elif enc_name == "shallownet":
            encoders[ftype] = shallownet(out_dim=128, C=62, T=100 if ftype=="windows" else 400).to(device)
        elif enc_name == "deepnet":
            encoders[ftype] = deepnet(out_dim=128, C=62, T=400).to(device)
        elif enc_name == "eegnet":
            encoders[ftype] = eegnet(out_dim=128, C=62, T=100 if ftype=="windows" else 400).to(device)
        elif enc_name == "tsconv":
            encoders[ftype] = tsconv(out_dim=128, C=62, T=100 if ftype=="windows" else 400).to(device)
        elif enc_name == "conformer":
            encoders[ftype] = conformer(out_dim=128, C=62, T=100 if ftype=="windows" else 400).to(device)
        else:
            raise ValueError(f"Unknown encoder {enc_name}")

    fusion = FusionModel(encoders, num_classes=40).to(device)
    fusion.load_state_dict(torch.load(ckpt_path, map_location=device))
    fusion.eval()
    for p in fusion.parameters():
        p.requires_grad = False
    return fusion, feat_types

# -------------------------------------------------
# Semantic Predictor
# -------------------------------------------------
class SemanticPredictor(nn.Module):
    def __init__(self, input_dim, hidden=[1024,2048,1024], out_dim=77*768):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -------------------------------------------------
# EEG-Text Dataset
# -------------------------------------------------
class EEGTextDataset(Dataset):
    def __init__(self, eeg_feats, text_embs):
        self.eeg_feats = eeg_feats
        self.text_embs = text_embs
    def __len__(self): return len(self.eeg_feats)
    def __getitem__(self, idx):
        return self.eeg_feats[idx], self.text_embs[idx]

# -------------------------------------------------
# Helper to load and reshape subject features
# -------------------------------------------------
def load_subject_features(subj_name, feat_types):
    base = "/content/drive/MyDrive/EEG2Video_data/processed"
    feats = {}
    for ftype in feat_types:
        if ftype == "de":
            arr = np.load(os.path.join(base,"EEG_DE_1per1s",f"{subj_name}.npy"))  # (7,40,5,2,62,5)
            feats[ftype] = arr.reshape(7,40,10,62,5)
        elif ftype == "psd":
            arr = np.load(os.path.join(base,"EEG_PSD_1per1s",f"{subj_name}.npy"))
            feats[ftype] = arr.reshape(7,40,10,62,5)
        elif ftype == "windows":
            feats[ftype] = np.load(os.path.join(base,"EEG_windows",f"{subj_name}.npy"))  # (7,40,5,7,62,100)
        elif ftype == "segments":
            feats[ftype] = np.load(os.path.join(base,"EEG_segments",f"{subj_name}.npy")) # (7,40,5,62,400)
        elif ftype == "combo":
            de = np.load(os.path.join(base,"EEG_DE_1per1s",f"{subj_name}.npy")).reshape(7,40,10,62,5)
            psd = np.load(os.path.join(base,"EEG_PSD_1per1s",f"{subj_name}.npy")).reshape(7,40,10,62,5)
            feats[ftype] = np.concatenate([de, psd], axis=-1)  # (7,40,10,62,10)
        else:
            raise ValueError(f"Unknown feature type {ftype}")
    return feats

# -------------------------------------------------
# Preprocess a batch for fusion input
# -------------------------------------------------
def preprocess_for_fusion(batch_dict, feat_types):
    processed = {}
    for ft in feat_types:
        x = batch_dict[ft]
        if ft in ["de","psd"]:
            x = x.reshape(x.shape[0], 62, 5)  # keep channels × bands
        elif ft == "combo":
            x = x.reshape(x.shape[0], 62, 10)
        elif ft == "windows":
            if x.ndim == 3 and x.shape[0] == 7:   # single sample case
                x = x.mean(0)                     # -> (62,100)
            elif x.ndim == 4:                     # batch case (batch,7,62,100)
                x = x.mean(1)                     # -> (batch,62,100)
            x = x.unsqueeze(1)                    # -> (batch,1,62,100)
        elif ft == "segments":
            x = x.unsqueeze(1)  # (batch,1,62,400)
        processed[ft] = x
    return processed

# -------------------------------------------------
# Train + Eval (7-fold CV with scaler saving)
# -------------------------------------------------
def run_cv(subj_name, fusion, feat_types, eeg_feats, text_emb, device):
    best_global_loss = float("inf")
    best_ckpt_path = None
    save_dir = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    top_losses = []

    # align trial count across features
    trial_count = min([eeg_feats[ft].shape[2] for ft in feat_types])
    print(f"Trial counts per feature: {[ (f,eeg_feats[f].shape[2]) for f in feat_types ]}, using {trial_count}")

    for test_block in tqdm(range(7), desc=f"Cross-validation {subj_name}"):
        val_block = (test_block - 1) % 7
        train_blocks = [i for i in range(7) if i not in [test_block, val_block]]

        # collect splits
        Xs, Ys = {ft: {"train": [], "val": [], "test": []} for ft in feat_types}, {"train": [], "val": [], "test": []}
        for split, blocks in [("train", train_blocks), ("val", [val_block]), ("test", [test_block])]:
            for b in blocks:
                for c in range(40):
                    for k in range(trial_count):
                        for ft in feat_types:
                            Xs[ft][split].append(eeg_feats[ft][b, c, k])
                        Ys[split].append(text_emb[b*40*trial_count + c*trial_count + k])

        # convert to tensors
        X_train = {ft: torch.tensor(np.array(Xs[ft]["train"]), dtype=torch.float32) for ft in feat_types}
        X_val   = {ft: torch.tensor(np.array(Xs[ft]["val"]),   dtype=torch.float32) for ft in feat_types}
        X_test  = {ft: torch.tensor(np.array(Xs[ft]["test"]),  dtype=torch.float32) for ft in feat_types}
        Y_train = torch.tensor(np.array(Ys["train"]), dtype=torch.float32)
        Y_val   = torch.tensor(np.array(Ys["val"]),   dtype=torch.float32)
        Y_test  = torch.tensor(np.array(Ys["test"]),  dtype=torch.float32)

        # preprocess into correct shapes for fusion
        X_train_proc = preprocess_for_fusion(X_train, feat_types)
        X_val_proc   = preprocess_for_fusion(X_val, feat_types)
        X_test_proc  = preprocess_for_fusion(X_test, feat_types)

        # extract fusion features
        with torch.no_grad():
            Feat_train = fusion({ft: X_train_proc[ft].to(device) for ft in feat_types}, return_feats=True).cpu().numpy()
            Feat_val   = fusion({ft: X_val_proc[ft].to(device)   for ft in feat_types}, return_feats=True).cpu().numpy()
            Feat_test  = fusion({ft: X_test_proc[ft].to(device)  for ft in feat_types}, return_feats=True).cpu().numpy()

        # fit scaler on training features
        scaler = StandardScaler()
        scaler.fit(Feat_train)

        # transform splits
        Feat_train = torch.tensor(scaler.transform(Feat_train), dtype=torch.float32)
        Feat_val   = torch.tensor(scaler.transform(Feat_val),   dtype=torch.float32)
        Feat_test  = torch.tensor(scaler.transform(Feat_test),  dtype=torch.float32)

        train_loader = DataLoader(EEGTextDataset(Feat_train, Y_train), batch_size=32, shuffle=True)
        val_loader   = DataLoader(EEGTextDataset(Feat_val,   Y_val),   batch_size=32, shuffle=False)
        test_loader  = DataLoader(EEGTextDataset(Feat_test,  Y_test),  batch_size=32, shuffle=False)

        predictor = SemanticPredictor(input_dim=fusion.total_dim).to(device)
        optimizer = torch.optim.Adam(predictor.parameters(), lr=5e-4)
        best_val_loss, best_state = float("inf"), None
        ckpt_path = os.path.join(save_dir, f"semantic_checkpoint_{subj_name}.pt")

        for epoch in tqdm(range(50), desc=f"Fold {test_block} training", leave=False):
            predictor.train()
            for eeg, text in train_loader:
                eeg, text = eeg.to(device), text.to(device)
                optimizer.zero_grad()
                pred = predictor(eeg)
                loss = F.mse_loss(pred, text)
                loss.backward()
                optimizer.step()

            # validation
            predictor.eval()
            val_loss = 0
            with torch.no_grad():
                for eeg, text in val_loader:
                    eeg, text = eeg.to(device), text.to(device)
                    pred = predictor(eeg)
                    val_loss += F.mse_loss(pred, text).item()
            avg_val = val_loss/len(val_loader)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = predictor.state_dict()
                torch.save(best_state, ckpt_path)
                joblib.dump(scaler, ckpt_path.replace(".pt", "_scaler.pkl"))

        # test with best
        predictor.load_state_dict(torch.load(ckpt_path, map_location=device))
        scaler = joblib.load(ckpt_path.replace(".pt", "_scaler.pkl"))
        predictor.eval()
        test_loss = 0
        with torch.no_grad():
            for eeg, text in test_loader:
                eeg, text = eeg.to(device), text.to(device)
                pred = predictor(eeg)
                test_loss += F.mse_loss(pred, text).item()
        avg_test = test_loss/len(test_loader)
        top_losses.append(avg_test)

        if avg_test < best_global_loss:
            best_global_loss = avg_test
            best_ckpt_path = ckpt_path

        print(f"Fold {test_block}: val_loss={best_val_loss:.4f}, test_loss={avg_test:.4f}")

    print("\n=== Final Results ===")
    print(f"Average Test Loss across folds: {np.mean(top_losses):.4f}")
    print(f"Best checkpoint: {best_ckpt_path} (test_loss={best_global_loss:.4f})")

    return best_ckpt_path

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = "/content/drive/MyDrive/EEG2Video_checkpoints/fusion_checkpoints"
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    ckpts = sorted(ckpts)
    print("Available fusion checkpoints:")
    for i, ckpt in enumerate(ckpts):
        print(f"{i}: {ckpt}")

    choice = int(input("Select subject index: ").strip())
    subj_name = ckpts[choice].replace("fusion_checkpoint_", "").replace(".pt", "")
    print(f"Selected subject: {subj_name}")

    fusion, feat_types = load_fusion(subj_name, device)
    eeg_feats = load_subject_features(subj_name, feat_types)
    text_emb = np.load("/content/drive/MyDrive/EEG2Video_data/processed/BLIP_embeddings/BLIP_embeddings.npy").reshape(-1, 77*768)

    run_cv(subj_name, fusion, feat_types, eeg_feats, text_emb, device)

if __name__ == "__main__":
    main()
