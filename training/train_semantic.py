# ==========================================
# train_semantic_with_expansion.py
# ==========================================
import os, sys, json, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

# === Repo imports ===
repo_root = "/content/EEG2Video_my_version"
sys.path.append(repo_root)
from core_files.models import shallownet, deepnet, eegnet, tsconv, conformer, glfnet_mlp

# -------------------------------------------------
# Fusion model (encoders frozen, classifier trainable)
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
            input_dim = 62*5 if ftype in ["de","psd"] else 62*10
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
    fusion = FusionModel(encoders, num_classes=40).to(device)
    fusion.load_state_dict(torch.load(ckpt_path, map_location=device))
    for _, p in fusion.encoders.named_parameters():
        p.requires_grad = False
    return fusion, feat_types

# -------------------------------------------------
# Semantic Predictor (EEG → pooled 768 BLIP)
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
# Expansion Head (pooled 768 → 77×768 tokens)
# -------------------------------------------------
class ExpansionHead(nn.Module):
    def __init__(self, in_dim=768, out_tokens=77, token_dim=768, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_tokens*token_dim)
        )
        self.out_tokens, self.token_dim = out_tokens, token_dim
    def forward(self, x):
        out = self.net(x)   # (batch, out_tokens*token_dim)
        return out.view(-1, self.out_tokens, self.token_dim)  # (batch,77,768)

# -------------------------------------------------
# Dataset
# -------------------------------------------------
class EEGMultiDataset(Dataset):
    def __init__(self, eeg_feats, pooled_embs, full_embs, labels):
        self.eeg_feats = eeg_feats
        self.pooled_embs = pooled_embs
        self.full_embs = full_embs
        self.labels = labels
    def __len__(self): return len(self.eeg_feats)
    def __getitem__(self, idx):
        return self.eeg_feats[idx], self.pooled_embs[idx], self.full_embs[idx], self.labels[idx]

# -------------------------------------------------
# Losses
# -------------------------------------------------
def cosine_loss(pred, target):
    return 1 - (F.normalize(pred, p=2, dim=-1) * F.normalize(target, p=2, dim=-1)).sum(dim=-1).mean()

# -------------------------------------------------
# Feature utils
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
# Training wrapper
# -------------------------------------------------
def run_cv(subj_name, fusion, feat_types, eeg_feats, text_emb, device,
           lambda_cls=1.0, lambda_sem=0.5, lambda_exp=0.5):

    save_dir = "/content/drive/MyDrive/EEG2Video_checkpoints/prototype_checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    trial_count = min([eeg_feats[ft].shape[2] for ft in feat_types])

    # BLIP embeddings: pool and keep full
    pooled_emb = text_emb.mean(axis=3)   # (7,40,5,768)
    full_emb   = text_emb                # (7,40,5,77,768)

    ce_loss = nn.CrossEntropyLoss()
    best_global_loss, best_ckpt_path = float("inf"), None

    fold_bar = tqdm(range(7), desc="Cross-validation folds")
    for test_block in fold_bar:
        val_block = (test_block - 1) % 7
        train_blocks = [i for i in range(7) if i not in [test_block,val_block]]

        Xs, Ys_pooled, Ys_full, Ls = [], [], [], []
        for split, blocks in [("train", train_blocks), ("val", [val_block]), ("test", [test_block])]:
            X_dict, Yp, Yf, lbls = [], [], [], []
            for b in blocks:
                for c in range(40):
                    for k in range(trial_count):
                        X_dict.append([eeg_feats[ft][b,c,k] for ft in feat_types])
                        Yp.append(pooled_emb[b,c,k])
                        Yf.append(full_emb[b,c,k])
                        lbls.append(c)
            yield_dict = {
                "X": np.array(X_dict),
                "Yp": np.array(Yp),
                "Yf": np.array(Yf),
                "L": np.array(lbls)
            }
            if split=="train": train = yield_dict
            elif split=="val": val = yield_dict
            else: test = yield_dict

        # convert
        def to_tensor(data):
            return torch.tensor(data, dtype=torch.float32)
        Y_train_p, Y_val_p, Y_test_p = map(to_tensor,[train["Yp"],val["Yp"],test["Yp"]])
        Y_train_f, Y_val_f, Y_test_f = map(to_tensor,[train["Yf"],val["Yf"],test["Yf"]])
        L_train, L_val, L_test = map(torch.tensor,[train["L"],val["L"],test["L"]])

        # Preprocess EEG
        def prep_split(data):
            Xs = {ft: np.array([row[i] for row in data["X"]]) for i,ft in enumerate(feat_types)}
            return {ft: torch.tensor(Xs[ft],dtype=torch.float32) for ft in feat_types}
        X_train, X_val, X_test = map(prep_split,[train,val,test])

        X_train_proc = preprocess_for_fusion(X_train, feat_types)
        X_val_proc   = preprocess_for_fusion(X_val, feat_types)
        X_test_proc  = preprocess_for_fusion(X_test, feat_types)

        with torch.no_grad():
            Feat_train = fusion({ft:X_train_proc[ft].to(device) for ft in feat_types}, return_feats=True).cpu().numpy()
            Feat_val   = fusion({ft:X_val_proc[ft].to(device) for ft in feat_types}, return_feats=True).cpu().numpy()
            Feat_test  = fusion({ft:X_test_proc[ft].to(device) for ft in feat_types}, return_feats=True).cpu().numpy()

        scaler = StandardScaler().fit(Feat_train)
        Feat_train = torch.tensor(scaler.transform(Feat_train),dtype=torch.float32)
        Feat_val   = torch.tensor(scaler.transform(Feat_val),dtype=torch.float32)
        Feat_test  = torch.tensor(scaler.transform(Feat_test),dtype=torch.float32)

        train_loader = DataLoader(EEGMultiDataset(Feat_train,Y_train_p,Y_train_f,L_train),batch_size=128,shuffle=True)
        val_loader   = DataLoader(EEGMultiDataset(Feat_val,Y_val_p,Y_val_f,L_val),batch_size=128,shuffle=False)
        test_loader  = DataLoader(EEGMultiDataset(Feat_test,Y_test_p,Y_test_f,L_test),batch_size=128,shuffle=False)

        predictor = SemanticPredictor(input_dim=fusion.total_dim, hidden=[512,512], out_dim=768).to(device)
        expansion = ExpansionHead(in_dim=768, out_tokens=77, token_dim=768).to(device)
        optimizer = torch.optim.Adam(
            list(predictor.parameters())+list(expansion.parameters())+list(fusion.classifier.parameters()),
            lr=5e-4
        )

        best_val_loss, best_state = float("inf"), None
        ckpt_path = os.path.join(save_dir, f"prototype_checkpoint_{subj_name}.pt")

        epoch_bar = tqdm(range(30), desc=f"Fold {test_block} training", leave=False)
        for epoch in epoch_bar:
            predictor.train(); expansion.train(); fusion.classifier.train()
            for eeg, y_p, y_f, lbl in train_loader:
                eeg, y_p, y_f, lbl = eeg.to(device), y_p.to(device), y_f.to(device), lbl.to(device)
                optimizer.zero_grad()
                pred_p = predictor(eeg)
                pred_cls = fusion.classifier(eeg)
                pred_full = expansion(pred_p)

                loss_sem = cosine_loss(pred_p, y_p)
                loss_cls = ce_loss(pred_cls, lbl)
                loss_exp = cosine_loss(pred_full.view(-1,768), y_f.view(-1,768))

                loss = lambda_sem*loss_sem + lambda_cls*loss_cls + lambda_exp*loss_exp
                loss.backward(); optimizer.step()

            # validation
            predictor.eval(); expansion.eval(); fusion.classifier.eval()
            val_loss = 0
            with torch.no_grad():
                for eeg, y_p, y_f, lbl in val_loader:
                    eeg, y_p, y_f, lbl = eeg.to(device), y_p.to(device), y_f.to(device), lbl.to(device)
                    pred_p = predictor(eeg); pred_cls = fusion.classifier(eeg); pred_full = expansion(pred_p)
                    val_loss += (
                        lambda_sem*cosine_loss(pred_p, y_p) +
                        lambda_cls*ce_loss(pred_cls, lbl) +
                        lambda_exp*cosine_loss(pred_full.view(-1,768), y_f.view(-1,768))
                    ).item()
            avg_val = val_loss/len(val_loader)
            epoch_bar.set_postfix(val_loss=avg_val)

            if avg_val < best_val_loss:
                best_val_loss, best_state = avg_val, {
                    "predictor": predictor.state_dict(),
                    "expansion": expansion.state_dict(),
                    "classifier": fusion.classifier.state_dict(),
                    "hidden": predictor.hidden
                }
                torch.save(best_state, ckpt_path)
                joblib.dump(scaler, ckpt_path.replace(".pt","_scaler.pkl"))

        if best_state and best_val_loss < best_global_loss:
            best_global_loss, best_ckpt_path = best_val_loss, ckpt_path

    print("\n=== Final Results ===")
    print(f"Best checkpoint: {best_ckpt_path} (val_loss={best_global_loss:.4f})")
    return best_ckpt_path

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = "/content/drive/MyDrive/EEG2Video_checkpoints/fusion_checkpoints"
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
    for i, ckpt in enumerate(ckpts): print(f"{i}: {ckpt}")
    choice = int(input("Select subject index: ").strip())
    subj_name = ckpts[choice].replace("fusion_checkpoint_","").replace(".pt","")
    print(f"Selected subject: {subj_name}")

    fusion, feat_types = load_fusion(subj_name, device)
    eeg_feats = load_subject_features(subj_name, feat_types)
    text_emb = np.load("/content/drive/MyDrive/EEG2Video_data/processed/BLIP_embeddings/BLIP_embeddings.npy")  # (7,40,5,77,768)
    run_cv(subj_name, fusion, feat_types, eeg_feats, text_emb, device,
           lambda_cls=1.0, lambda_sem=0.5, lambda_exp=0.5)

if __name__ == "__main__":
    main()
