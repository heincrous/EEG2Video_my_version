# ==========================================
# train_eval_classification.py
# Subject-specific EEG → Class (40-way)
# ==========================================

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter

# === Repo imports ===
repo_root = "/content/EEG2Video_my_version"
sys.path.append(repo_root)
from core_files.models import shallownet, deepnet, eegnet, tsconv, conformer

# ==========================================
# EEG Dataset (classification only)
# ==========================================
class EEGClassDataset(Dataset):
    def __init__(self, bundle_file, subject_id, feature_type="windows", train=True):
        bundle = np.load(bundle_file, allow_pickle=True)
        self.eeg_all = bundle["EEG_data"].item()
        self.feature_type = feature_type
        self.blocks = range(0,6) if train else [6]

        if subject_id not in self.eeg_all:
            raise ValueError(f"Subject {subject_id} not found in bundle")
        self.subj_data = self.eeg_all[subject_id]

        self.index = []
        for b in self.blocks:
            for c in range(40):
                for k in range(5):
                    if feature_type == "windows":
                        for w in range(7):
                            self.index.append((b,c,k,w))
                    else:
                        self.index.append((b,c,k,None))

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        b,c,k,w = self.index[idx]
        if self.feature_type == "windows":
            eeg_clip = self.subj_data["EEG_windows"][b,c,k][w]
        else:
            eeg_clip = self.subj_data["EEG_segments"][b,c,k]

        eeg_clip = torch.tensor(eeg_clip, dtype=torch.float32).unsqueeze(0)

        # z-score normalize per channel
        mean = eeg_clip.mean(dim=-1, keepdim=True)
        std = eeg_clip.std(dim=-1, keepdim=True) + 1e-6
        eeg_clip = (eeg_clip - mean) / std

        return eeg_clip, c

# ==========================================
# Classifier Head
# ==========================================
class EEGClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=[1024,512], num_classes=40, drop_p=0.5):
        super().__init__()
        layers=[]
        prev=input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev,h), nn.ReLU(), nn.Dropout(drop_p)]
            prev=h
        layers.append(nn.Linear(prev,num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

# ==========================================
# Main
# ==========================================
def main():
    drive_root="/content/drive/MyDrive/EEG2Video_data/processed"
    bundle_file=f"{drive_root}/BLIP_EEG_bundle.npz"

    mode_choice=input("Select mode (train/eval/dry): ").strip()

    encoders_all={
        "shallownet": shallownet,
        "deepnet": deepnet,
        "eegnet": eegnet,
        "tsconv": tsconv,
        "conformer": conformer
    }

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Dry run ===
    if mode_choice=="dry":
        for feature_type,(C,T) in [("windows",(62,100)),("segments",(62,400))]:
            eeg=torch.randn(1,1,C,T).to(device)
            valid_encoders = {k:v for k,v in encoders_all.items()} if feature_type=="segments" \
                             else {k:v for k,v in encoders_all.items() if k!="deepnet"}
            for name,EncoderClass in valid_encoders.items():
                encoder=EncoderClass(out_dim=512,C=C,T=T).to(device)
                feats=encoder(eeg)
                clf=EEGClassifier(input_dim=512).to(device)
                out=clf(feats)
                print(f"[{feature_type}] {name}: EEG {eeg.shape} -> Class logits {out.shape}")
        print("Dry run completed")
        return

    # === Training ===
    if mode_choice=="train":
        subject_id=input("Enter subject ID (e.g. sub1): ").strip()
        print("Available encoders: ", list(encoders_all.keys()))
        encoder_choice=input("Select encoder: ").strip()
        epochs_choice=int(input("Enter number of epochs: "))

        if encoder_choice not in encoders_all:
            raise ValueError("Invalid encoder choice")

        # infer feature type from encoder choice (deepnet only supports segments)
        feature_type="segments" if encoder_choice=="deepnet" else "windows"
        C,T=(62,400) if feature_type=="segments" else (62,100)

        EncoderClass=encoders_all[encoder_choice]
        encoder=EncoderClass(out_dim=512,C=C,T=T).to(device)
        classifier=EEGClassifier(input_dim=512).to(device)

        train_ds=EEGClassDataset(bundle_file,subject_id,feature_type=feature_type,train=True)
        train_loader=DataLoader(train_ds,batch_size=128,shuffle=True)

        criterion=nn.CrossEntropyLoss()
        optimizer=optim.Adam(list(encoder.parameters())+list(classifier.parameters()),lr=1e-4)

        # sanity check: label distribution
        labels=[c for _,c in [train_ds[i] for i in np.random.choice(len(train_ds),200)]]
        print("Sanity check (sampled labels):", Counter(labels))

        ckpt_dir="/content/drive/MyDrive/EEG2Video_checkpoints/classification_checkpoints"
        os.makedirs(ckpt_dir,exist_ok=True)

        for epoch in range(epochs_choice):
            total=0; correct=0; total_samples=0
            for eeg,cls in tqdm(train_loader,desc=f"Epoch {epoch+1}/{epochs_choice}"):
                eeg,cls=eeg.to(device),cls.to(device)
                optimizer.zero_grad()
                feats=encoder(eeg)
                logits=classifier(feats)
                loss=criterion(logits,cls)
                loss.backward(); optimizer.step()
                total+=loss.item()
                pred=logits.argmax(dim=1)
                correct+=(pred==cls).sum().item()
                total_samples+=cls.size(0)
            avg_loss=total/len(train_loader)
            acc=correct/total_samples
            print(f"Epoch {epoch+1}/{epochs_choice} Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

        ckpt_path=os.path.join(
            ckpt_dir,f"classifier_{subject_id}_{feature_type}_{encoder_choice}.pt"
        )
        torch.save({
            'epoch':epochs_choice,
            'encoder_state_dict':encoder.state_dict(),
            'classifier_state_dict':classifier.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'final_loss':avg_loss
        },ckpt_path)
        print(f"Final checkpoint saved to {ckpt_path}")
        return

    # === Evaluation ===
    if mode_choice=="eval":
        ckpt_dir="/content/drive/MyDrive/EEG2Video_checkpoints/classification_checkpoints"
        ckpts=[f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        for i,ck in enumerate(ckpts): print(f"[{i}] {ck}")
        choice=int(input("\nSelect checkpoint index: ").strip())
        ckpt_file=ckpts[choice]
        ckpt_path=os.path.join(ckpt_dir,ckpt_file)

        # Parse filename: classifier_sub1_windows_shallownet.pt
        parts=ckpt_file.replace(".pt","").split("_")
        subject_id,feature_type,encoder_choice=parts[1],parts[2],parts[3]
        print(f"\n[Config] Subject={subject_id}, Feature={feature_type}, Encoder={encoder_choice}")

        data=np.load(bundle_file,allow_pickle=True)
        eeg_dict=data["EEG_data"].item()

        C,T=(62,100) if feature_type=="windows" else (62,400)
        encoder=encoders_all[encoder_choice](out_dim=512,C=C,T=T).cuda()
        classifier=EEGClassifier(input_dim=512).cuda()

        ckpt=torch.load(ckpt_path,map_location="cuda")
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        classifier.load_state_dict(ckpt["classifier_state_dict"])
        encoder.eval(); classifier.eval()

        feats=eeg_dict[subject_id]
        eeg=feats[f"EEG_{feature_type}"][6]   # block 6 test

        correct1=correct5=total=0
        pred_classes=[]; true_classes=[]

        with tqdm(total=40*5,desc=f"Evaluating {subject_id}") as pbar:
            for ci in range(40):
                for cj in range(5):
                    if feature_type=="windows":
                        windows=torch.tensor(eeg[ci,cj],dtype=torch.float32).unsqueeze(1).cuda()
                        with torch.no_grad():
                            enc=encoder(windows)
                            logits=classifier(enc.mean(0,keepdim=True))
                    else:
                        x=torch.tensor(eeg[ci,cj],dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
                        with torch.no_grad():
                            enc=encoder(x)
                            logits=classifier(enc)

                    probs=torch.softmax(logits,dim=-1).cpu().numpy().flatten()
                    top1=int(np.argmax(probs))
                    top5=np.argsort(probs)[-5:][::-1]

                    correct1+=int(top1==ci)
                    correct5+=int(ci in top5)
                    total+=1
                    pred_classes.append(top1); true_classes.append(ci)
                    pbar.update(1)

        print(f"\nTop-1 Acc: {correct1/total:.4f}, Top-5 Acc: {correct5/total:.4f}")
        pc,tc=Counter(pred_classes),Counter(true_classes)
        print("\nPredicted class distribution:")
        for cls in range(40):
            print(f"Class {cls}: {pc.get(cls,0)} ({pc.get(cls,0)/total:.2%})")
        print("\nTrue class distribution:")
        for cls in range(40):
            print(f"Class {cls}: {tc.get(cls,0)} ({tc.get(cls,0)/total:.2%})")

if __name__=="__main__":
    main()
