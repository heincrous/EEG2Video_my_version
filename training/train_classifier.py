# ==========================================
# train_classification.py
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
            eeg_clip = torch.tensor(eeg_clip, dtype=torch.float32).unsqueeze(0)
        else:
            eeg_clip = self.subj_data["EEG_segments"][b,c,k]
            eeg_clip = torch.tensor(eeg_clip, dtype=torch.float32).unsqueeze(0)
        return eeg_clip, c  # return EEG + class ID

# ==========================================
# Classifier Head
# ==========================================
class EEGClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=[1024,512], num_classes=40):
        super().__init__()
        layers=[]
        prev=input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev,h), nn.ReLU()]
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

    mode_choice=input("Select mode (train/dry): ").strip()
    feature_type=input("Select feature type (windows/segments): ").strip()

    encoders_all={
        "shallownet": shallownet,
        "deepnet": deepnet,
        "eegnet": eegnet,
        "tsconv": tsconv,
        "conformer": conformer
    }
    if feature_type=="windows":
        valid_encoders={k:v for k,v in encoders_all.items() if k!="deepnet"}
        C,T=62,100
    elif feature_type=="segments":
        valid_encoders=encoders_all
        C,T=62,400
    else:
        raise ValueError("Invalid feature type")

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Dry run ===
    if mode_choice=="dry":
        eeg=torch.randn(1,1,C,T).to(device)
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
        print(f"Available encoders for {feature_type}: {list(valid_encoders.keys())}")
        encoder_choice=input("Select encoder: ").strip()
        epochs_choice=int(input("Enter number of epochs: "))

        if encoder_choice not in valid_encoders:
            raise ValueError("Invalid encoder choice")

        EncoderClass=valid_encoders[encoder_choice]
        encoder=EncoderClass(out_dim=512,C=C,T=T).to(device)
        classifier=EEGClassifier(input_dim=512).to(device)

        train_ds=EEGClassDataset(bundle_file,subject_id,feature_type=feature_type,train=True)
        train_loader=DataLoader(train_ds,batch_size=256,shuffle=True)

        criterion=nn.CrossEntropyLoss()
        optimizer=optim.Adam(list(encoder.parameters())+list(classifier.parameters()),lr=5e-4)

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

if __name__=="__main__":
    main()
