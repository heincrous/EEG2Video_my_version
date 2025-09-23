import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import joblib
from einops import rearrange
import imageio
from diffusers.models import AutoencoderKL

# ------------------------------------------------
# Model components (same as training)
# ------------------------------------------------
class MyEEGNet_embedding(nn.Module):
    def __init__(self, d_model=128, C=62, T=100, F1=16, D=4, F2=16, cross_subject=False):
        super().__init__()
        self.drop_out = 0.25 if cross_subject else 0.5
        self.block_1 = nn.Sequential(nn.ZeroPad2d((31,32,0,0)), nn.Conv2d(1,F1,(1,64),bias=False), nn.BatchNorm2d(F1))
        self.block_2 = nn.Sequential(nn.Conv2d(F1,F1*D,(C,1),groups=F1,bias=False), nn.BatchNorm2d(F1*D), nn.ELU(),
                                     nn.AvgPool2d((1,4)), nn.Dropout(self.drop_out))
        self.block_3 = nn.Sequential(nn.ZeroPad2d((7,8,0,0)),
                                     nn.Conv2d(F1*D,F1*D,(1,16),groups=F1*D,bias=False),
                                     nn.Conv2d(F1*D,F2,(1,1),bias=False), nn.BatchNorm2d(F2), nn.ELU(),
                                     nn.AvgPool2d((1,8)), nn.Dropout(self.drop_out))
        self.embedding = nn.Linear(48, d_model)
    def forward(self, x):
        x = self.block_1(x); x = self.block_2(x); x = self.block_3(x)
        x = x.view(x.shape[0], -1)
        return self.embedding(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2)*-(math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position*div_term); pe[:,1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    def forward(self,x):
        return self.dropout(x + self.pe[:,:x.size(1)].requires_grad_(False))

class myTransformer(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.img_embedding = nn.Linear(4*36*64, d_model)
        self.eeg_embedding = MyEEGNet_embedding(d_model=d_model,C=62,T=100)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model,nhead=4,batch_first=True), num_layers=2)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model,nhead=4,batch_first=True), num_layers=4)
        self.positional_encoding = PositionalEncoding(d_model,dropout=0)
        self.txtpredictor = nn.Linear(512,13)
        self.predictor = nn.Linear(512,4*36*64)
    def forward(self, src, tgt):
        src = self.eeg_embedding(src.reshape(src.shape[0]*src.shape[1],1,62,100))
        src = src.reshape(src.shape[0]//7,7,-1)
        tgt = tgt.reshape(tgt.shape[0],tgt.shape[1],-1); tgt = self.img_embedding(tgt)
        src = self.positional_encoding(src); tgt = self.positional_encoding(tgt)
        enc = self.transformer_encoder(src)
        new_tgt = torch.zeros((tgt.shape[0],1,tgt.shape[2])).to(tgt.device)
        for i in range(6):
            dec = self.transformer_decoder(new_tgt,enc)
            new_tgt = torch.cat((new_tgt,dec[:,-1:,:]),dim=1)
        enc = torch.mean(enc,dim=1)
        return self.txtpredictor(enc), self.predictor(new_tgt).reshape(
            new_tgt.shape[0],new_tgt.shape[1],4,36,64)

# ------------------------------------------------
# Inference
# ------------------------------------------------
if __name__ == "__main__":
    base_dir = "/content/drive/MyDrive/EEG2Video_data/processed/"
    ckpt_dir = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_checkpoints/"
    out_dir = "/content/drive/MyDrive/EEG2Video_outputs/test_seq2seq/"
    vae_dir = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
    os.makedirs(out_dir, exist_ok=True)

    # list checkpoints
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if not ckpts:
        raise RuntimeError("No checkpoints found in seq2seq_checkpoints/")
    print("\nAvailable checkpoints:")
    for idx, ck in enumerate(ckpts):
        print(f"{idx}: {ck}")
    choice = int(input("\nSelect subject index: "))
    ckpt_path = os.path.join(ckpt_dir, ckpts[choice])
    subj = ckpts[choice].replace("seq2seqmodel_","").replace(".pt","")
    print(f"Using checkpoint for subject: {subj}")

    scaler_path = os.path.join(ckpt_dir, f"scaler_{subj}.pkl")
    scaler = joblib.load(scaler_path)

    # load model
    model = myTransformer().cuda()
    state = torch.load(ckpt_path,map_location="cuda")
    model.load_state_dict(state['state_dict'])
    model.eval()

    # pick a test EEG file
    with open(os.path.join(base_dir,"EEG_windows/test_list.txt")) as f:
        eeg_test_all = [l.strip() for l in f]
    eeg_files = [e for e in eeg_test_all if e.startswith(subj)]
    eeg_file = random.choice(eeg_files)
    print(f"Testing with {eeg_file}")

    eeg = np.load(os.path.join(base_dir,"EEG_windows",eeg_file))
    eeg_flat = scaler.transform(eeg.reshape(-1,62*100))
    eeg = eeg_flat.reshape(eeg.shape)
    eeg = torch.from_numpy(eeg).unsqueeze(0).float().cuda()  # [1,7,62,100]

    # dummy tgt of zeros for autoregressive decoding
    b = 1
    padded_video = torch.zeros((b,1,4,36,64)).cuda()
    _, pred_latents = model(eeg, padded_video)
    pred_latents = pred_latents[:,1:,:,:,:].squeeze(0).cpu().detach().numpy()  # [6,4,36,64]

    # decode latents with VAE (local version from Drive)
    vae = AutoencoderKL.from_pretrained(vae_dir, subfolder="vae").cuda()
    latents_t = torch.from_numpy(pred_latents).float().cuda()
    latents_t = latents_t / 0.18215
    with torch.no_grad():
        frames = vae.decode(latents_t).sample  # [6,3,H,W]
    frames = (frames.clamp(-1,1)+1)/2
    frames = frames.permute(0,2,3,1).cpu().numpy()*255
    frames = frames.astype(np.uint8)

    # save mp4
    class_clip = os.path.basename(eeg_file).replace(".npy",".mp4")
    save_path = os.path.join(out_dir,class_clip)
    imageio.mimsave(save_path,frames,fps=3)
    print(f"Saved generated video to {save_path}")

    # show caption
    blip_text_path = os.path.join(base_dir,"BLIP_text", "/".join(eeg_file.split("/")[1:]).replace(".npy",".txt"))
    with open(blip_text_path,"r") as f:
        caption = f.readline().strip()
    print(f"Ground-truth caption: {caption}")
