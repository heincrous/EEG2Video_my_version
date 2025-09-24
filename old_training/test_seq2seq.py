import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import joblib
import imageio
from diffusers.models import AutoencoderKL
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchmetrics

# ------------------------------------------------
# Import CONFIG from training
# ------------------------------------------------
from train_seq2seq import CONFIG

# ------------------------------------------------
# Model components (same as training)
# ------------------------------------------------
class MyEEGNet_embedding(nn.Module):
    def __init__(self, d_model, C, T, F1, D, F2, cross_subject=False):
        super().__init__()
        self.drop_out = 0.25 if cross_subject else 0.5
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((31,32,0,0)),
            nn.Conv2d(1,F1,(1,64),bias=False),
            nn.BatchNorm2d(F1)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(F1,F1*D,(C,1),groups=F1,bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1,4)),
            nn.Dropout(self.drop_out)
        )
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7,8,0,0)),
            nn.Conv2d(F1*D,F1*D,(1,16),groups=F1*D,bias=False),
            nn.Conv2d(F1*D,F2,(1,1),bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1,8)),
            nn.Dropout(self.drop_out)
        )
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
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self,x):
        return self.dropout(x + self.pe[:,:x.size(1)].requires_grad_(False))


class myTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.img_embedding = nn.Linear(4*36*64, cfg["d_model"])
        self.eeg_embedding = MyEEGNet_embedding(
            d_model=cfg["d_model"], C=cfg["C"], T=cfg["T"],
            F1=cfg["F1"], D=cfg["D"], F2=cfg["F2"],
            cross_subject=cfg["cross_subject"]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=cfg["d_model"],
                                       nhead=cfg["nhead"],
                                       batch_first=True),
            num_layers=cfg["encoder_layers"]
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=cfg["d_model"],
                                       nhead=cfg["nhead"],
                                       batch_first=True),
            num_layers=cfg["decoder_layers"]
        )
        self.positional_encoding = PositionalEncoding(cfg["d_model"],dropout=0)
        self.predictor = nn.Linear(cfg["d_model"], 4*36*64)

    def forward(self, src, num_frames):
        src = self.eeg_embedding(src.reshape(src.shape[0]*src.shape[1],1,62,100))
        src = src.reshape(src.shape[0]//7,7,-1)
        src = self.positional_encoding(src)
        memory = self.transformer_encoder(src)

        b = src.shape[0]
        outputs = []
        prev_tokens = torch.zeros((b,1,4*36*64), device=src.device)
        for t in range(num_frames):
            tgt_emb = self.img_embedding(prev_tokens)
            tgt_emb = self.positional_encoding(tgt_emb)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(1)).to(tgt_emb.device)
            out = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            pred = self.predictor(out[:, -1])
            outputs.append(pred)
            prev_tokens = torch.cat([prev_tokens, pred.unsqueeze(1)], dim=1)
        outputs = torch.stack(outputs, dim=1)
        return outputs.view(b, num_frames, 4, 36, 64)


# ------------------------------------------------
# Inference using bundled .npz + Metrics
# ------------------------------------------------
if __name__ == "__main__":
    bundle_dir = CONFIG["bundle_dir"]
    ckpt_dir   = os.path.join(CONFIG["save_root"], "seq2seq_checkpoints")
    out_dir    = os.path.join(CONFIG["save_root_output"], "test_seq2seq")
    vae_dir    = CONFIG.get("vae_dir", None)
    os.makedirs(out_dir, exist_ok=True)

    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if not ckpts:
        raise RuntimeError("No checkpoints found in seq2seq_checkpoints/")
    print("\nAvailable checkpoints:")
    for idx, ck in enumerate(ckpts):
        print(f"{idx}: {ck}")
    choice = int(input("\nSelect checkpoint index: "))
    ckpt_path = os.path.join(ckpt_dir, ckpts[choice])
    ckname = ckpts[choice]

    # Load global scaler for this checkpoint
    tag = ckname.replace("seq2seqmodel_","").replace(".pt","")
    scaler_path = os.path.join(ckpt_dir, f"scaler_{tag}.pkl")
    scaler = joblib.load(scaler_path)

    # Find matching test subjects
    if tag == "all":
        test_paths = [os.path.join(bundle_dir,f) for f in os.listdir(bundle_dir) if f.endswith("_test.npz")]
    else:
        subs = tag.split("_")
        test_paths = [os.path.join(bundle_dir, f"{s}_test.npz") for s in subs]

    # Load model
    model = myTransformer(CONFIG).cuda()
    state = torch.load(ckpt_path,map_location="cuda")
    model.load_state_dict(state['state_dict'])
    model.eval()

    eeg_all, vids_all, caps_all, subj_list = [], [], [], []
    for p in test_paths:
        d = np.load(p, allow_pickle=True)
        eeg_all.append(d["EEG_windows"])
        vids_all.append(d["Video_latents"])
        if "BLIP_text" in d:
            caps_all.extend(d["BLIP_text"])
        else:
            caps_all.extend(["[NO CAPTION AVAILABLE]"] * len(d["EEG_windows"]))
        subj_name = os.path.basename(p).replace("_test.npz","")
        subj_list.extend([subj_name]*len(d["EEG_windows"]))
    eeg_all = np.concatenate(eeg_all, axis=0)
    vids_all = np.concatenate(vids_all, axis=0)

    idx = random.randint(0, len(eeg_all)-1)
    eeg = eeg_all[idx]
    gt_latents = vids_all[idx]
    caption = caps_all[idx]
    subj_for_sample = subj_list[idx]
    print(f"Testing sample {idx} from {subj_for_sample}")
    print(f"Caption: {caption}")

    # Apply global scaler
    eeg_flat = scaler.transform(eeg.reshape(-1,62*100))
    eeg = eeg_flat.reshape(eeg.shape)
    eeg = torch.from_numpy(eeg).unsqueeze(0).float().cuda()

    num_frames = gt_latents.shape[0]
    pred_latents = model(eeg, num_frames)
    pred_latents = pred_latents.squeeze(0).cpu().detach().numpy()

    if vae_dir is None:
        raise RuntimeError("vae_dir must be set in CONFIG for decoding")
    vae = AutoencoderKL.from_pretrained(vae_dir, subfolder="vae").cuda()

    # Decode prediction
    latents_t = torch.from_numpy(pred_latents).float().cuda() / 0.18215
    with torch.no_grad():
        frames = vae.decode(latents_t).sample
    frames = (frames.clamp(-1, 1) + 1) / 2
    frames = frames.permute(0, 2, 3, 1).cpu().numpy() * 255
    frames = frames.astype(np.uint8)

    # Decode ground truth
    gt_latents_t = torch.from_numpy(gt_latents).float().cuda() / 0.18215
    with torch.no_grad():
        gt_frames = vae.decode(gt_latents_t).sample
    gt_frames = (gt_frames.clamp(-1, 1) + 1) / 2
    gt_frames = gt_frames.permute(0, 2, 3, 1).cpu().numpy() * 255
    gt_frames = gt_frames.astype(np.uint8)

    # Metrics
    ssim_scores, psnr_scores = [], []
    for f in range(min(len(frames), len(gt_frames))):
        ssim_scores.append(ssim(frames[f], gt_frames[f], channel_axis=2, data_range=255))
        psnr_scores.append(psnr(frames[f], gt_frames[f], data_range=255))
    print(f"Mean SSIM: {np.mean(ssim_scores):.4f}")
    print(f"Mean PSNR: {np.mean(psnr_scores):.2f} dB")

    lpips_metric = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='alex').cuda()
    gen_t = torch.from_numpy(frames).permute(0,3,1,2).float().cuda() / 255.0
    gt_t  = torch.from_numpy(gt_frames).permute(0,3,1,2).float().cuda() / 255.0
    with torch.no_grad():
        lpips_score = lpips_metric(gen_t, gt_t)
    print(f"LPIPS: {lpips_score.item():.4f}")

    # Save predicted and ground truth videos
    pred_path = os.path.join(out_dir, f"{subj_for_sample}_sample{idx}_pred.mp4")
    gt_path   = os.path.join(out_dir, f"{subj_for_sample}_sample{idx}_gt.mp4")
    imageio.mimsave(pred_path, frames, fps=3)
    imageio.mimsave(gt_path, gt_frames, fps=3)
    print(f"Saved predicted video to {pred_path}")
    print(f"Saved ground-truth video to {gt_path}")