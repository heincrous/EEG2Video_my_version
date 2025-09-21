# '''
# Description: 
# Author: Zhou Tianyi
# LastEditTime: 2025-04-11 12:10:33
# LastEditors:  
# '''
# from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
# from models.unet import UNet3DConditionModel
# from transformers import CLIPTextModel, CLIPTokenizer
# from tuneavideo.util import save_videos_grid,ddim_inversion
# import torch
# from models.train_semantic_predictor import CLIP
# import numpy as np
# from einops import rearrange
# from sklearn import preprocessing
# import random
# import math
# model = None
# torch.cuda.empty_cache()
# torch.cuda.ipc_collect()
# import os
# from diffusers.schedulers import (
#     DDPMScheduler,
#     DDIMScheduler,
# )
# torch.cuda.set_device(2)
# def seed_everything(seed=0, cudnn_deterministic=True):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     if cudnn_deterministic:
#         torch.backends.cudnn.deterministic = True
#     else:
#         ## needs to be False to use conv3D
#         print('Note: not using cudnn.deterministic')
# seed_everything(114514)

# eeg = torch.load('',map_location='cpu')

# negative = eeg.mean(dim=0)

# pretrained_model_path = "Zhoutianyi/huggingface/stable-diffusion-v1-4"
# my_model_path = "outputs/40_classes_200_epoch"

# unet = UNet3DConditionModel.from_pretrained(my_model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')
# pipe = TuneAVideoPipeline.from_pretrained(pretrained_model_path ,unet=unet, torch_dtype=torch.float16).to("cuda")
# pipe.enable_xformers_memory_efficient_attention()
# pipe.enable_vae_slicing()



# latents = np.load('Seq2Seq/latent_out_block7_40_classes.npy')
# latents = torch.from_numpy(latents).half()
# latents = rearrange(latents, 'a b c d e -> a c b d e')
# latents = latents.to('cuda')

# latents_add_noise = torch.load('DANA/40_classes_latent_add_noise.pt')
# latents_add_noise = latents_add_noise.half()
# latents_add_noise = rearrange(latents_add_noise, 'a b c d e -> a c b d e')
# latents_add_noise = latents_add_noise.to('cuda')
# # Ablation, inference w/o Seq2Seq and w/o DANA
# woSeq2Seq = False
# woDANA = False
# for i in range(len(eeg)):
#     if woSeq2Seq:
#         video = pipe(model, eeg[i:i+1,...],negative_eeg=negative, latents=None, video_length=6, height=288, width=512, num_inference_steps=100, guidance_scale=12.5).videos
#         savename = f'40_Classes_woSeq2Seq/EEG2Video/'
#         import os
#         os.makedirs(savename, exist_ok=True)
#     elif woDANA:
#         video = pipe(model, eeg[i:i+1,...],negative_eeg=negative, latents=latents[i:i+1,...], video_length=6, height=288, width=512, num_inference_steps=100, guidance_scale=12.5).videos
#         savename = f'40_Classes_woDANA/EEG2Video/'
#         import os
#         os.makedirs(savename, exist_ok=True)
#     else:
#         video = pipe(model, eeg[i:i+1,...],negative_eeg=negative, latents=latents_add_noise[i:i+1,...], video_length=6, height=288, width=512, num_inference_steps=100, guidance_scale=12.5).videos
#         savename = f'40_Classes_Fullmodel/EEG2Video/'
#         import os
#         os.makedirs(savename, exist_ok=True)
#     save_videos_grid(video, f"./{savename}/{i}.gif")
 
# ---------------------------------------------------------------------------------------------------------------
# NEW VERSION
# ---------------------------------------------------------------------------------------------------------------
import os, random, shutil, numpy as np, torch, imageio

# ---------------- Paths ----------------
BASE = "/content/drive/MyDrive/EEG2Video_data/processed/Split_4train1test/test"
EEG_SEG_DIR = os.path.join(BASE, "EEG_segments")
GT_GIF_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/Video_Gif/Block1"
SAVE_DIR = "/content/drive/MyDrive/EEG2Video_inference"
os.makedirs(SAVE_DIR, exist_ok=True)

SEQ2SEQ_CKPT = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_checkpoint.pt"
SEMANTIC_CKPT = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_predictor.pt"
DIFFUSION_UNET_DIR = "/content/drive/MyDrive/EEG2Video_checkpoints/EEG2Video_diffusion_output/unet"
SD_BASE_DIR = os.path.dirname(DIFFUSION_UNET_DIR)

VIDEO_LENGTH, HEIGHT, WIDTH = 6, 288, 512
STEPS, GUIDANCE = 50, 12.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Utils ----------------
def save_gif(frames, path, fps=5):
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()
    out = []
    for f in frames:
        f = f.squeeze(0) if f.ndim==3 and f.shape[0]==1 else (f.transpose(1,2,0) if f.ndim==3 else f)
        out.append(np.clip(f*255.0,0,255).astype(np.uint8))
    imageio.mimsave(path, out, fps=fps)

def pick_valid_7x62x200(root):
    subs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))]
    random.shuffle(subs)
    for sub in subs:
        subp = os.path.join(root, sub)
        blocks = [b for b in os.listdir(subp) if os.path.isdir(os.path.join(subp,b))]
        random.shuffle(blocks)
        for blk in blocks:
            blkp = os.path.join(subp, blk)
            files = [f for f in os.listdir(blkp) if f.endswith(".npy")]
            random.shuffle(files)
            for f in files:
                p = os.path.join(blkp, f)
                try:
                    arr = np.load(p, mmap_mode="r")
                except Exception:
                    continue
                if arr.ndim==3 and arr.shape==(7,62,200):
                    return sub, blk, p
    raise RuntimeError("No EEG file with shape (7,62,200) found.")

# ---------------- Select one clip ----------------
sub, block, EEG_PATH = pick_valid_7x62x200(EEG_SEG_DIR)
print(f"Selected: Subject={sub}, Block={block}, File={os.path.basename(EEG_PATH)}")
eeg_raw = torch.from_numpy(np.load(EEG_PATH)).unsqueeze(0).to(device)  # [1,7,62,200]
neg_raw = eeg_raw.mean(dim=1, keepdim=True)                            # [1,1,62,200]

# ---------------- Load Seq2Seq for embeddings ----------------
from training.my_autoregressive_transformer import myTransformer
seq2seq = myTransformer().to(device)
ckpt = torch.load(SEQ2SEQ_CKPT, map_location=device)
seq2seq.load_state_dict(ckpt["state_dict"])
seq2seq.eval()
with torch.no_grad():
    emb = seq2seq(eeg_raw)            # [1,77,768]
    neg_emb = seq2seq(neg_raw)        # [1,77,768]

# ---------------- Optional Semantic Predictor Ablation ----------------
from training.train_semantic_predictor import SemanticPredictor
de = np.load(EEG_PATH).reshape(-1)[None, :]  # flatten 62*5 -> [1,310]
sem = SemanticPredictor(input_dim=310).to(device).eval()
sem_sd = torch.load(SEMANTIC_CKPT, map_location=device)["state_dict"]
sem.load_state_dict(sem_sd)
with torch.no_grad():
    emb_sem = sem(torch.from_numpy(de).float().to(device))   # [1,59136]
neg_sem = torch.zeros_like(emb_sem)

# ---------------- Load diffusion UNet and pipeline ----------------
from core_files.unet import UNet3DConditionModel
from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
unet = UNet3DConditionModel.from_pretrained_2d(DIFFUSION_UNET_DIR).to(device)
pipe = TuneAVideoPipeline.from_pretrained(SD_BASE_DIR, unet=unet)
pipe.to(device)
pipe.enable_vae_slicing()
pipe.enable_attention_slicing()

# ---------------- Trained GIF (Seq2Seq embeddings) ----------------
with torch.no_grad():
    trained = pipe(
        model=None,
        eeg=emb.reshape(1,-1),          # flatten to [1,59136] to match pipeline reshape
        negative_eeg=neg_emb.reshape(1,-1),
        latents=None,
        video_length=VIDEO_LENGTH,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE
    ).videos
save_gif(trained, os.path.join(SAVE_DIR, "clip_trained.gif"))

# ---------------- Random GIF ----------------
with torch.no_grad():
    rand_emb = torch.randn_like(emb).reshape(1,-1)
    random_vid = pipe(
        model=None,
        eeg=rand_emb,
        negative_eeg=neg_emb.reshape(1,-1),
        latents=None,
        video_length=VIDEO_LENGTH,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE
    ).videos
save_gif(random_vid, os.path.join(SAVE_DIR, "clip_random.gif"))

# ---------------- Semantic Predictor Ablation GIF ----------------
with torch.no_grad():
    sem_vid = pipe(
        model=None,
        eeg=emb_sem,       # Semantic Predictor output [1,59136]
        negative_eeg=neg_sem,
        latents=None,
        video_length=VIDEO_LENGTH,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE
    ).videos
save_gif(sem_vid, os.path.join(SAVE_DIR, "clip_semantic.gif"))

# ---------------- Ground-truth GIF ----------------
if os.path.isdir(GT_GIF_DIR):
    gts = [f for f in os.listdir(GT_GIF_DIR) if f.endswith(".gif")]
    if gts:
        shutil.copy(os.path.join(GT_GIF_DIR, random.choice(gts)),
                    os.path.join(SAVE_DIR, "clip_ground_truth.gif"))

print("Saved: clip_trained.gif, clip_random.gif, clip_semantic.gif, clip_ground_truth.gif")
