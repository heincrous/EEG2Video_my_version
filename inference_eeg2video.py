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
import os
import random
import numpy as np
import torch
import torch.nn as nn
import imageio
from einops import rearrange

from transformers import CLIPTokenizer
from diffusers import DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel

from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from core_files.unet import UNet3DConditionModel

# -------------------------
# Helper: save video tensor to GIF
# -------------------------
def save_videos_grid(videos, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(videos, torch.Tensor):
        videos = videos.detach().cpu().numpy()

    if videos.ndim == 5:  # [B, F, C, H, W]
        video = videos[0]
    else:
        video = videos
    frames = []
    for frame in video:
        if frame.shape[0] == 1:
            frame = frame.squeeze(0)
        else:
            frame = frame.transpose(1, 2, 0)
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        frames.append(frame)
    imageio.mimsave(path, frames, fps=5)

# -------------------------
# Paths
# -------------------------
BASE = "/content/drive/MyDrive/EEG2Video_data/processed/Split_4train1test/test"
GT_GIF_BASE = "/content/drive/MyDrive/EEG2Video_data/processed/Video_Gif"
CHECKPOINTS = "/content/drive/MyDrive/EEG2Video_checkpoints"

SEQ2SEQ_CKPT = os.path.join(CHECKPOINTS, "seq2seq_checkpoint.pt")
SEMANTIC_CKPT = os.path.join(CHECKPOINTS, "semantic_predictor.pt")
DIFFUSION_CKPT = os.path.join(CHECKPOINTS, "EEG2Video_diffusion_output")
SD_BASE = os.path.join(CHECKPOINTS, "stable-diffusion-v1-4")

OUTPUT_DIR = "/content/drive/MyDrive/EEG2Video_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load models
# -------------------------
# Diffusion pipeline
unet = UNet3DConditionModel.from_pretrained(DIFFUSION_CKPT, subfolder="unet").to(device, dtype=torch.float16)
vae = AutoencoderKL.from_pretrained(SD_BASE, subfolder="vae").to(device, dtype=torch.float16)
text_encoder = CLIPTextModel.from_pretrained(SD_BASE, subfolder="text_encoder").to(device, dtype=torch.float16)
tokenizer = CLIPTokenizer.from_pretrained(SD_BASE, subfolder="tokenizer")
scheduler = DDIMScheduler.from_pretrained(SD_BASE, subfolder="scheduler")

pipe = TuneAVideoPipeline(
    vae=vae,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler
).to("cuda")
pipe.enable_vae_slicing()

# Seq2Seq model
from training.my_autoregressive_transformer import myTransformer
seq2seq = myTransformer()
seq2seq.load_state_dict(torch.load(SEQ2SEQ_CKPT, map_location="cpu")["state_dict"])
seq2seq = seq2seq.to(device).eval()

# Semantic Predictor
from training.train_semantic_predictor import SemanticPredictor
semantic_model = SemanticPredictor(input_dim=310)
semantic_model.load_state_dict(torch.load(SEMANTIC_CKPT, map_location="cpu")["state_dict"])
semantic_model = semantic_model.to(device).eval()

# -------------------------
# Pick random test EEG file
# -------------------------
eeg_dir = os.path.join(BASE, "EEG_segments")
vid_dir = os.path.join(BASE, "Video_latents")

subj = random.choice(os.listdir(eeg_dir))  # e.g. "sub1"
block = random.choice(os.listdir(os.path.join(eeg_dir, subj)))  # e.g. "Block1"
clip_file = random.choice(os.listdir(os.path.join(eeg_dir, subj, block)))  # e.g. "class01_clip02.npy"

eeg_file = os.path.join(eeg_dir, subj, block, clip_file)
vid_file = os.path.join(vid_dir, block, clip_file)
gt_gif_file = os.path.join(GT_GIF_BASE, block, clip_file.replace(".npy", ".gif"))

print(f"Chosen test sample: subj={subj}, block={block}, clip={clip_file}")

# -------------------------
# Load EEG + Video latent
# -------------------------
eeg = np.load(eeg_file)  # (62, 400)
vid_latent = np.load(vid_file)  # (F, 4, 36, 64)

# Trim or pad video latents to exactly 4 frames
if vid_latent.shape[0] > 4:
    vid_latent = vid_latent[:4, :, :, :]
elif vid_latent.shape[0] < 4:
    pad = np.zeros((4 - vid_latent.shape[0], 4, 36, 64), dtype=vid_latent.dtype)
    vid_latent = np.concatenate([vid_latent, pad], axis=0)

# Segment EEG into 7 overlapping windows of 200 samples
segments = []
for start in range(0, eeg.shape[1] - 200 + 1, 100):
    segments.append(eeg[:, start:start+200])
while len(segments) < 7:
    segments.append(np.zeros((62, 200)))
eeg = np.stack(segments[:7], axis=0)  # shape (7, 62, 200)

eeg_tensor = torch.tensor(eeg, dtype=torch.float16).unsqueeze(0).to(device)  # [1, 7, 62, 200]

# -------------------------
# Run Seq2Seq → predicted latents
# -------------------------
# EEG tensor must be float32 for Seq2Seq
eeg_tensor = torch.tensor(eeg, dtype=torch.float32).unsqueeze(0).to(device)

# Init with 1 BOS + 5 empty slots = 6 steps, also float32 for Seq2Seq
init_latents = torch.zeros((1, 6, 4, 36, 64), device=device, dtype=torch.float32)

with torch.no_grad():
    _, pred_latents = seq2seq(eeg_tensor, init_latents)

# Drop the first dummy BOS frame
pred_latents = pred_latents[:, 1:, :, :, :]

# Convert to float16 for diffusion
pred_latents = pred_latents.to(torch.float16)

# Rearrange to [B, C, F, H, W]
pred_latents = rearrange(pred_latents, "b f c h w -> b c f h w")

# Trim/pad to exactly 4 for the diffusion UNet
if pred_latents.shape[2] > 4:
    pred_latents = pred_latents[:, :, :4, :, :]
elif pred_latents.shape[2] < 4:
    pad = torch.zeros((pred_latents.shape[0], pred_latents.shape[1], 4 - pred_latents.shape[2],
                       pred_latents.shape[3], pred_latents.shape[4]),
                      device=pred_latents.device, dtype=pred_latents.dtype)
    pred_latents = torch.cat((pred_latents, pad), dim=2)

# -------------------------
# Run Semantic predictor → embeddings
# -------------------------
# Path to DE features for this clip (310-dim expected after flattening)
de_dir = os.path.join(BASE, "EEG_features", "DE_1per2s", subj, block)
de_file = os.path.join(de_dir, clip_file)  # must exist in DE_1per2s

if not os.path.exists(de_file):
    raise FileNotFoundError(f"DE feature file not found: {de_file}")

# 1. Load DE features (could be (62,5) or (310,))
de_features = np.load(de_file)

# Flatten if needed
if de_features.ndim > 1:
    de_features = de_features.reshape(-1)

if de_features.shape[0] != 310:
    raise ValueError(f"Expected 310 features, got {de_features.shape}")

# 2. Convert to tensor [1,310]
eeg_de = torch.tensor(de_features, dtype=torch.float16).unsqueeze(0).to(device)

# 3. Run semantic predictor → [1,59136]
with torch.no_grad():
    eeg_embeds = semantic_model(eeg_de)

# -------------------------
# Run diffusion pipeline
# -------------------------
with torch.no_grad():
    eeg_embeds = eeg_embeds.to(torch.float16)
    neg_eeg = eeg_tensor.mean(dim=1, keepdim=True).to(torch.float16)

    video = pipe(
        None,                           # model
        eeg_embeds,                     # eeg
        negative_eeg=neg_eeg,           # negative eeg
        latents=pred_latents,           # seq2seq-predicted latents
        video_length=pred_latents.shape[1],
        height=288,
        width=512,
        num_inference_steps=50,
        guidance_scale=12.5
    ).videos

# -------------------------
# Save results
# -------------------------
gen_path = os.path.join(OUTPUT_DIR, f"{clip_file.replace('.npy','')}_gen.gif")
gt_path = os.path.join(OUTPUT_DIR, f"{clip_file.replace('.npy','')}_gt.gif")

save_videos_grid(video, gen_path)

# copy ground truth gif
if os.path.exists(gt_gif_file):
    import shutil
    shutil.copy(gt_gif_file, gt_path)

print(f"Generated video saved: {gen_path}")
print(f"Ground truth video saved: {gt_path}")
