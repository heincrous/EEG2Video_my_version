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
# '''
# Description: EEG2Video inference script with Seq2Seq + Semantic Predictor + Diffusion (DANA)
# Author: Heinrich Crous (adapted from Zhou Tianyi)
# LastEditTime: 2025-09-21
# '''

import os
import sys
import torch
import numpy as np
import imageio
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------- Add repo root ----------------
repo_root = "/content/EEG2Video_my_version"
sys.path.append(repo_root)

# ---------------- Imports ----------------
from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from core_files.unet import UNet3DConditionModel
from training.my_autoregressive_transformer import myTransformer, EEGVideoDataset
from training.train_semantic_predictor import SemanticPredictor

# ---------------- Paths & configs ----------------
BASE = "/content/drive/MyDrive/EEG2Video_data/processed/Split_4train1test"
SEQ2SEQ_CKPT = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_checkpoint.pt"
SEMANTIC_CKPT = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_predictor.pt"
DIFFUSION_DIR = "/content/drive/MyDrive/EEG2Video_checkpoints/EEG2Video_diffusion_output"
BLIP_CAP_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_captions"
TEST_VIDEO_DIR = os.path.join(BASE, "test/Video_latents")
DE_TEST_DIR = os.path.join(BASE, "test/EEG_features/DE_1per2s")  # DE features for test
SAVE_DIR = "/content/drive/MyDrive/EEG2Video_inference"
os.makedirs(SAVE_DIR, exist_ok=True)

VIDEO_LENGTH = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Inline GIF saving function ----------------
def save_videos_grid(videos, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(videos, torch.Tensor):
        videos = videos.cpu().numpy()
    if videos.ndim == 5:  # [B,F,C,H,W]
        videos = videos[0]  # only first video
    frames = []
    for frame in videos:
        if frame.shape[0] == 1:
            frame = frame.squeeze(0)
        else:
            frame = frame.transpose(1,2,0)
        frames.append(np.clip(frame*255, 0, 255).astype(np.uint8))
    imageio.mimsave(path, frames, fps=4)

# ---------------- Check DE test dataset ----------------
if not os.path.exists(DE_TEST_DIR):
    raise RuntimeError(f"DE test folder does not exist: {DE_TEST_DIR}")

# ---------------- Load test dataset ----------------
test_ds = EEGVideoDataset(DE_TEST_DIR, TEST_VIDEO_DIR)
if len(test_ds) == 0:
    raise RuntimeError(f"No DE feature files found in {DE_TEST_DIR}")

test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

# ---------------- Load Seq2Seq model ----------------
seq2seq_model = myTransformer().to(device)
seq2seq_state = torch.load(SEQ2SEQ_CKPT, map_location=device)
seq2seq_model.load_state_dict(seq2seq_state["state_dict"])
seq2seq_model.eval()

# ---------------- Load Semantic Predictor ----------------
semantic_model = SemanticPredictor(input_dim=310).to(device)
semantic_state = torch.load(SEMANTIC_CKPT, map_location=device)
semantic_model.load_state_dict(semantic_state["state_dict"])
semantic_model.eval()

# ---------------- Load diffusion model (DANA) ----------------
unet = UNet3DConditionModel.from_pretrained_2d(DIFFUSION_DIR, subfolder="unet").to(device)
pipe = TuneAVideoPipeline.from_pretrained(DIFFUSION_DIR, unet=unet)
pipe.to(device)
pipe.enable_vae_slicing()
pipe.enable_attention_slicing()

# ---------------- Collect test BLIP captions ----------------
test_captions = []
for block in sorted(os.listdir(BLIP_CAP_DIR)):
    blip_block_dir = os.path.join(BLIP_CAP_DIR, block)
    test_block_dir = os.path.join(TEST_VIDEO_DIR, block)
    if not os.path.exists(blip_block_dir) or not os.path.exists(test_block_dir):
        continue
    test_clips = [f.replace(".npy",".txt") for f in os.listdir(test_block_dir) if f.endswith(".npy")]
    for txt_file in sorted(os.listdir(blip_block_dir)):
        if txt_file in test_clips:
            test_captions.append((block, txt_file, os.path.join(blip_block_dir, txt_file)))
    if test_captions:
        break

print(f"Testing 1 caption with video_length={VIDEO_LENGTH}")

# ---------------- Run inference ----------------
block, txt_file, txt_path = test_captions[0]
with open(txt_path, "r") as f:
    prompt_text = f.read().strip()

for i, (eeg, vid) in enumerate(tqdm(test_loader, desc="Running EEG2Video inference")):
    # ---------------- Flatten DE features to match training ----------------
    eeg_flat = eeg.reshape(eeg.shape[0], -1).to(device)  # [B, 310]

    vid = vid.to(device)

    # ---------------- Generate Seq2Seq latent ----------------
    b, f, c, h, w = vid.shape
    padded = torch.zeros((b, 1, c, h, w), device=device)
    full_vid = torch.cat((padded, vid), dim=1)
    with torch.no_grad():
        _, seq2seq_latent = seq2seq_model(eeg_flat, full_vid)
    seq2seq_latent = seq2seq_latent[:, :-1, :]

    # ---------------- Generate semantic embedding ----------------
    with torch.no_grad():
        semantic_embed = semantic_model(eeg_flat)  # [B, 77*768]
        semantic_embed = semantic_embed.view(eeg_flat.shape[0], 77, 768)  # reshape for DANA

    # ---------------- Diffusion inference ----------------
    with torch.no_grad():
        video_out = pipe(
            prompt_embeddings=semantic_embed,
            latents=seq2seq_latent.unsqueeze(2),
            video_length=VIDEO_LENGTH,
            height=288,
            width=512,
            num_inference_steps=50,
            guidance_scale=12.5
        ).videos

    # ---------------- Save GIF ----------------
    save_name = f"{block}_{txt_file.replace('.txt','.gif')}"
    save_path = os.path.join(SAVE_DIR, save_name)
    save_videos_grid(video_out, save_path)
    print(f"Saved sample {i} -> {save_path}")

    if i == 2:  # limit for quick test
        break



