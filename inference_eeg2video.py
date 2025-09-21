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
import torch
import numpy as np
import imageio
from torch.utils.data import DataLoader
from tqdm import tqdm
from training.my_autoregressive_transformer import myTransformer, EEGVideoDataset
from training.train_semantic_predictor import SemanticPredictor
from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from core_files.unet import UNet3DConditionModel
import shutil

# ---------------- Paths ----------------
BASE = "/content/drive/MyDrive/EEG2Video_data/processed/Split_4train1test/test"
DE_TEST_DIR = os.path.join(BASE, "EEG_features/DE_1per2s")
TEST_VIDEO_DIR = os.path.join(BASE, "Video_latents")
PROCESSED_GIF_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/Video_Gif/Block1"
SAVE_DIR = "/content/drive/MyDrive/EEG2Video_inference"
os.makedirs(SAVE_DIR, exist_ok=True)

SEQ2SEQ_CKPT = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_checkpoint.pt"
SEMANTIC_CKPT = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_predictor.pt"
DIFFUSION_UNET_DIR = "/content/drive/MyDrive/EEG2Video_checkpoints/EEG2Video_diffusion_output/unet"

VIDEO_LENGTH = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- GIF saving ----------------
def save_gif(frames, path, fps=4):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()
    gif_frames = []
    for f in frames:
        if f.ndim == 3 and f.shape[0] in [1,3]:
            f = f.squeeze(0).transpose(1,2,0) if f.shape[0] != 1 else f.squeeze(0)
        gif_frames.append((f*255).astype(np.uint8))
    imageio.mimsave(path, gif_frames, fps=fps)

# ---------------- Load dataset ----------------
test_ds = EEGVideoDataset(DE_TEST_DIR, TEST_VIDEO_DIR)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)

# ---------------- Load models ----------------
seq2seq_model = myTransformer().to(device)
seq2seq_state = torch.load(SEQ2SEQ_CKPT, map_location=device)
seq2seq_model.load_state_dict(seq2seq_state["state_dict"])
seq2seq_model.eval()

semantic_model = SemanticPredictor(input_dim=310).to(device)
semantic_state = torch.load(SEMANTIC_CKPT, map_location=device)
semantic_model.load_state_dict(semantic_state["state_dict"])
semantic_model.eval()

# UNet from diffusion checkpoint
unet = UNet3DConditionModel.from_pretrained_2d(DIFFUSION_UNET_DIR).to(device)
pipe = TuneAVideoPipeline.from_pretrained(
    os.path.dirname(DIFFUSION_UNET_DIR),
    unet=unet
)
pipe.to(device)
pipe.enable_vae_slicing()
pipe.enable_attention_slicing()

# ---------------- Run inference ----------------
for i, (eeg_raw, vid) in enumerate(tqdm(test_loader, desc="Running EEG2Video inference")):
    # ---------------- Seq2Seq input ----------------
    eeg_seq = eeg_raw.to(device)  # [B=1, segments=7, channels=62, timepoints=200]

    # ---------------- Semantic predictor input ----------------
    eeg_flat = eeg_raw[:,0,:,:5].reshape(eeg_raw.shape[0], -1).to(device)  # [B=1, 310]
    vid = vid.to(device)  # [B=1, frames=6, channels=4, H=36, W=64]

    # ---------------- Generate Seq2Seq latent ----------------
    b,f,c,h,w = vid.shape
    padded = torch.zeros((b,1,c,h,w), device=device)
    full_vid = torch.cat((padded, vid), dim=1)
    with torch.no_grad():
        _, seq2seq_latent = seq2seq_model(eeg_seq, full_vid)
        seq2seq_latent = seq2seq_latent[:, :-1, :]

    # ---------------- Generate semantic embedding ----------------
    with torch.no_grad():
        semantic_embed = semantic_model(eeg_flat).view(eeg_flat.shape[0],77,768)

    # ---------------- Trained GIF ----------------
    with torch.no_grad():
        trained_gif = pipe(
            prompt_embeddings=semantic_embed,
            latents=seq2seq_latent.unsqueeze(2),
            video_length=VIDEO_LENGTH,
            height=288,
            width=512,
            num_inference_steps=50,
            guidance_scale=12.5
        ).videos
    save_gif(trained_gif, os.path.join(SAVE_DIR, f"clip{i+1}_trained.gif"))

    # ---------------- Random GIF ----------------
    random_latent = torch.randn_like(seq2seq_latent).unsqueeze(2)
    random_embed = torch.randn_like(semantic_embed)
    with torch.no_grad():
        random_gif = pipe(
            prompt_embeddings=random_embed,
            latents=random_latent,
            video_length=VIDEO_LENGTH,
            height=288,
            width=512,
            num_inference_steps=50,
            guidance_scale=12.5
        ).videos
    save_gif(random_gif, os.path.join(SAVE_DIR, f"clip{i+1}_random.gif"))

    # ---------------- Ground-truth GIF ----------------
    gt_gif_files = [f for f in os.listdir(PROCESSED_GIF_DIR) if f.startswith("class") and f.endswith(".gif")]
    if gt_gif_files:
        gt_gif_path = os.path.join(PROCESSED_GIF_DIR, gt_gif_files[i % len(gt_gif_files)])
        shutil.copy(gt_gif_path, os.path.join(SAVE_DIR, f"clip{i+1}_ground_truth.gif"))

    if i >= 9:  # remove this to process all 40 clips
        break

print("Inference complete: random, trained, and ground-truth GIFs saved.")
