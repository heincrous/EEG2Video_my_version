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
from einops import rearrange
import shutil
import random

from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from core_files.unet import UNet3DConditionModel
from training.my_autoregressive_transformer import myTransformer

# ----------------------- Paths -----------------------
TEST_BASE = "/content/drive/MyDrive/EEG2Video_data/processed/Split_4train1test/test"
PROCESSED_GIF_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/Video_Gif"
SAVE_DIR = "/content/drive/MyDrive/EEG2Video_inference"
os.makedirs(SAVE_DIR, exist_ok=True)

SEQ2SEQ_CKPT = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_checkpoint.pt"
DIFFUSION_UNET_DIR = "/content/drive/MyDrive/EEG2Video_checkpoints/EEG2Video_diffusion_output/unet"

VIDEO_LENGTH = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------- Pick a random test clip -----------------------
all_blocks = [b for b in os.listdir(TEST_BASE) if os.path.isdir(os.path.join(TEST_BASE, b))]
random_block = random.choice(all_blocks)
block_path = os.path.join(TEST_BASE, random_block)

all_classes = [c for c in os.listdir(block_path) if os.path.isdir(os.path.join(block_path, c))]
random_class = random.choice(all_classes)
class_path = os.path.join(block_path, random_class)

# Pick first available .npy files inside the class folder
eeg_files = [f for f in os.listdir(os.path.join(class_path, "EEG_segments")) if f.endswith(".npy")]
video_files = [f for f in os.listdir(os.path.join(class_path, "Video_latents")) if f.endswith(".npy")]
latents_files = [f for f in os.listdir(os.path.join(class_path, "latents_add_noise")) if f.endswith(".pt")]

TEST_EEG_PATH = os.path.join(class_path, "EEG_segments", random.choice(eeg_files))
LATENTS_NOISE_PATH = os.path.join(class_path, "latents_add_noise", random.choice(latents_files))

print(f"Selected random clip: Block={random_block}, Class={random_class}")
print(f"EEG path: {TEST_EEG_PATH}")
print(f"Latents path: {LATENTS_NOISE_PATH}")

# ----------------------- GIF saving -----------------------
def save_gif(frames, path, fps=5):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()
    gif_frames = []
    for f in frames:
        if f.shape[0] in [1,3]:
            f = f.squeeze(0).transpose(1,2,0) if f.shape[0] != 1 else f.squeeze(0)
        gif_frames.append((f*255).astype(np.uint8))
    imageio.mimsave(path, gif_frames, fps=fps)

# ----------------------- Load models -----------------------
seq2seq_model = myTransformer().to(device)
seq2seq_state = torch.load(SEQ2SEQ_CKPT, map_location=device)
seq2seq_model.load_state_dict(seq2seq_state["state_dict"])
seq2seq_model.eval()

unet = UNet3DConditionModel.from_pretrained_2d(DIFFUSION_UNET_DIR).to(device)
pipe = TuneAVideoPipeline.from_pretrained(
    os.path.dirname(DIFFUSION_UNET_DIR),
    unet=unet
)
pipe.to(device)
pipe.enable_vae_slicing()
pipe.enable_attention_slicing()

# ----------------------- Load EEG and negative -----------------------
eeg_raw = torch.from_numpy(np.load(TEST_EEG_PATH)).unsqueeze(0).to(device)
negative_eeg = eeg_raw.mean(dim=0, keepdim=True)

# ----------------------- Load latents (DANA) -----------------------
latents_add_noise = torch.load(LATENTS_NOISE_PATH).half()
latents_add_noise = rearrange(latents_add_noise, 'a b c d e -> a c b d e').to(device)

# ----------------------- Trained GIF -----------------------
with torch.no_grad():
    trained_video = pipe(
        model=seq2seq_model,
        eeg=eeg_raw,
        negative_eeg=negative_eeg,
        latents=latents_add_noise,
        video_length=VIDEO_LENGTH,
        height=288,
        width=512,
        num_inference_steps=50,
        guidance_scale=12.5
    ).videos
save_gif(trained_video, os.path.join(SAVE_DIR, "clip_trained.gif"))

# ----------------------- Random GIF -----------------------
random_latents = torch.randn_like(latents_add_noise)
with torch.no_grad():
    random_video = pipe(
        model=seq2seq_model,
        eeg=eeg_raw,
        negative_eeg=negative_eeg,
        latents=random_latents,
        video_length=VIDEO_LENGTH,
        height=288,
        width=512,
        num_inference_steps=50,
        guidance_scale=12.5
    ).videos
save_gif(random_video, os.path.join(SAVE_DIR, "clip_random.gif"))

# ----------------------- Ground-truth GIF -----------------------
gt_gif_files = [f for f in os.listdir(PROCESSED_GIF_DIR) if f.endswith(".gif")]
if gt_gif_files:
    shutil.copy(os.path.join(PROCESSED_GIF_DIR, random.choice(gt_gif_files)),
                os.path.join(SAVE_DIR, "clip_ground_truth.gif"))

print("Inference complete: random, trained, and ground-truth GIFs saved.")
