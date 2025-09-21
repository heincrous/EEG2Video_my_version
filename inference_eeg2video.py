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
import random
import shutil

from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from core_files.unet import UNet3DConditionModel
from training.my_autoregressive_transformer import myTransformer

# ----------------------- Paths -----------------------
TEST_BASE = "/content/drive/MyDrive/EEG2Video_data/processed/Split_4train1test/test"
EEG_SEG_DIR = os.path.join(TEST_BASE, "EEG_segments")
SAVE_DIR = "/content/drive/MyDrive/EEG2Video_inference"
os.makedirs(SAVE_DIR, exist_ok=True)

SEQ2SEQ_CKPT = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_checkpoint.pt"
DIFFUSION_UNET_DIR = "/content/drive/MyDrive/EEG2Video_checkpoints/EEG2Video_diffusion_output/unet"

VIDEO_LENGTH = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------- Pick a random test clip from EEG_segments -----------------------
subdirs = [d for d in os.listdir(EEG_SEG_DIR) if os.path.isdir(os.path.join(EEG_SEG_DIR, d))]
random_subj = random.choice(subdirs)
block_dir = os.path.join(EEG_SEG_DIR, random_subj)
blocks = [b for b in os.listdir(block_dir) if os.path.isdir(os.path.join(block_dir, b))]
random_block = random.choice(blocks)
class_dir = os.path.join(block_dir, random_block)
classes = [f for f in os.listdir(class_dir) if f.endswith(".npy")]
TEST_EEG_PATH = os.path.join(class_dir, random.choice(classes))

print(f"Selected random clip: Subject={random_subj}, Block={random_block}, File={os.path.basename(TEST_EEG_PATH)}")
print(f"EEG path: {TEST_EEG_PATH}")

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

# ----------------------- Load raw EEG and negative -----------------------
eeg_raw = torch.from_numpy(np.load(TEST_EEG_PATH)).unsqueeze(0).to(device)  # shape [1, segments, 62, 200]
negative_eeg = eeg_raw.mean(dim=0, keepdim=True)

# ----------------------- Generate Trained GIF -----------------------
with torch.no_grad():
    trained_video = pipe(
        model=seq2seq_model,
        eeg=eeg_raw,
        negative_eeg=negative_eeg,
        latents=None,
        video_length=VIDEO_LENGTH,
        height=288,
        width=512,
        num_inference_steps=50,
        guidance_scale=12.5
    ).videos
save_gif(trained_video, os.path.join(SAVE_DIR, "clip_trained.gif"))

# ----------------------- Generate Random GIF -----------------------
with torch.no_grad():
    random_video = pipe(
        model=seq2seq_model,
        eeg=eeg_raw,
        negative_eeg=negative_eeg,
        latents=None,
        video_length=VIDEO_LENGTH,
        height=288,
        width=512,
        num_inference_steps=50,
        guidance_scale=12.5
    ).videos
save_gif(random_video, os.path.join(SAVE_DIR, "clip_random.gif"))

# ----------------------- Copy Ground-Truth GIF -----------------------
PROCESSED_GIF_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/Video_Gif/Block1"
gt_gif_files = [f for f in os.listdir(PROCESSED_GIF_DIR) if f.endswith(".gif")]
if gt_gif_files:
    shutil.copy(os.path.join(PROCESSED_GIF_DIR, random.choice(gt_gif_files)),
                os.path.join(SAVE_DIR, "clip_ground_truth.gif"))

print("Inference complete: random, trained, and ground-truth GIFs saved.")
