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
import shutil
import numpy as np
import torch
import imageio

from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from core_files.unet import UNet3DConditionModel
from training.my_autoregressive_transformer import myTransformer

# ----------------------- Config -----------------------
TEST_BASE = "/content/drive/MyDrive/EEG2Video_data/processed/Split_4train1test/test"
EEG_SEG_DIR = os.path.join(TEST_BASE, "EEG_segments")
GT_GIF_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/Video_Gif/Block1"  # adjust if needed
SAVE_DIR = "/content/drive/MyDrive/EEG2Video_inference"
os.makedirs(SAVE_DIR, exist_ok=True)

SEQ2SEQ_CKPT = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_checkpoint.pt"
DIFFUSION_UNET_DIR = "/content/drive/MyDrive/EEG2Video_checkpoints/EEG2Video_diffusion_output/unet"

VIDEO_LENGTH = 6
HEIGHT = 288
WIDTH = 512
STEPS = 50
GUIDANCE = 12.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------- Utils -----------------------
def save_gif(frames, path, fps=5):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()
    out = []
    for f in frames:
        if f.ndim == 3 and f.shape[0] in (1, 3):
            f = f.squeeze(0) if f.shape[0] == 1 else f.transpose(1, 2, 0)
        out.append(np.clip(f * 255.0, 0, 255).astype(np.uint8))
    imageio.mimsave(path, out, fps=fps)

def pick_random_eeg_clip(root):
    subs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    assert subs, "No subjects found under EEG_segments."
    sub = random.choice(subs)
    sub_path = os.path.join(root, sub)

    blocks = [b for b in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, b))]
    assert blocks, f"No blocks under {sub_path}."
    block = random.choice(blocks)
    block_path = os.path.join(sub_path, block)

    clips = [f for f in os.listdir(block_path) if f.endswith(".npy")]
    assert clips, f"No .npy EEG files under {block_path}."
    clip = random.choice(clips)

    return sub, block, os.path.join(block_path, clip)

# ----------------------- Select test clip -----------------------
sub, block, EEG_PATH = pick_random_eeg_clip(EEG_SEG_DIR)
print(f"Selected random clip: Subject={sub}, Block={block}, File={os.path.basename(EEG_PATH)}")
print(f"EEG path: {EEG_PATH}")

# ----------------------- Load raw EEG (robust) -----------------------
eeg_np = np.load(EEG_PATH)

def to_7x62x200(arr):
    # cases:
    # [7,62,200] -> return
    # [62, 200*k] -> reshape to [k,62,200], then pad/trim to 7
    # [62,200] -> [1,62,200] then pad to 7
    if arr.ndim == 3 and arr.shape == (7, 62, 200):
        return arr
    if arr.ndim == 2 and arr.shape[0] == 62 and arr.shape[1] % 200 == 0:
        k = arr.shape[1] // 200
        arr = arr.reshape(62, k, 200).transpose(1, 0, 2)  # [k,62,200]
        if k < 7:
            pad = np.repeat(arr[-1:, ...], 7 - k, axis=0)  # repeat last
            arr = np.concatenate([arr, pad], axis=0)
        elif k > 7:
            arr = arr[:7, ...]
        return arr
    raise RuntimeError(f"Unsupported EEG shape {arr.shape}. Expected [7,62,200] or [62,200*k].")

eeg_np = to_7x62x200(eeg_np)
eeg_raw = torch.from_numpy(eeg_np).unsqueeze(0).to(device)  # [1,7,62,200]
negative_eeg_raw = eeg_raw.mean(dim=1, keepdim=True)        # [1,1,62,200]

print("eeg_raw:", tuple(eeg_raw.shape))
print("negative_eeg_raw:", tuple(negative_eeg_raw.shape))

# ----------------------- Load Seq2Seq -----------------------
seq2seq = myTransformer().to(device)
ckpt = torch.load(SEQ2SEQ_CKPT, map_location=device)
seq2seq.load_state_dict(ckpt["state_dict"])
seq2seq.eval()

with torch.no_grad():
    emb = seq2seq(eeg_raw)                # expect [1,77,768] per authors
    neg_emb = seq2seq(negative_eeg_raw)   # [1,77,768]

if emb.ndim != 3 or emb.shape[1:] != (77, 768):
    raise RuntimeError(f"Seq2Seq must output [B,77,768], got {tuple(emb.shape)}")
if neg_emb.ndim != 3 or neg_emb.shape[1:] != (77, 768):
    raise RuntimeError(f"Seq2Seq negative must output [B,77,768], got {tuple(neg_emb.shape)}")

emb_flat = emb.reshape(emb.shape[0], -1)         # [1, 59136]
neg_emb_flat = neg_emb.reshape(neg_emb.shape[0], -1)

print("emb_flat:", tuple(emb_flat.shape))
print("neg_emb_flat:", tuple(neg_emb_flat.shape))

# ----------------------- Load diffusion UNet and pipeline -----------------------
unet = UNet3DConditionModel.from_pretrained_2d(DIFFUSION_UNET_DIR).to(device)
pipe = TuneAVideoPipeline.from_pretrained(os.path.dirname(DIFFUSION_UNET_DIR), unet=unet)
pipe.to(device)
pipe.enable_vae_slicing()
pipe.enable_attention_slicing()

# ----------------------- Inference: trained -----------------------
with torch.no_grad():
    trained = pipe(
        model=None,                   # pipeline will ignore 'model' since we pass embeddings already flattened
        eeg=emb_flat,                 # feed flattened [B, 59136] so pipeline reshape to [B,77,768] succeeds
        negative_eeg=neg_emb_flat,    # same for negative path
        latents=None,
        video_length=VIDEO_LENGTH,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE
    ).videos
save_gif(trained, os.path.join(SAVE_DIR, "clip_trained.gif"))

# ----------------------- Inference: random -----------------------
with torch.no_grad():
    rand_emb = torch.randn_like(emb)          # [1,77,768]
    rand_emb_flat = rand_emb.reshape(1, -1)   # [1,59136]
    random_vid = pipe(
        model=None,
        eeg=rand_emb_flat,
        negative_eeg=neg_emb_flat,
        latents=None,
        video_length=VIDEO_LENGTH,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE
    ).videos
save_gif(random_vid, os.path.join(SAVE_DIR, "clip_random.gif"))

# ----------------------- Ground-truth GIF -----------------------
if os.path.isdir(GT_GIF_DIR):
    gt_list = [f for f in os.listdir(GT_GIF_DIR) if f.endswith(".gif")]
    if gt_list:
        shutil.copy(os.path.join(GT_GIF_DIR, random.choice(gt_list)),
                    os.path.join(SAVE_DIR, "clip_ground_truth.gif"))

print("Done. Saved: clip_trained.gif, clip_random.gif, clip_ground_truth.gif")
