# THIS FILE HAS BEEN PATCHED

# ----------------------------------------------------------------
# FAULTY CODE: IMPORTS WERE INCORRECT
# from tuneavideo.pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
# from tuneavideo.models.unet import UNet3DConditionModel
# from tuneavideo.util import save_videos_grid
# from tuneavideo.models.eeg_text import CLIP
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# PATCHED CODE: CORRECTED IMPORTS
from pipelines_original.pipeline_tuneeeg2video import TuneAVideoPipeline
from models_original.tuneavideo.unet import UNet3DConditionModel
from models_original.tuneavideo.util import save_videos_grid
# From EEG2Video.EEG2Video_New.Semantic.eeg_text import CLIP
# Using CLIPSmall as a stub semantic predictor
# This keeps the pipeline runnable without a full CLIP checkpoint
# from EEG2Video.EEG2Video_New.Semantic.eeg_text import CLIPSmall as CLIP
# ----------------------------------------------------------------

import torch
import numpy as np
from einops import rearrange
from sklearn import preprocessing

# ----------------------------------------------------------------
# SEQ2SEQ MODEL FOR EEG -> LATENTS (TOY VERSION ADDED BY ME)
import torch.nn as nn

CKPT_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_semantic.pt"

class EEG2Latent(nn.Module):
    def __init__(self, eeg_dim=62*512, latent_dim=77*768, hidden=512, num_layers=2):  # <-- update latent_dim
        super().__init__()
        self.embed = nn.Linear(eeg_dim, hidden)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(hidden, latent_dim)

    def forward(self, x):
        b, c, t = x.shape
        x = x.view(b, -1)
        x = self.embed(x).unsqueeze(0)
        h = self.transformer(x)
        out = self.out(h).squeeze(0)
        return out

seq2seq = EEG2Latent().to("cuda")
seq2seq.load_state_dict(torch.load(CKPT_PATH, map_location="cuda"))
seq2seq.eval()
print("Loaded Seq2Seq checkpoint:", CKPT_PATH)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# PATCHED CODE: DUMMY CHECKPOINT USED TO TEST PIPELINE
# pretrained_eeg_encoder_path = "/content/drive/MyDrive/EEG2Video_checkpoints/eeg2text_40_eeg.pt"
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# COMMENTED OUT BY ME TO TEST SEQ2SEQ PIPELINE
# model = CLIP()
# model.load_state_dict(torch.load(pretrained_eeg_encoder_path, map_location=lambda storage, loc: storage)['state_dict'])
# model.to(torch.device('cuda'))
# model.eval()
# ----------------------------------------------------------------

eeg_data_path = "/content/drive/MyDrive/Data/Raw/EEG/sub1.npy" # your own data path for eeg data
EEG_dim = 62*200                             # the dimension of an EEG segment
eegdata = np.load(eeg_data_path)
GT_label = np.array([[23, 22, 9, 6, 18,       14, 5, 36, 25, 19,      28, 35, 3, 16, 24,      40, 15, 27, 38, 33,
             34, 4, 39, 17, 1,       26, 20, 29, 13, 32,     37, 2, 11, 12, 30,      31, 8, 21, 7, 10, ],
            [27, 33, 22, 28, 31,     12, 38, 4, 18, 17,      35, 39, 40, 5, 24,      32, 15, 13, 2, 16,
 	         34, 25, 19, 30, 23,     3, 8, 29, 7, 20,        11, 14, 37, 6, 21,      1, 10, 36, 26, 9, ],
            [15, 36, 31, 1, 34,      3, 37, 12, 4, 5,        21, 24, 14, 16, 39,     20, 28, 29, 18, 32,
             2, 27, 8, 19, 13,       10, 30, 40, 17, 26,     11, 9, 33, 25, 35,      7, 38, 22, 23, 6,],
            [16, 28, 23, 1, 39,      10, 35, 14, 19, 27,     37, 31, 5, 18, 11,      25, 29, 13, 20, 24,
            7, 34, 26, 4, 40 ,       12, 8, 22, 21, 30,      17, 2, 38, 9,  3 ,      36, 33, 6, 32, 15,],
            [18, 29, 7, 35, 22  ,    19, 12, 36, 8, 15,      28, 1, 34, 23, 20 ,     13, 37, 9, 16, 30  ,
             2, 33, 27, 21, 14 ,     38, 10, 17, 31, 3,      24, 39, 11, 32, 4,      25, 40, 5, 26, 6 ,],
            [29, 16, 1, 22, 34,      39, 24, 10, 8, 35,      27, 31, 23, 17, 2,      15, 25, 40, 3, 36,
             26, 6, 14, 37, 9,       12, 19, 30, 5, 28,      32, 4, 13, 18, 21,      20, 7, 11, 33, 38],
            [38, 34, 40, 10, 28,     7, 1, 37, 22, 9,        16, 5, 12, 36, 20,      30, 6, 15, 35, 2,
             31, 26, 18, 24, 8,      3, 23, 19, 14, 13,      21, 4, 25, 11, 32,      17, 39, 29, 33, 27]
            ])

# chosed_label = [1, 10, 12, 16, 19, 23, 25, 31, 34, 39]
chosed_label = [i for i in range(1, 41)]

EEG = []
for i in range(6):
    indices = [list(GT_label[i]).index(element) for element in chosed_label]
    chosed_eeg = eegdata[i][indices,:]
    EEG.append(chosed_eeg)
EEG = np.stack(EEG, axis=0)

# ----------------------------------------------------------------
# FAULTY CODE
# test_indices = [list(GT_label[6]).index(element) for element in chosed_label]
# eeg_test = eegdata[6][test_indices, :]
# eeg_test = torch.from_numpy(eeg_test)
# eeg_test = rearrange(eeg_test, 'a b c d e -> (a b) c (d e)')
# eeg_test = torch.mean(eeg_test, dim=1).resize(eeg_test.shape[0], EEG_dim)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# PATCHED CODE
test_indices = [list(GT_label[6]).index(element) for element in chosed_label]
eeg_test = eegdata[6][test_indices, :]          # shape [clips, timepoints]
eeg_test = torch.from_numpy(eeg_test)

# Flatten into [clips, features]
eeg_test = eeg_test.view(eeg_test.shape[0], -1)

# crop to match CLIPSmall input dimension
EEG_dim = 310
eeg_test = eeg_test[:, :EEG_dim]

print(">>> eeg_test shape after shrink:", eeg_test.shape)
# ----------------------------------------------------------------

EEG = torch.from_numpy(EEG)
print(EEG.shape)

# ----------------------------------------------------------------
# FAULTY CODE
# EEG = rearrange(EEG, 'a b c d e f -> (a b c) d (e f)')
# print(EEG.shape)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# PATCHED CODE: PAD TO 6D SO EINOPS DOESN'T BREAK
while EEG.ndim < 6:
    EEG = EEG.unsqueeze(-1)

print(">>> EEG shape padded for rearrange:", EEG.shape)
EEG = rearrange(EEG, 'a b c d e f -> (a b c) d (e f)')
print(">>> EEG shape after rearrange:", EEG.shape)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# FAULTY CODE
# EEG = torch.mean(EEG, dim=1).resize(EEG.shape[0], EEG_dim)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# PATCHED CODE: SAFE SHRINK FOR DEBUGGING
EEG = torch.mean(EEG, dim=1)       # reduce along dim=1
EEG = EEG.view(EEG.shape[0], -1)   # flatten

# crop EEG to CLIPSmall input dimension
EEG_dim = 310
EEG = EEG[:, :EEG_dim]             # take only first 310 features

print(">>> EEG shape after shrink:", EEG.shape)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# FAULTY CODE
# scaler = preprocessing.StandardScaler().fit(EEG)
# EEG = scaler.transform(EEG)
# EEG = torch.from_numpy(EEG).float().cuda()
# eeg_test = scaler.transform(eeg_test)
# eeg_test = torch.from_numpy(eeg_test).float().cuda()
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# PATCHED CODE
print(">>> Skipping StandardScaler (too big), using simple normalization instead")

# EEG is already a torch.Tensor here
EEG = EEG.float()
EEG = (EEG - EEG.mean()) / (EEG.std() + 1e-6)
EEG = EEG.cuda()

# eeg_test is ALSO already a tensor at this point (flattened above), so no torch.from_numpy
eeg_test = eeg_test.float()
eeg_test = (eeg_test - eeg_test.mean()) / (eeg_test.std() + 1e-6)
eeg_test = eeg_test.cuda()

print(">>> EEG shape:", EEG.shape, "device:", EEG.device)
print(">>> eeg_test shape:", eeg_test.shape, "device:", eeg_test.device)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# FAULTY CODE
# pretrained_model_path = "./checkpoints/stable-diffusion-v1-4"
# my_model_path = "./outputs/40_classes_video_200_epoch"
# unet = UNet3DConditionModel.from_pretrained(my_model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')
# pipe = TuneAVideoPipeline.from_pretrained(pretrained_model_path, unet=unet, torch_dtype=torch.float16).to("cuda")
# pipe.enable_xformers_memory_efficient_attention()
# pipe.enable_vae_slicing()
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# PATCHED CODE
from models_original.tuneavideo.unet import UNet3DConditionModel

print(">>> Starting UNet initialization")
unet = UNet3DConditionModel(
    sample_size=64,
    in_channels=4,
    out_channels=4,
    layers_per_block=2,
    block_out_channels=(320, 640, 1280),
    down_block_types=("DownBlock3D", "DownBlock3D", "DownBlock3D"),
    up_block_types=("UpBlock3D", "UpBlock3D", "UpBlock3D"),
    cross_attention_dim=768
).to("cuda").to(torch.float32)   # force float32

print(">>> UNet initialized and moved to CUDA (float32)")

print(">>> Starting pipeline creation")
pipe = TuneAVideoPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    unet=unet,
    torch_dtype=torch.float32   # force float32 for the whole pipeline
).to("cuda")

pipe.unet.to(torch.float32)
pipe.vae.to(torch.float32)

print(">>> Pipeline created (all float32)")
print(">>> Enabling memory optimizations...")
pipe.enable_vae_slicing()
pipe.to("cuda")

print(">>> Pipeline device:", pipe.device)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# SKIPPING THIS TO VERIFY THE PIPELINE WORKS
# this are latents with DANA, these latents are pre-prepared by Seq2Seq model
# print(">>> Loading pre-prepared latents with DANA")
# latents_add_noise = np.load('./tuneavideo/models/latent_add_noise.npy')
# latents_add_noise = torch.from_numpy(latents_add_noise).half()
# latents_add_noise = rearrange(latents_add_noise, 'a b c d e -> a c b d e')
# print(">>> latents_add_noise shape:", latents_add_noise.shape)

# SKIPPING THIS TO VERIFY THE PIPELINE WORKS
# this are latents w/o DANA, these latents are pre-prepared by Seq2Seq model
# print(">>> Loading pre-prepared latents w/o DANA")
# latents = np.load('./tuneavideo/models/latents.npy')
# latents = torch.from_numpy(latents).half()
# latents = rearrange(latents, 'a b c d e -> a c b d e')
# print(">>> latents shape:", latents.shape)
# print(">>> eeg_test shape:", eeg_test.shape)

print(">>> Generating latents from Seq2Seq instead of loading from file")
with torch.no_grad():
    # eeg_test: [40, 310]
    B, F = 1, 6   # 1 video, 6 frames
    eeg_input = eeg_test[0:1, :]  # take first sample [1,310]

    # Pad back to [1, 62*512]
    padded = torch.zeros((B, 62*512), device=eeg_input.device)
    padded[:, :310] = eeg_input  # fill first 310 features

    # Reshape to [B,62,512]
    eeg_segment = padded.view(B, 62, 512)

    # Repeat per frame so Seq2Seq predicts latents for 6 frames
    eeg_segment = eeg_segment.repeat(F, 1, 1)  # [6,62,512]

    # Predict latents
    pred_latents = seq2seq(eeg_segment)  # [6,4096]

    # Reshape into [1,6,4,32,32]
    latents = pred_latents.view(1, F, 4, 32, 32).to("cuda")

    # Rearrange to [1,4,6,32,32]
    latents = latents.permute(0, 2, 1, 3, 4)

    # Upsample from 32x32 -> 36x64 so it matches pipeline expectations
    latents = torch.nn.functional.interpolate(
        latents.reshape(1*4*6, 1, 32, 32),  # flatten B,C,F
        size=(36, 64),
        mode="bilinear",
        align_corners=False
    )

    # Reshape back to [1,4,6,36,64]
    latents = latents.view(1, 4, F, 36, 64).to("cuda")

print(">>> latents final shape:", latents.shape)
# ----------------------------------------------------------------

# PATCHED CODE: CHANGED FROM TRUE TO FALSE FOR TESTING + APPARENTLY NOT USED
# Ablation, inference w/o Seq2Seq and w/o DANA
woSeq2Seq = False
woDANA = True

# ----------------------------------------------------------------
# PATCHED CODE: CHANGING FROM 200 TO 2 FOR TESTING (BUT WITH LATENTS)
# print(">>> Starting inference loop")
# for i in range(0,2):
#     print(f">>> Generating video {i}")
#     if woSeq2Seq:
#         video = pipe(model, eeg_test[i:i+1,...], latents=None, video_length=6, height=288, width=512, num_inference_steps=100, guidance_scale=12.5).videos
#         savename = '40_Classes_woSeq2Seq'
#     elif woDANA:
#         video = pipe(model, eeg_test[i:i+1,...], latents=latents[i:i+1,...], video_length=6, height=288, width=512, num_inference_steps=100, guidance_scale=12.5).videos
#         savename = '40_Classes_woDANA'
#     else:
#         video = pipe(model, eeg_test[i:i+1,...], latents=latents_add_noise[i:i+1,...], video_length=6, height=288, width=512, num_inference_steps=100, guidance_scale=12.5).videos
#         savename = '40_Classes_Fullmodel'
#     save_videos_grid(video, f"./{savename}/{i}.gif")
#     print(f">>> Saved {savename}/{i}.gif")
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# PATCHED CODE: UPDATED LOOP TO ONLY TEST SEQ2SEQ PIPELINE
# class DummySemantic(torch.nn.Module):
#     def forward(self, x):
#         b = x.size(0)
#         # expand or pad from 310 -> 77*768 = 59136
#         expanded = torch.zeros((b, 77*768), device=x.device)
#         expanded[:, :min(310, 77*768)] = x[:, :min(310, 77*768)]
#         return expanded

# dummy_model = DummySemantic().to("cuda")



print(">>> Starting inference loop")
for i in range(0, 2):
    print(f">>> Generating video {i}")
    print(">>> eeg_embeddings dtype:", eeg_test.dtype)
    video = pipe(
        seq2seq,                      
        eeg_test[i:i+1, ...],
        latents=latents[:, :6, ...],  # make sure we take the 6 frames
        video_length=6,
        height=288,
        width=512,
        num_inference_steps=100,
        guidance_scale=12.5
    ).videos
    savename = "40_Classes_Test"
    save_videos_grid(video, f"./{savename}/{i}.gif")
    print(f">>> Saved {savename}/{i}.gif")
# ----------------------------------------------------------------