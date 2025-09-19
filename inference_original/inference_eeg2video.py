# EEG2Video Inference (trained vs random, with ground truth info)

import os
import torch
import numpy as np
from einops import rearrange

from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTokenizer

from models_original.tuneavideo.unet import UNet3DConditionModel
from pipelines_original.pipeline_tuneeeg2video import TuneAVideoPipeline
from models_original.tuneavideo.util import save_videos_grid
from models_original.seq2seq import Seq2SeqModel
from training_original.train_semantic_predictor import SemanticPredictor

# ----------------------------------------------------------------
# Load EEG features (example: subject1, block0, class0, clip0)
eeg_file = "/content/drive/MyDrive/Data/Processed/EEG_timewindows_100/sub1.npy"
eeg_features = np.load(eeg_file)  # [blocks, classes, clips, 4, 62, 100]
eeg_clip = eeg_features[0, 0, 0]  # block0, class0, clip0

# Prepare EEG
eeg_input_4d = torch.from_numpy(eeg_clip).unsqueeze(0).float().cuda()  # [1, 4, 62, 100]
eeg_flat = rearrange(eeg_input_4d, "b f c t -> b (f c t)")             # [1, features]
input_dim = eeg_flat.shape[1]
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Load trained semantic predictor
semantic_model = SemanticPredictor(input_dim=input_dim).to("cuda")
semantic_model.load_state_dict(torch.load(
    "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_predictor_full.pt",
    map_location="cuda"
)["state_dict"])
semantic_model.eval()
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Build fine-tuned diffusion pipeline
DIFFUSION_CKPT = "/content/drive/MyDrive/EEG2Video_checkpoints/videodiffusion"

vae = AutoencoderKL.from_pretrained(DIFFUSION_CKPT, subfolder="vae")
unet = UNet3DConditionModel.from_pretrained_2d(DIFFUSION_CKPT, subfolder="unet")
scheduler = DDIMScheduler.from_pretrained(DIFFUSION_CKPT, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(DIFFUSION_CKPT, subfolder="tokenizer")

pipe = TuneAVideoPipeline(
    vae=vae,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler
).to("cuda")
pipe.enable_vae_slicing()
# ----------------------------------------------------------------

def run_inference(seq2seq_model, save_name, normalize_latents=True):
    with torch.no_grad():
        B = eeg_input_4d.size(0)
        F = eeg_input_4d.size(1)  # number of windows
        dummy_tgt = torch.zeros((B, F, 4, 36, 64), device="cuda")

        # Seq2Seq forward: [B,F,62,100] -> [B,F,9216]
        pred_latents = seq2seq_model(eeg_input_4d, dummy_tgt)

        # Debug stats before normalization
        print(f"[{save_name}] raw latents -> mean: {pred_latents.mean().item():.4f}, std: {pred_latents.std().item():.4f}")

        if normalize_latents:
            pred_latents = (pred_latents - pred_latents.mean()) / (pred_latents.std() + 1e-6)
            print(f"[{save_name}] normalized latents -> mean: {pred_latents.mean().item():.4f}, std: {pred_latents.std().item():.4f}")

        latents = pred_latents.view(B, F, 4, 36, 64).permute(0, 2, 1, 3, 4).contiguous()

        # Semantic predictor: [B, features] -> [B,77,768]
        eeg_embed = semantic_model(eeg_flat)

        video = pipe(
            model=None,   # unused in our wrapper
            eeg=eeg_embed,
            video_length=F,
            height=288,
            width=512,
            num_inference_steps=50,
            guidance_scale=12.5,
            latents=latents
        ).videos

        os.makedirs("Results", exist_ok=True)
        save_path = f"./Results/{save_name}.gif"
        save_videos_grid(video, save_path)
        return save_path

# ----------------------------------------------------------------
# Run with trained Seq2Seq
CKPT_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_sub1to10_blocks1to4.pt"
seq2seq_trained = Seq2SeqModel().to("cuda")
seq2seq_trained.load_state_dict(torch.load(CKPT_PATH, map_location="cuda"))
seq2seq_trained.eval()
trained_path = run_inference(seq2seq_trained, "sample_trained")

# Run with random Seq2Seq
seq2seq_random = Seq2SeqModel().to("cuda")
seq2seq_random.eval()
random_path = run_inference(seq2seq_random, "sample_random")

# ----------------------------------------------------------------
# Ground truth info for the same EEG segment (block0, class0, clip0)
labels = np.load("/content/drive/MyDrive/Data/Raw/meta-info/All_video_label.npy")  # (7,40)
expanded_labels = np.repeat(labels, 5, axis=1)  # (7,200)

clip_idx = 0  # block=0, concept=0, clip=0
concept_id = expanded_labels[0, clip_idx]

with open("/content/drive/MyDrive/Data/Raw/BLIP-caption/1st_10min.txt") as f:
    lines = f.readlines()
caption = lines[clip_idx].strip()

color = np.load("/content/drive/MyDrive/Data/Raw/meta-info/All_video_color.npy")[0, clip_idx]
motion = np.load("/content/drive/MyDrive/Data/Raw/meta-info/All_video_optical_flow_score.npy")[0, clip_idx]

print("\nGround truth for block0,class0,clip0:")
print("Concept ID:", concept_id)
print("Caption:", caption)
print("Color:", color)
print("Motion score:", motion)
