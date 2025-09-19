# EEG2Video Inference (trained vs random)

import os
import torch
import numpy as np
from einops import rearrange

from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTokenizer

from models_original.tuneavideo.unet import UNet3DConditionModel
from models_original.tuneavideo.util import save_videos_grid
from models_original.seq2seq import Seq2SeqModel
from training_original.train_semantic_predictor import SemanticPredictor

# ----------------------------------------------------------------
# Load EEG features (example: first subject)
eeg_file = "/content/drive/MyDrive/Data/Processed/EEG_timewindows_100/sub1.npy"
eeg_features = np.load(eeg_file)  # [blocks, classes, clips, 4, 62, 100]
eeg_clip = eeg_features[0, 0, 0]  # block0, class0, clip0

# Reshape EEG same way as in training
eeg_input = torch.from_numpy(eeg_clip).unsqueeze(0).float().cuda()  # [1, 4, 62, 100]
eeg_input = rearrange(eeg_input, "b d c t -> b (d c t)")  # flatten to [1, features]
input_dim = eeg_input.shape[1]
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Load trained semantic predictor
semantic_model = SemanticPredictor(input_dim=input_dim).to("cuda")
semantic_model.load_state_dict(torch.load(
    "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_predictor.pt",
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
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

pipe = TuneAVideoPipeline(
    vae=vae,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler
).to("cuda")
pipe.enable_vae_slicing()
# ----------------------------------------------------------------

def run_inference(seq2seq_model, save_name):
    with torch.no_grad():
        B = eeg_input.size(0)
        F = 4  # number of frames
        dummy_tgt = torch.zeros((B, F, 4, 36, 64), device="cuda")

        pred_latents = seq2seq_model(eeg_input, dummy_tgt)  # [B,F,9216]
        latents = pred_latents.view(B, F, 4, 36, 64).permute(0, 2, 1, 3, 4).contiguous()

        eeg_embed = semantic_model(eeg_input)  # [1, 77*768] after reshaping in pipeline

        video = pipe(
            model=None,              # unused, kept for signature compatibility
            eeg=eeg_embed,           # semantic embeddings go here
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
CKPT_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_subset.pt"
seq2seq_trained = Seq2SeqModel().to("cuda")
seq2seq_trained.load_state_dict(torch.load(CKPT_PATH, map_location="cuda"))
seq2seq_trained.eval()
trained_path = run_inference(seq2seq_trained, "sample_trained")

# Run with random Seq2Seq
seq2seq_random = Seq2SeqModel().to("cuda")
seq2seq_random.eval()
random_path = run_inference(seq2seq_random, "sample_random")

print("Inference complete. Saved results:")
print("Trained:", trained_path)
print("Random:", random_path)
