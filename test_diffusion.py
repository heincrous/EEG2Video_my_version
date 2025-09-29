# ==========================================
# Test Diffusion Model with Single Caption (subset pipeline)
# ==========================================
import os, sys, torch, imageio, gc
import numpy as np
from einops import rearrange
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

from pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from core.unet import UNet3DConditionModel

# === DIRECTORIES ===
PRETRAINED_MODEL_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
FINETUNED_PIPELINE    = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final_subset1-10-12-16-19-23-25-31-34-39"
BLIP_TEXT             = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/BLIP_text.npy"
SAVE_PATH             = "/content/drive/MyDrive/EEG2Video_outputs/test_diffusion_caption.mp4"

# === MEMORY CONFIG ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect(); torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"

# === LOAD TRAINED PIPELINE (same as training setup) ===
pipe = TuneAVideoPipeline(
    vae=AutoencoderKL.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="vae", torch_dtype=torch.float16),
    text_encoder=CLIPTextModel.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="text_encoder", torch_dtype=torch.float16),
    tokenizer=CLIPTokenizer.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="tokenizer"),
    unet=UNet3DConditionModel.from_pretrained_2d(FINETUNED_PIPELINE, subfolder="unet"),
    scheduler=DDIMScheduler.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="scheduler"),
)
pipe.unet.to(torch.float16)
pipe.enable_vae_slicing()
pipe = pipe.to(device)

# === PICK ONE CAPTION FROM BLOCK 7 (subset class) ===
blip_text = np.load(BLIP_TEXT, allow_pickle=True)  # shape (7,40,5)
caption = blip_text[6, 1, 0]   # Block 7, class=1, clip=0
print("Testing caption:", caption)

# === RUN INFERENCE ===
generator = torch.Generator(device=device).manual_seed(42)
video_length, fps = 6, 3  # 6 frames at 3 fps → 2 seconds

result = pipe(
    caption,
    video_length=video_length,
    width=512,
    height=288,
    num_inference_steps=100,
    guidance_scale=12.5,
    generator=generator,
)

video_tensor = result.videos
print("Result.videos shape:", video_tensor.shape)

# === SAVE MP4 ===
frames = (video_tensor[0] * 255).clamp(0,255).to(torch.uint8)
frames = rearrange(frames, 'c f h w -> f h w c').cpu().numpy()

if frames.shape[-1] > 3:
    frames = frames[..., :3]
elif frames.shape[-1] == 1:
    frames = np.repeat(frames, 3, axis=-1)

writer = imageio.get_writer(SAVE_PATH, fps=fps, codec="libx264")
for f in frames: writer.append_data(f)
writer.close()

print("Saved test video to:", SAVE_PATH)
print("Final caption used for generation:", caption)
