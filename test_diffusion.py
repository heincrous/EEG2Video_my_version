# ==========================================
# Test Diffusion Model with Single Caption
# ==========================================
import os, gc, torch, imageio, numpy as np
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from core.unet import UNet3DConditionModel

# ==========================================
# Paths
# ==========================================
PRETRAINED_MODEL_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
FINETUNED_PIPELINE    = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final_subset1-10-12-16-19-23-25-31-34-39"
BLIP_TEXT             = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/BLIP_text.npy"
SAVE_PATH             = "/content/drive/MyDrive/EEG2Video_outputs/test_diffusion_caption.mp4"

device = "cuda" if torch.cuda.is_available() else "cpu"
gc.collect(); torch.cuda.empty_cache()

# ==========================================
# Load pretrained backbone + finetuned UNet
# ==========================================
vae       = AutoencoderKL.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="vae").to(device, dtype=torch.float32)
scheduler = DDIMScheduler.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="tokenizer")

unet = UNet3DConditionModel.from_pretrained(
    FINETUNED_PIPELINE,
    subfolder="unet",
    torch_dtype=torch.float32
).to(device)

os.makedirs("/content/EEG2Video_my_version/core_files", exist_ok=True)

pipe = TuneAVideoPipeline.from_pretrained(
    PRETRAINED_MODEL_PATH,
    vae=vae,
    unet=unet,
    scheduler=scheduler,
    tokenizer=tokenizer,
    torch_dtype=torch.float32
).to(device)

pipe.enable_vae_slicing()

# ==========================================
# Pick one caption from Block 7 (subset class)
# ==========================================
blip_text = np.load(BLIP_TEXT, allow_pickle=True)  # shape (7,40,5)
caption = blip_text[6, 1, 0]   # block7, class=1, clip=0
print("Testing caption:", caption)

# ==========================================
# Run inference
# ==========================================
video = pipe(
    prompt=caption,
    video_length=6,     # 6 frames
    height=288,
    width=512,
    num_inference_steps=100,
    guidance_scale=12.5,
).videos

frames = (video[0] * 255).clamp(0,255).to(torch.uint8).permute(0,2,3,1).cpu().numpy()

if frames.shape[-1] > 3:
    frames = frames[..., :3]
elif frames.shape[-1] == 1:
    frames = np.repeat(frames, 3, axis=-1)

fps = 3
writer = imageio.get_writer(SAVE_PATH, fps=fps, codec="libx264")
for f in frames:
    writer.append_data(f)
writer.close()

print("Saved test video to:", SAVE_PATH)
