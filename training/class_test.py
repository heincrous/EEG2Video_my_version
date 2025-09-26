# ==========================================
# Diffusion Inference (Class Average Test)
# ==========================================
import os, sys, gc, torch, imageio
import numpy as np

from diffusers import AutoencoderKL, DDIMScheduler
from core_files.unet import UNet3DConditionModel
from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline

# ==========================================
# Paths
# ==========================================
drive_root            = "/content/drive/MyDrive/EEG2Video_data/processed"
pretrained_model_path = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
finetuned_model_path  = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final"
blip_path             = os.path.join(drive_root, "BLIP_embeddings", "BLIP_embeddings.npy")
save_dir              = "/content/drive/MyDrive/EEG2Video_outputs/class_avg_test"

os.makedirs(save_dir, exist_ok=True)

# ==========================================
# Memory config
# ==========================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()
device = "cuda"

# ==========================================
# Load pipeline backbone
# ==========================================
vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").to(device, dtype=torch.float32)
scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet").to(device, dtype=torch.float32)

pipe = TuneAVideoPipeline.from_pretrained(
    finetuned_model_path,
    vae=vae,
    unet=unet,
    scheduler=scheduler,
    torch_dtype=torch.float32,
).to(device)
pipe.enable_vae_slicing()

# ==========================================
# Load BLIP embeddings and compute class average
# ==========================================
blip = np.load(blip_path)   # (7,40,5,77,768)
assert blip.ndim == 5, f"Unexpected shape: {blip.shape}"

cls_idx = 0   # choose one class for test
class_avg = blip[:, cls_idx, :, :, :].mean(axis=(0,1))   # (77,768)

semantic_pred = torch.tensor(class_avg, dtype=torch.float32, device=device).unsqueeze(0)
negative = semantic_pred.mean(dim=0, keepdim=True).float().to(device)

print("Class average embedding shape:", semantic_pred.shape)

# ==========================================
# Run inference
# ==========================================
video_length = 6
fps = 3

video = pipe(
    model=None,
    eeg=semantic_pred,
    negative_eeg=negative,
    latents=None,
    video_length=video_length,
    height=288,
    width=512,
    num_inference_steps=100,
    guidance_scale=12.5,
).videos

frames = (video[0] * 255).clamp(0, 255).to(torch.uint8)
frames = frames.permute(0, 2, 3, 1).cpu().numpy()

if frames.shape[-1] > 3:
    frames = frames[..., :3]
elif frames.shape[-1] == 1:
    frames = np.repeat(frames, 3, axis=-1)

mp4_path = os.path.join(save_dir, f"class{cls_idx}_avg_{video_length}f_{fps}fps.mp4")
writer = imageio.get_writer(mp4_path, fps=fps, codec="libx264")
for f in frames:
    writer.append_data(f)
writer.close()

print("Saved class-average reconstruction video:", mp4_path)
