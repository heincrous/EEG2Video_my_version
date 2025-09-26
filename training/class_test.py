# ==========================================
# Class-Averaged Embeddings → Video
# ==========================================
import os, torch, numpy as np, imageio, sys
from einops import rearrange
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTokenizer

# === Repo imports ===
repo_root = "/content/EEG2Video_my_version"
sys.path.append(os.path.join(repo_root, "pipelines"))
from pipeline_tuneeeg2video import TuneAVideoPipeline

sys.path.append(os.path.join(repo_root, "core_files"))
from unet import UNet3DConditionModel

# --- Paths ---
pretrained_model_path = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
finetuned_model_path  = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final"
embeddings_path       = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_embeddings/sub1.npy"
save_dir              = "/content/drive/MyDrive/EEG2Video_outputs/class_test"
os.makedirs(save_dir, exist_ok=True)

device = "cuda"

# --- Load pipeline ---
vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").to(device, dtype=torch.float32)
scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
unet = UNet3DConditionModel.from_pretrained_2d(finetuned_model_path, subfolder="unet").to(device, dtype=torch.float32)
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")

pipe = TuneAVideoPipeline(
    vae=vae,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
).to(device)

pipe.enable_vae_slicing()

# --- Load embeddings (7,40,5,77,768) ---
all_embeds = np.load(embeddings_path)  # shape (7,40,5,77,768)
print("Original embeddings shape:", all_embeds.shape)

# --- Average over blocks (7) and clips (5) → (40,77,768) ---
class_embeds = all_embeds.mean(axis=(0,2))
print("Class-averaged embeddings shape:", class_embeds.shape)

# --- Pick one class (e.g. class 0) ---
chosen_class = 0
embed = class_embeds[chosen_class]        # (77,768)

# --- Duplicate across frames ---
video_length = 6
embed = torch.tensor(embed, dtype=torch.float32).unsqueeze(0).to(device)  # (1,77,768)
embed = embed.repeat(video_length,1,1)   # (6,77,768)

# --- Run inference ---
result = pipe(
    model=None,
    eeg=embed,
    video_length=video_length,
    height=288,
    width=512,
    num_inference_steps=50,
    guidance_scale=12.5,
)

video_tensor = result.videos  # (B,C,F,H,W)
frames = (video_tensor[0] * 255).clamp(0,255).to(torch.uint8)
frames = rearrange(frames, "c f h w -> f h w c").cpu().numpy()

# --- Save MP4 ---
out_path = os.path.join(save_dir, f"class{chosen_class}_demo.mp4")
writer = imageio.get_writer(out_path, fps=3, codec="libx264")
for f in frames:
    writer.append_data(f)
writer.close()

print("Saved class-averaged video to:", out_path)
