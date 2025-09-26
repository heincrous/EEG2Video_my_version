# ==========================================
# Diffusion Inference from ONE Class-Average CLIP Embedding
# Handles (7,40,5,77,768) BLIP/CLIP embeddings
# ==========================================
import os, sys, gc, torch, imageio
import numpy as np
from einops import rearrange
from diffusers import DDIMScheduler, AutoencoderKL

# === Repo imports ===
repo_root = "/content/EEG2Video_my_version"
sys.path.append(os.path.join(repo_root, "pipelines"))
from pipeline_tuneavideo import TuneAVideoPipeline
sys.path.append(os.path.join(repo_root, "core_files"))
from unet import UNet3DConditionModel

# ==========================================
# Paths
# ==========================================
pretrained_model_path = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
trained_output_dir    = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final"
save_dir              = "/content/drive/MyDrive/EEG2Video_outputs/class_avg_videos"
embed_path            = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_embeddings/BLIP_embeddings.npy"

os.makedirs(save_dir, exist_ok=True)

# ==========================================
# Memory config
# ==========================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()

# ==========================================
# Load trained pipeline
# ==========================================
pipe = TuneAVideoPipeline(
    vae=AutoencoderKL.from_pretrained(
        pretrained_model_path, subfolder="vae", torch_dtype=torch.float16
    ),
    text_encoder=None,    # not used here
    tokenizer=None,       # not used here
    unet=UNet3DConditionModel.from_pretrained_2d(trained_output_dir, subfolder="unet"),
    scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler"),
)
pipe.unet.to(torch.float16)
pipe.enable_vae_slicing()
pipe = pipe.to("cuda")

# ==========================================
# Load CLIP-space embeddings
# ==========================================
# Shape: (7 blocks, 40 classes, 5 clips, 77 tokens, 768 dims)
all_embeds = np.load(embed_path)
assert all_embeds.ndim == 5 and all_embeds.shape[1:] == (40,5,77,768), f"Unexpected shape {all_embeds.shape}"
print("Original embeddings shape:", all_embeds.shape)

# Average across blocks (axis=0) and clips (axis=2)
# Result: (40,77,768) → one embedding per class
class_embeds = all_embeds.mean(axis=(0,2))
print("Class-averaged embeddings shape:", class_embeds.shape)

# ==========================================
# Pick ONE class
# ==========================================
target_class = 5  # <--- change this index (0–39) to whichever class you want
avg_embed = class_embeds[target_class:target_class+1]  # (1,77,768)

clip_embeds = torch.tensor(avg_embed, dtype=torch.float16).to("cuda")

# Duplicate across frames
video_length = 6   # frames (≈2s at 3fps)
fps = 3
clip_embeds = clip_embeds.repeat(video_length, 1, 1)  # (6,77,768)

# ==========================================
# Run inference
# ==========================================
generator = torch.Generator(device="cuda").manual_seed(42)

result = pipe(
    prompt_embeds=clip_embeds,
    video_length=video_length,
    width=512,
    height=288,
    num_inference_steps=100,
    generator=generator,
)

video_tensor = result.videos
frames = (video_tensor[0] * 255).clamp(0, 255).to(torch.uint8)
frames = rearrange(frames, "c f h w -> f h w c").cpu().numpy()

if frames.shape[-1] > 3:
    frames = frames[..., :3]
elif frames.shape[-1] == 1:
    frames = frames.repeat(3, axis=-1)

mp4_path = os.path.join(save_dir, f"class{target_class:02d}_avg.mp4")
writer = imageio.get_writer(mp4_path, fps=fps, codec="libx264")
for f in frames:
    writer.append_data(f)
writer.close()

print(f"Saved average video for class {target_class:02d} → {mp4_path}")
