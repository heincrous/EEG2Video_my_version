# ==========================================
# Diffusion Inference (Class Average Embeddings)
# ==========================================
import os, sys, gc
import numpy as np
import torch
import imageio
from einops import rearrange
from diffusers import DDIMScheduler, AutoencoderKL
from transformers import CLIPTokenizer

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
save_dir              = "/content/drive/MyDrive/EEG2Video_outputs/class_average_videos"
blip_path             = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_embeddings/BLIP_embeddings.npy"

os.makedirs(save_dir, exist_ok=True)

# ==========================================
# Memory config
# ==========================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()

# ==========================================
# Load pipeline
# ==========================================
pipe = TuneAVideoPipeline(
    vae=AutoencoderKL.from_pretrained(
        pretrained_model_path, subfolder="vae", torch_dtype=torch.float16
    ),
    tokenizer=CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer"),
    unet=UNet3DConditionModel.from_pretrained_2d(trained_output_dir, subfolder="unet"),
    scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler"),
)
pipe.unet.to(torch.float16)
pipe.enable_vae_slicing()
pipe = pipe.to("cuda")

# ==========================================
# Load BLIP embeddings
# ==========================================
blip = np.load(blip_path)  # shape (7,40,5,77,768)
assert blip.ndim == 5, f"Unexpected BLIP embedding shape: {blip.shape}"

# Average over 7 blocks × 5 clips
class_avgs = blip.mean(axis=(0,2))  # shape (40,77,768)

print("Class average embeddings shape:", class_avgs.shape)

# ==========================================
# Generate video per class
# ==========================================
generator = torch.Generator(device="cuda").manual_seed(42)
video_length = 6
fps = 3

for cls_idx, emb in enumerate(class_avgs):
    emb_torch = torch.tensor(emb, dtype=torch.float16, device="cuda").unsqueeze(0)

    result = pipe(
        prompt_embeds=emb_torch,
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

    mp4_path = os.path.join(save_dir, f"class{cls_idx}_avg.mp4")
    writer = imageio.get_writer(mp4_path, fps=fps, codec="libx264")
    for f in frames:
        writer.append_data(f)
    writer.close()

    print(f"Saved video for class {cls_idx} -> {mp4_path}")
