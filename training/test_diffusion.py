# ==========================================
# Diffusion Inference (Random Test Bundle, Text-to-Video)
# ==========================================

# === Standard libraries ===
import os
import sys
import gc
import random

# === Third-party libraries ===
import numpy as np
import torch
import imageio
from einops import rearrange
from diffusers import DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

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
bundle_root           = "/content/drive/MyDrive/EEG2Video_data/processed/SubjectBundles"
save_dir              = "/content/drive/MyDrive/EEG2Video_outputs/test_diffusion"

os.makedirs(trained_output_dir, exist_ok=True)
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
    text_encoder=CLIPTextModel.from_pretrained(
        pretrained_model_path, subfolder="text_encoder", torch_dtype=torch.float16
    ),
    tokenizer=CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer"),
    unet=UNet3DConditionModel.from_pretrained_2d(trained_output_dir, subfolder="unet"),
    scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler"),
)
pipe.unet.to(torch.float16)
pipe.enable_vae_slicing()
pipe = pipe.to("cuda")


# ==========================================
# Pick random test prompt from bundle
# ==========================================
test_bundles = [f for f in os.listdir(bundle_root) if f.endswith("_test.npz")]
if not test_bundles:
    raise FileNotFoundError("No *_test.npz found in SubjectBundles.")

chosen_bundle = random.choice(test_bundles)
bundle_path = os.path.join(bundle_root, chosen_bundle)

data = np.load(bundle_path, allow_pickle=True)
texts = data["BLIP_text"]  # (N,)

idx = random.randrange(len(texts))
prompt_text = str(texts[idx])

print("Chosen test bundle:", chosen_bundle)
print("Chosen sample index:", idx)
print("Prompt text:", prompt_text)


# ==========================================
# Run inference
# ==========================================
generator = torch.Generator(device="cuda").manual_seed(42)

video_length = 6   # frames
fps = 3            # frames per second

result = pipe(
    prompt_text,
    video_length=video_length,
    width=512,
    height=288,
    num_inference_steps=100,
    generator=generator,
)

video_tensor = result.videos
print("Result.videos shape:", video_tensor.shape)


# ==========================================
# Save video
# ==========================================
frames = (video_tensor[0] * 255).clamp(0, 255).to(torch.uint8)
frames = rearrange(frames, "c f h w -> f h w c").cpu().numpy()

if frames.shape[-1] > 3:
    frames = frames[..., :3]
elif frames.shape[-1] == 1:
    frames = frames.repeat(3, axis=-1)

print("Final frame shape:", frames.shape)

base_name = f"{os.path.splitext(chosen_bundle)[0]}_idx{idx}_{video_length}f_{fps}fps.mp4"
mp4_path = os.path.join(save_dir, base_name)

writer = imageio.get_writer(mp4_path, fps=fps, codec="libx264")
for f in frames:
    writer.append_data(f)
writer.close()

print("Video saved to:", mp4_path)
print("Final caption used for generation:", prompt_text)
