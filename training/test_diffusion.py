import os
import torch
import imageio

from pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from diffusers import DDIMScheduler

# paths
model_dir = "/content/drive/MyDrive/EEG2Video_outputs"
text_list_path = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/test_list.txt"
save_dir = "/content/drive/MyDrive/EEG2Video_results"
os.makedirs(save_dir, exist_ok=True)

# load pipeline
pipe = TuneAVideoPipeline.from_pretrained(
    model_dir,
    torch_dtype=torch.float16
).to("cuda")
pipe.scheduler = DDIMScheduler.from_pretrained(model_dir, subfolder="scheduler")
pipe.enable_vae_slicing()

# pick a test prompt
with open(text_list_path) as f:
    prompts = [line.strip() for line in f if line.strip()]

idx = 0  # choose which test clip to use, 0 = first
prompt_path = prompts[idx]
with open(prompt_path, "r") as f:
    prompt_text = f.read().strip()

print("=== Test Sample ===")
print("Clip:", prompt_path)
print("Caption:", prompt_text)

# generate video
generator = torch.Generator("cuda").manual_seed(42)
video = pipe(
    prompt_text,
    num_inference_steps=50,
    generator=generator,
).videos  # [B, F, C, H, W]

# convert to mp4
video = (video[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")  # [F, H, W, C]
save_path = os.path.join(save_dir, f"test_sample_{idx}.mp4")
imageio.mimwrite(save_path, video, fps=8, quality=8, format="mp4")

print(f"Saved generated video to {save_path}")
