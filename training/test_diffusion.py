import os, sys, random, torch, imageio
from einops import rearrange
from diffusers import DDIMScheduler

# === PATH SETUP ===
repo_root = "/content/EEG2Video_my_version"
sys.path.append(os.path.join(repo_root, "pipelines"))

from pipeline_tuneavideo import TuneAVideoPipeline

# === DIRECTORIES ===
pretrained_model_path = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
trained_output_dir    = "/content/drive/MyDrive/EEG2Video_outputs"
test_text_list        = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/test_list.txt"
save_dir              = os.path.join(trained_output_dir, "test_samples")
os.makedirs(save_dir, exist_ok=True)

# === LOAD PIPELINE ===
pipe = TuneAVideoPipeline.from_pretrained(
    trained_output_dir,
    scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler"),
)
pipe.enable_vae_slicing()
pipe = pipe.to("cuda")

# === PICK RANDOM PROMPT ===
with open(test_text_list, "r") as f:
    test_prompts = [line.strip() for line in f]

prompt = random.choice(test_prompts)
print("Chosen prompt:", prompt)

# === GENERATE VIDEO ===
generator = torch.Generator(device="cuda").manual_seed(42)

result = pipe(
    prompt,
    video_length=24,   # set to 24 or 48 depending on how your latents were made
    generator=generator,
)

video_tensor = result.videos  # [1, f, c, h, w]

# === SAVE MP4 ===
frames = (video_tensor[0].permute(0,2,3,1).cpu().numpy() * 255).astype("uint8")
mp4_path = os.path.join(save_dir, "sample_test.mp4")
imageio.mimsave(mp4_path, frames, fps=8)

print("Video saved to:", mp4_path)
