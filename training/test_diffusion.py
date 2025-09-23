import os, sys, random, torch, imageio, gc
from einops import rearrange
from diffusers import DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

repo_root = "/content/EEG2Video_my_version"
sys.path.append(os.path.join(repo_root, "pipelines"))
from pipeline_tuneavideo import TuneAVideoPipeline

sys.path.append(os.path.join(repo_root, "core_files"))
from unet import UNet3DConditionModel

# === DIRECTORIES ===
pretrained_model_path = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
trained_output_dir    = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final"
test_text_list        = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/test_list.txt"
blip_text_root        = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text"
save_dir              = "/content/drive/MyDrive/EEG2Video_outputs/test_diffusion"

os.makedirs(trained_output_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

# === MEMORY CONFIG ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()

# === LOAD TRAINED PIPELINE WITH FP16 ===
pipe = TuneAVideoPipeline(
    vae=AutoencoderKL.from_pretrained(trained_output_dir, subfolder="vae", torch_dtype=torch.float16),
    text_encoder=CLIPTextModel.from_pretrained(trained_output_dir, subfolder="text_encoder", torch_dtype=torch.float16),
    tokenizer=CLIPTokenizer.from_pretrained(trained_output_dir, subfolder="tokenizer"),
    unet=UNet3DConditionModel.from_pretrained_2d(trained_output_dir, subfolder="unet"),
    scheduler=DDIMScheduler.from_pretrained(trained_output_dir, subfolder="scheduler"),
)
pipe.unet.to(torch.float16)
pipe.enable_vae_slicing()
pipe = pipe.to("cuda")

# === PICK RANDOM PROMPT FROM TEST LIST ===
with open(test_text_list, "r") as f:
    test_prompts = [os.path.join(blip_text_root, line.strip()) for line in f]

prompt_path = random.choice(test_prompts)
with open(prompt_path, "r") as pf:
    prompt_text = pf.read().strip()

print("Chosen prompt file:", prompt_path)
print("Caption text:", prompt_text)

# === GENERATE VIDEO ===
generator = torch.Generator(device="cuda").manual_seed(42)

# Match training setup → 6 frames, save as 2 s video
video_length = 6
fps = 3

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

# === SAVE MP4 ===
frames = (video_tensor[0] * 255).clamp(0, 255).to(torch.uint8)  # [f, c, h, w]
frames = frames.permute(0, 2, 3, 1).cpu().numpy()               # [f, h, w, c]

if frames.shape[-1] > 3:
    frames = frames[..., :3]
elif frames.shape[-1] == 1:
    frames = frames.repeat(3, axis=-1)

print("Final frame shape:", frames.shape)

base_name = os.path.splitext(os.path.basename(prompt_path))[0] + f"_{video_length}f_{fps}fps.mp4"
mp4_path = os.path.join(save_dir, base_name)

writer = imageio.get_writer(mp4_path, fps=fps, codec="libx264")
for f in frames:
    writer.append_data(f)
writer.close()

print("Video saved to:", mp4_path)
print("Final caption used for generation:", prompt_text)
