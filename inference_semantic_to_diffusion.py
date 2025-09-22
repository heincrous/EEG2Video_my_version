import os, sys, torch, imageio, gc, numpy as np
from diffusers import DDIMScheduler, AutoencoderKL
from einops import rearrange

# === repo root and sys.path setup ===
repo_root = "/content/EEG2Video_my_version"
sys.path.append(os.path.join(repo_root, "pipelines"))
from pipeline_tuneavideo import TuneAVideoPipeline

# UNet is under core_files
sys.path.append(os.path.join(repo_root, "core_files"))
from unet import UNet3DConditionModel

# Semantic predictor is under training
sys.path.append(os.path.join(repo_root, "training"))
from train_semantic import SemanticPredictor

# Paths
drive_root = "/content/drive/MyDrive/EEG2Video_data/processed"
ckpt_path  = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_predictor.pt"
pretrained_model_path = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
output_dir = "/content/drive/MyDrive/EEG2Video_outputs"
save_dir   = os.path.join(output_dir, "semantic_only_one")
os.makedirs(save_dir, exist_ok=True)

# === Load test lists ===
eeg_test_list  = os.path.join(drive_root, "EEG_features/test_list.txt")
text_test_list = os.path.join(drive_root, "BLIP_text/test_list.txt")

with open(eeg_test_list, "r") as f:
    eeg_files = [line.strip() for line in f.readlines()]
with open(text_test_list, "r") as f:
    text_files = [line.strip() for line in f.readlines()]

assert len(eeg_files) == len(text_files), "Mismatch EEG vs text test files"

# === Select one sample (index 0 by default) ===
idx = 0
eeg_path, text_path = eeg_files[idx], text_files[idx]
base_name = os.path.splitext(os.path.basename(eeg_path))[0]  # classXX_clipYY

# Load EEG and caption
eeg = torch.tensor(np.load(eeg_path).reshape(1, -1), dtype=torch.float32).cuda()
with open(text_path, "r") as f:
    caption = f.read().strip()

# === Load semantic predictor ===
sem_model = SemanticPredictor().cuda()
sem_model.load_state_dict(torch.load(ckpt_path, map_location="cuda")["state_dict"])
sem_model.eval()

with torch.no_grad():
    pred_embed = sem_model(eeg).half()
pred_embed = pred_embed.view(1, 77, 768)

# === Load diffusion pipeline ===
gc.collect(); torch.cuda.empty_cache()
pipe = TuneAVideoPipeline(
    vae=AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae", torch_dtype=torch.float16),
    text_encoder=None, tokenizer=None,
    unet=UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet"),
    scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler"),
).to("cuda")
pipe.unet.to(torch.float16)
pipe.enable_vae_slicing()

# === Generate video ===
generator = torch.Generator(device="cuda").manual_seed(42)
result = pipe(
    prompt_embeds=pred_embed,
    video_length=8,
    width=512,
    height=288,
    num_inference_steps=20,
    generator=generator,
)

video_tensor = result.videos
frames = (video_tensor[0] * 255).clamp(0,255).to(torch.uint8).permute(0,2,3,1).cpu().numpy()
if frames.shape[-1] > 3:
    frames = frames[...,:3]

mp4_path = os.path.join(save_dir, base_name + ".mp4")
writer = imageio.get_writer(mp4_path, fps=8, codec="libx264")
for f in frames: writer.append_data(f)
writer.close()

print(f"Saved video: {mp4_path}")
print(f"Sample: {base_name} | Caption: {caption}")
