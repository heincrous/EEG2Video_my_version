import os, sys, torch, imageio, gc, numpy as np
from diffusers import DDIMScheduler, AutoencoderKL

# === repo setup ===
repo_root = "/content/EEG2Video_my_version"
sys.path.append(os.path.join(repo_root, "pipelines"))
from pipeline_tuneeeg2video import TuneAVideoPipeline   # <-- use EEG pipeline

sys.path.append(os.path.join(repo_root, "core_files"))
from unet import UNet3DConditionModel

sys.path.append(os.path.join(repo_root, "training"))
from train_semantic import SemanticPredictor

# === paths ===
drive_root = "/content/drive/MyDrive/EEG2Video_data/processed"
ckpt_path  = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_predictor.pt"
pretrained_model_path = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
output_dir = "/content/drive/MyDrive/EEG2Video_outputs"
save_dir   = os.path.join(output_dir, "semantic_only_one")
os.makedirs(save_dir, exist_ok=True)

# === load file lists ===
eeg_test_list  = os.path.join(drive_root, "EEG_features/test_list.txt")
text_test_list = os.path.join(drive_root, "BLIP_text/test_list.txt")

with open(eeg_test_list, "r") as f:
    eeg_files = [line.strip() for line in f.readlines()]
with open(text_test_list, "r") as f:
    text_files = [line.strip() for line in f.readlines()]

# build mapping for text files
text_map = {os.path.splitext(os.path.basename(p))[0]: p for p in text_files}

# filter EEG files to those with matching caption
paired = [(e, text_map[os.path.splitext(os.path.basename(e))[0]])
          for e in eeg_files if os.path.splitext(os.path.basename(e))[0] in text_map]

assert len(paired) > 0, "No matching EEG↔caption pairs found!"

# === pick one pair (first by default) ===
eeg_path, text_path = paired[0]
base_name = os.path.splitext(os.path.basename(eeg_path))[0]

# load EEG + caption
eeg = torch.tensor(np.load(eeg_path).reshape(1, -1), dtype=torch.float32).cuda()
with open(text_path, "r") as f:
    caption = f.read().strip()

print("Chosen sample:", base_name)
print("Caption:", caption)

# === semantic predictor ===
sem_model = SemanticPredictor().cuda()
sem_model.load_state_dict(torch.load(ckpt_path, map_location="cuda")["state_dict"])
sem_model.eval()
with torch.no_grad():
    pred_embed = sem_model(eeg).half()
pred_embed = pred_embed.view(1, 77, 768)

# === diffusion pipeline ===
gc.collect(); torch.cuda.empty_cache()
pipe = TuneAVideoPipeline.from_pretrained(
    pretrained_model_path,
    unet=UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet"),
    torch_dtype=torch.float32,   # <-- run everything in full precision
).to("cuda")
pipe.enable_vae_slicing()

# === generate video ===
generator = torch.Generator(device="cuda").manual_seed(42)
result = pipe(
    model=None,
    eeg=pred_embed,         # EEG embedding [1,77,768]
    video_length=8,
    height=288,
    width=512,
    num_inference_steps=20,
    guidance_scale=7.5,
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
