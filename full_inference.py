# ==========================================
# Full Inference (with Semantic Predictor auto-detection, cleaned)
# ==========================================

import os, sys, gc, glob, imageio, torch
import numpy as np
import joblib
from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from core_files.unet import UNet3DConditionModel

repo_root = "/content/EEG2Video_my_version"
sys.path.append(os.path.join(repo_root, "core_files"))
from add_noise import Diffusion

drive_root            = "/content/drive/MyDrive/EEG2Video_data/processed"
pretrained_model_path = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
finetuned_model_path  = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final"
semantic_ckpt_dir     = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"

eeg_feat_list_path    = os.path.join(drive_root, "EEG_DE/test_list.txt")
blip_text_root        = os.path.join(drive_root, "BLIP_text")

output_dir            = "/content/drive/MyDrive/EEG2Video_outputs/test_full_inference"
os.makedirs(output_dir, exist_ok=True)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()
device = "cuda"

def select_ckpt(ckpt_dir, name="checkpoint"):
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    print(f"\nAvailable {name} checkpoints:")
    for idx, path in enumerate(ckpts):
        print(f"  [{idx}] {os.path.basename(path)}")
    choice = int(input(f"Select {name} checkpoint index: "))
    return ckpts[choice]

semantic_ckpt = select_ckpt(semantic_ckpt_dir, "semantic")
semantic_tag  = os.path.basename(semantic_ckpt).replace("semantic_predictor_", "").replace(".pt", "")
print("Using semantic checkpoint:", semantic_ckpt)
print("Semantic tag:", semantic_tag)

parts = semantic_tag.split("_")
feature_type, encoder_type, loss_type = parts[-3], parts[-2], parts[-1]
print(f"Feature: {feature_type}, Encoder: {encoder_type}, Loss: {loss_type}")

semantic_scaler = joblib.load(os.path.join(semantic_ckpt_dir, f"scaler_{semantic_tag}.pkl"))

from diffusers import AutoencoderKL, DDIMScheduler
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

from core_files.models import eegnet, shallownet, deepnet, tsconv, conformer, mlpnet
from training.train_semantic import WindowEncoderWrapper, ReshapeWrapper

output_dim = 77 * 768

if feature_type in ["DE", "PSD"]:
    input_dim = 62 * 5
    model = mlpnet(out_dim=output_dim, input_dim=input_dim).to(device)
elif feature_type == "windows":
    if encoder_type == "mlp":
        input_dim = 7 * 62 * 100
        model = mlpnet(out_dim=output_dim, input_dim=input_dim).to(device)
    elif encoder_type == "eegnet":
        model = WindowEncoderWrapper(eegnet(out_dim=output_dim, C=62, T=100), out_dim=output_dim).to(device)
    elif encoder_type == "shallownet":
        model = WindowEncoderWrapper(shallownet(out_dim=output_dim, C=62, T=100), out_dim=output_dim).to(device)
    elif encoder_type == "deepnet":
        model = WindowEncoderWrapper(deepnet(out_dim=output_dim, C=62, T=100), out_dim=output_dim).to(device)
    elif encoder_type == "tsconv":
        model = WindowEncoderWrapper(tsconv(out_dim=output_dim, C=62, T=100), out_dim=output_dim).to(device)
    elif encoder_type == "conformer":
        model = WindowEncoderWrapper(conformer(out_dim=output_dim), out_dim=output_dim).to(device)
    else:
        raise ValueError(f"Unknown encoder type {encoder_type}")
else:
    raise ValueError(f"Invalid feature type {feature_type}")

model = ReshapeWrapper(model).to(device)
ckpt = torch.load(semantic_ckpt, map_location=device)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print("Semantic predictor rebuilt and loaded.")

dana_diffusion = Diffusion(time_steps=500)

with open(eeg_feat_list_path, "r") as f:
    eeg_feat_files = [line.strip() for line in f]

eeg_feat_file = os.path.join(drive_root, "EEG_DE", eeg_feat_files[0])
print("Chosen EEG feature file:", eeg_feat_file)

parts = eeg_feat_files[0].split("/")
subj, block, fname = parts[0], parts[1], parts[2]
blip_caption_path = os.path.join(blip_text_root, block, fname.replace(".npy", ".txt"))

if os.path.exists(blip_caption_path):
    with open(blip_caption_path, "r") as f:
        blip_prompt = f.read().strip()
else:
    blip_prompt = "[Prompt not found]"
print("Associated BLIP caption:", blip_prompt)
print("BLIP caption file:", blip_caption_path)

eeg_feat = np.load(eeg_feat_file)
if eeg_feat.ndim > 1:
    eeg_feat = eeg_feat.reshape(-1)
eeg_feat_scaled = semantic_scaler.transform([eeg_feat])[0]
eeg_feat_tensor = torch.tensor(eeg_feat_scaled, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    semantic_pred = model(eeg_feat_tensor)
print("Semantic embedding shape:", semantic_pred.shape)

negative = semantic_pred.mean(dim=0, keepdim=True).float().to(device)

def run_inference():
    video = pipe(
        model=None,
        eeg=semantic_pred,
        negative_eeg=negative,
        latents=None,
        video_length=6,
        height=288,
        width=512,
        num_inference_steps=100,
        guidance_scale=12.5,
    ).videos

    print("Generated video tensor:", video.shape)

    frames = (video[0] * 255).clamp(0, 255).to(torch.uint8)
    frames = frames.permute(0, 2, 3, 1).cpu().numpy()
    if frames.shape[-1] > 3:
        frames = frames[..., :3]
    elif frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)

    mp4_path = os.path.join(output_dir, "sample_eeg2video_semantic_only.mp4")
    writer = imageio.get_writer(mp4_path, fps=3, codec="libx264")
    for f in frames:
        writer.append_data(f)
    writer.close()

    print("Video saved to:", mp4_path)
    print("EEG file:", eeg_feat_file)
    print("Prompt:", blip_prompt)

run_inference()
