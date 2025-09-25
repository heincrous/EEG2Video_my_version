# ==========================================
# Full Inference (Semantic Predictor auto-detection, random samples, all models)
# ==========================================

import os, sys, gc, glob, random, imageio, torch, shutil
import numpy as np
import joblib

from core_files.models import (
    eegnet, shallownet, deepnet, tsconv, conformer, mlpnet,
    glfnet, glfnet_mlp
)
from core_files.unet import UNet3DConditionModel
from training.train_semantic import WindowEncoderWrapper, ReshapeWrapper
from core_files.add_noise import Diffusion
from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline

drive_root            = "/content/drive/MyDrive/EEG2Video_data/processed"
pretrained_model_path = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
finetuned_model_path  = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final"
semantic_ckpt_dir     = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"

blip_text_root        = os.path.join(drive_root, "BLIP_text")
video_mp4_root        = os.path.join(drive_root, "Video_mp4")

output_dir            = "/content/drive/MyDrive/EEG2Video_outputs/test_full_inference"
os.makedirs(output_dir, exist_ok=True)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()
device = "cuda"

# === Helper: pick checkpoint ===
def select_ckpt(ckpt_dir, name="checkpoint"):
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    print(f"\nAvailable {name} checkpoints:")
    for idx, path in enumerate(ckpts):
        print(f"  [{idx}] {os.path.basename(path)}")
    choice = int(input(f"Select {name} checkpoint index: "))
    return ckpts[choice]

# === Load semantic predictor ===
semantic_ckpt = select_ckpt(semantic_ckpt_dir, "semantic")
semantic_tag  = os.path.basename(semantic_ckpt).replace("semantic_predictor_", "").replace(".pt", "")
print("Using semantic checkpoint:", semantic_ckpt)
print("Semantic tag:", semantic_tag)

parts = semantic_tag.split("_")

# --- Robust parser for feature/encoder/loss ---
feature_type, encoder_type, loss_type = None, None, None
if any(ft in parts for ft in ["DE", "PSD", "windows"]):
    for candidate in ["DE", "PSD", "windows"]:
        if candidate in parts:
            feature_type = candidate
            break

    # Handle glfnet_mlp special case
    if "glfnet" in parts and "mlp" in parts:
        encoder_type = "glfnet_mlp"
        idx = parts.index("mlp")
        loss_type = "_".join(parts[idx+1:]) if idx+1 < len(parts) else "unknown"
    else:
        idx = parts.index(feature_type)
        if idx + 1 < len(parts):
            encoder_type = parts[idx+1]
        if idx + 2 < len(parts):
            loss_type = "_".join(parts[idx+2:])
else:
    raise ValueError(f"Could not parse semantic tag: {semantic_tag}")

print(f"Feature: {feature_type}, Encoder: {encoder_type}, Loss: {loss_type}")

semantic_scaler = joblib.load(os.path.join(semantic_ckpt_dir, f"scaler_{semantic_tag}.pkl"))

# === Load diffusion backbone ===
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

# === Build semantic predictor model ===
output_dim = 77 * 768

if feature_type in ["DE", "PSD"]:
    input_dim = 62 * 5
    if encoder_type == "mlp":
        base_model = mlpnet(out_dim=output_dim, input_dim=input_dim)
    elif encoder_type == "glfnet":
        base_model = glfnet(out_dim=output_dim, emb_dim=256, C=62, T=5)
    elif encoder_type == "glfnet_mlp":
        base_model = glfnet_mlp(out_dim=output_dim, emb_dim=256, input_dim=input_dim)
    else:
        raise ValueError(f"Unknown encoder type {encoder_type}")
elif feature_type == "windows":
    if encoder_type == "mlp":
        input_dim = 7 * 62 * 100
        base_model = mlpnet(out_dim=output_dim, input_dim=input_dim)
    elif encoder_type == "eegnet":
        base_model = WindowEncoderWrapper(eegnet(out_dim=output_dim, C=62, T=100), out_dim=output_dim)
    elif encoder_type == "shallownet":
        base_model = WindowEncoderWrapper(shallownet(out_dim=output_dim, C=62, T=100), out_dim=output_dim)
    elif encoder_type == "deepnet":
        base_model = WindowEncoderWrapper(deepnet(out_dim=output_dim, C=62, T=100), out_dim=output_dim)
    elif encoder_type == "tsconv":
        base_model = WindowEncoderWrapper(tsconv(out_dim=output_dim, C=62, T=100), out_dim=output_dim)
    elif encoder_type == "conformer":
        base_model = conformer(out_dim=output_dim)
    else:
        raise ValueError(f"Unknown encoder type {encoder_type}")
else:
    raise ValueError(f"Invalid feature type {feature_type}")

model = ReshapeWrapper(base_model, n_tokens=77).to(device)
model.load_state_dict(torch.load(semantic_ckpt, map_location=device)["state_dict"])
model.eval()

# === Diffusion wrapper ===
dana_diffusion = Diffusion(time_steps=500)

# === Load test bundles and pick random sample ===
bundle_root = os.path.join(drive_root, "SubjectBundles")
test_bundles = sorted([f for f in os.listdir(bundle_root) if f.endswith("_test.npz")])
if not test_bundles:
    raise FileNotFoundError("No test bundles found!")

test_bundle = random.choice(test_bundles)
data = np.load(os.path.join(bundle_root, test_bundle), allow_pickle=True)
eeg_data = data[f"EEG_{feature_type}"]
keys = data["keys"]

idx = random.randrange(len(keys))
eeg_feat = eeg_data[idx]
key = keys[idx]
print("Chosen test sample:", test_bundle, key)

# === Parse key for naming and caption ===
parts = key.split("/")
if len(parts) == 3:
    _, block, fname = parts
elif len(parts) == 2:
    block, fname = parts
else:
    raise ValueError(f"Unexpected key format: {key}")

blip_caption_path = os.path.join(blip_text_root, block, fname.replace(".npy", ".txt"))
blip_prompt = "[Prompt not found]"
if os.path.exists(blip_caption_path):
    with open(blip_caption_path, "r") as f:
        blip_prompt = f.read().strip()

video_mp4_path = os.path.join(video_mp4_root, block, fname.replace(".npy", ".mp4"))

# === Prepare EEG features ===
eeg_flat = eeg_feat.reshape(-1)
eeg_scaled = semantic_scaler.transform([eeg_flat])[0]
if feature_type == "windows":
    eeg_tensor = torch.tensor(eeg_scaled.reshape(7,62,100), dtype=torch.float32).unsqueeze(0).to(device)
else:
    eeg_tensor = torch.tensor(eeg_scaled.reshape(62,5), dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    semantic_pred = model(eeg_tensor)

negative = semantic_pred.mean(dim=0, keepdim=True).float().to(device)

# === Run inference ===
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

    frames = (video[0] * 255).clamp(0, 255).to(torch.uint8)
    frames = frames.permute(0, 2, 3, 1).cpu().numpy()
    if frames.shape[-1] > 3:
        frames = frames[..., :3]
    elif frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)

    clip_name = fname.replace(".npy", "")
    inf_path = os.path.join(output_dir, f"{clip_name}_inference.mp4")
    gt_path  = os.path.join(output_dir, f"{clip_name}_groundtruth.mp4")

    writer = imageio.get_writer(inf_path, fps=3, codec="libx264")
    for f in frames:
        writer.append_data(f)
    writer.close()

    if os.path.exists(video_mp4_path):
        shutil.copy(video_mp4_path, gt_path)

    print("Saved:")
    print(" - Inference video:", inf_path)
    print(" - Ground-truth video:", gt_path)
    print(" - EEG sample key:", key)
    print(" - BLIP caption:", blip_prompt)

run_inference()
