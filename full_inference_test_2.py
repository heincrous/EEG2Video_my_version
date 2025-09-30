# ==========================================
# Full Inference (EEG → Video, using CLIP_embeddings.npy)
# ==========================================
import os, gc, torch, numpy as np
from diffusers import AutoencoderKL, DDIMScheduler
from core.unet import UNet3DConditionModel
from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from core.util import save_videos_grid  # helper they use

# ==========================================
# Config
# ==========================================
PRETRAINED_SD_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
FINETUNED_SD_PATH  = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final_subset1-10-12-16-19-23-25-31-34-39"
OUTPUT_DIR         = "/content/drive/MyDrive/EEG2Video_outputs/test_full_inference"
CLIP_EMB_PATH      = "/content/drive/MyDrive/EEG2Video_data/processed/CLIP_embeddings/CLIP_embeddings.npy"

CLASS_SUBSET       = [1, 10, 12, 16, 19, 23, 25, 31, 34, 39]
BLOCK_INDEX        = 6   # test block (7th)

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect(); torch.cuda.empty_cache()

# === Load CLIP embeddings ===
clip_all = np.load(CLIP_EMB_PATH)  # shape (7,40,5,77,768)
print("CLIP embeddings shape:", clip_all.shape)

# Select test block + subset
embeds = []
for c in CLASS_SUBSET:
    embeds.append(clip_all[BLOCK_INDEX, c, :, :, :])  # (5,77,768)
embeds = np.stack(embeds, axis=0)  # (10,5,77,768)

# Choose one example, e.g. class 1 (index 0), trial 0
clip_embeddings = torch.tensor(embeds[0,0], dtype=torch.float32).unsqueeze(0).to(device)  # (1,77,768)

# Negative = mean over all 10×5 = 50 embeddings in block 6 subset
neg_embeddings = torch.tensor(embeds.reshape(-1,77,768).mean(axis=0), dtype=torch.float32).unsqueeze(0).to(device)  # (1,77,768)

print("Positive emb shape:", clip_embeddings.shape)
print("Negative emb shape:", neg_embeddings.shape)

# === Load pipeline ===
unet = UNet3DConditionModel.from_pretrained(
    FINETUNED_SD_PATH,
    subfolder="unet",
    torch_dtype=torch.float32
).to(device)

pipe = TuneAVideoPipeline.from_pretrained(
    PRETRAINED_SD_PATH,
    unet=unet,
    torch_dtype=torch.float32
).to(device)
pipe.enable_vae_slicing()

def run_inference():
    video_length, fps = 6, 3
    video = pipe(
        model=None,
        eeg=clip_embeddings,         # now CLIP_embeddings.npy
        negative_eeg=neg_embeddings, # unconditional = mean of subset test block
        latents=None,
        video_length=video_length,
        height=288,
        width=512,
        num_inference_steps=100,
        guidance_scale=12.5,
    ).videos

    out_path = os.path.join(OUTPUT_DIR, "test_clipemb.gif")
    save_videos_grid(video, out_path, fps=fps)
    print("Saved video:", out_path)

run_inference()
