# ==========================================
# Full Inference (EEG → Video, but using BLIP text → CLIP embeddings)
# ==========================================
import os, gc, torch, numpy as np
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

from core.unet import UNet3DConditionModel
from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from core.util import save_videos_grid  # helper they use

# ==========================================
# Config
# ==========================================
PRETRAINED_SD_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
FINETUNED_SD_PATH  = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final"
OUTPUT_DIR         = "/content/drive/MyDrive/EEG2Video_outputs/test_full_inference"
BLIP_TEXT_PATH     = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/BLIP_text.npy"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect(); torch.cuda.empty_cache()

# === Load BLIP caption ===
blip_text = np.load(BLIP_TEXT_PATH, allow_pickle=True)  # shape (7,40,5)
caption = blip_text[6, 1, 0]   # block 7, class 1, clip 0
print("Using caption:", caption)

# === Load tokenizer + text encoder ===
tokenizer   = CLIPTokenizer.from_pretrained(PRETRAINED_SD_PATH, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(PRETRAINED_SD_PATH, subfolder="text_encoder").to(device)

# Tokenize and encode
text_inputs = tokenizer(caption, padding="max_length", max_length=77, return_tensors="pt")
input_ids   = text_inputs.input_ids.to(device)
with torch.no_grad():
    clip_embeddings = text_encoder(input_ids)[0]  # (1,77,768)

# Negative embedding: encode empty string "" or " " (same trick SD uses)
neg_inputs = tokenizer("", padding="max_length", max_length=77, return_tensors="pt")
neg_ids    = neg_inputs.input_ids.to(device)
with torch.no_grad():
    neg_embeddings = text_encoder(neg_ids)[0]     # (1,77,768)

# === Load pipeline ===
# Finetuned UNet only
unet = UNet3DConditionModel.from_pretrained(
    FINETUNED_SD_PATH,
    subfolder="unet",
    torch_dtype=torch.float32
).to(device)

# Base SD backbone + custom UNet
pipe = TuneAVideoPipeline.from_pretrained(
    PRETRAINED_SD_PATH,
    unet=unet,
    torch_dtype=torch.float32
).to(device)
pipe.enable_vae_slicing()

def run_inference():
    # === Run pipeline ===
    video_length, fps = 6, 3  # 2 seconds
    video = pipe(
        model=None,
        eeg=clip_embeddings,         # use BLIP→CLIP embedding here
        negative_eeg=neg_embeddings, # unconditional embedding
        latents=None,
        video_length=video_length,
        height=288,
        width=512,
        num_inference_steps=100,
        guidance_scale=12.5,
    ).videos

    # === Save as GIF ===
    out_path = os.path.join(OUTPUT_DIR, "test_blip.gif")
    save_videos_grid(video, out_path, fps=fps)
    print("Saved video:", out_path)

    # Save caption alongside
    with open(os.path.join(OUTPUT_DIR, "test_blip.txt"), "w") as f:
        f.write(caption + "\n")

run_inference()
