# ==========================================
# Full Inference (Video generation using BLIP captions directly)
# ==========================================
import os, gc, torch, numpy as np
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
import re

from core.unet import UNet3DConditionModel
from pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from core.util import save_videos_grid

# ==========================================
# Config
# ==========================================
CLASS_SUBSET       = [0, 2, 4, 10, 11, 12, 22, 26, 29, 37]  # choose your classes here
PRETRAINED_SD_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
FINETUNED_SD_PATH  = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final_subset0-2-4-10-11-12-22-26-29-37_variants"
OUTPUT_DIR         = "/content/drive/MyDrive/EEG2Video_outputs/test_full_inference_captions"
BLIP_TEXT_PATH     = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/BLIP_text.npy"

# Toggle between vanilla or finetuned diffusion
USE_FINETUNED = False   # set True to use your fine-tuned UNet

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === MEMORY CONFIG ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect(); torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load captions ===
blip_text   = np.load(BLIP_TEXT_PATH, allow_pickle=True)       # (7,40,5)
test_block  = 6
trials_per_class = 1   # one caption per class

# === Load pipeline ===
unet_path = FINETUNED_SD_PATH if USE_FINETUNED else PRETRAINED_SD_PATH
unet_sub  = "unet"

pipe = TuneAVideoPipeline(
    vae=AutoencoderKL.from_pretrained(PRETRAINED_SD_PATH, subfolder="vae", torch_dtype=torch.float16),
    text_encoder=CLIPTextModel.from_pretrained(PRETRAINED_SD_PATH, subfolder="text_encoder", torch_dtype=torch.float16),
    tokenizer=CLIPTokenizer.from_pretrained(PRETRAINED_SD_PATH, subfolder="tokenizer"),
    unet=UNet3DConditionModel.from_pretrained_2d(unet_path, subfolder=unet_sub),
    scheduler=DDIMScheduler.from_pretrained(PRETRAINED_SD_PATH, subfolder="scheduler"),
).to(device)
pipe.unet.to(torch.float16)
pipe.enable_vae_slicing()

# ==========================================
# Run inference over subset captions
# ==========================================
def run_inference():
    video_length, fps = 6, 3
    for class_id in CLASS_SUBSET:
        for trial in range(trials_per_class):
            caption = str(blip_text[test_block, class_id, trial])

            result = pipe(
                prompt=caption,              # pass caption string
                video_length=video_length,
                height=288,
                width=512,
                num_inference_steps=100,
                guidance_scale=12.5,
            )

            video = result.videos

            safe_caption = re.sub(r'[^a-zA-Z0-9_-]', '_', caption)
            if len(safe_caption) > 120:
                safe_caption = safe_caption[:120]

            out_gif = os.path.join(OUTPUT_DIR, f"{safe_caption}.gif")
            save_videos_grid(video, out_gif, fps=fps)
            print(f"Saved: {out_gif}")

run_inference()
