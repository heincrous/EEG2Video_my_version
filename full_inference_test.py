# ==========================================
# Full Inference (Video generation using precomputed CLIP embeddings)
# ==========================================
import os, gc, torch, numpy as np
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
import re

from core.unet import UNet3DConditionModel
from pipelines.my_pipeline import TuneAVideoPipeline
from core.util import save_videos_grid


# ==========================================
# Config
# ==========================================
CLASS_SUBSET       = [0, 2, 4, 10, 11, 12, 22, 26, 29, 37]
PRETRAINED_SD_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
FINETUNED_SD_PATH  = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final_subset0-2-4-10-11-12-22-26-29-37_variants"
OUTPUT_DIR         = "/content/drive/MyDrive/EEG2Video_outputs/test_full_inference"
BLIP_TEXT_PATH     = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/BLIP_text.npy"
CLIP_EMB_PATH      = "/content/drive/MyDrive/EEG2Video_data/processed/CLIP_embeddings/CLIP_embeddings.npy"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === MEMORY CONFIG ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect(); torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load captions + CLIP embeddings ===
blip_text   = np.load(BLIP_TEXT_PATH, allow_pickle=True)       # (7,40,5)
clip_embs   = np.load(CLIP_EMB_PATH)                           # (7,40,5,77,768)
test_block  = 6
trials_per_class = 5

# === Build empty-string negative embedding ===
tokenizer    = CLIPTokenizer.from_pretrained(PRETRAINED_SD_PATH, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(PRETRAINED_SD_PATH, subfolder="text_encoder").to(device)
with torch.no_grad():
    empty_inputs = tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
    empty_emb    = text_encoder(empty_inputs.input_ids.to(device))[0]
neg_embeddings = empty_emb.to(torch.float16).to(device)

# === Load pipeline ===
pipe = TuneAVideoPipeline(
    vae=AutoencoderKL.from_pretrained(PRETRAINED_SD_PATH, subfolder="vae", torch_dtype=torch.float16),
    text_encoder=CLIPTextModel.from_pretrained(PRETRAINED_SD_PATH, subfolder="text_encoder", torch_dtype=torch.float16),
    tokenizer=CLIPTokenizer.from_pretrained(PRETRAINED_SD_PATH, subfolder="tokenizer"),
    unet=UNet3DConditionModel.from_pretrained_2d(FINETUNED_SD_PATH, subfolder="unet"),
    scheduler=DDIMScheduler.from_pretrained(PRETRAINED_SD_PATH, subfolder="scheduler"),
).to(device)
pipe.unet.to(torch.float16)
pipe.enable_vae_slicing()


# ==========================================
# Run inference over subset CLIP embeddings
# ==========================================
def run_inference():
    video_length, fps = 6, 3
    for class_id in CLASS_SUBSET:
        for trial in range(trials_per_class):
            emb      = clip_embs[test_block, class_id, trial]
            caption  = str(blip_text[test_block, class_id, trial])
            clip_emb = torch.tensor(emb, dtype=torch.float16).unsqueeze(0).to(device)

            video = pipe(
                prompt=clip_emb,
                negative_prompt=neg_embeddings,
                video_length=video_length,
                height=288,
                width=512,
                num_inference_steps=100,
                guidance_scale=12.5,
            ).videos

            # sanitize caption to safe filename
            safe_caption = re.sub(r'[^a-zA-Z0-9_-]', '_', caption)  
            out_gif = os.path.join(OUTPUT_DIR, f"{safe_caption}.gif")

            save_videos_grid(video, out_gif, fps=fps)
            print(f"Saved: {out_gif}")

run_inference()
