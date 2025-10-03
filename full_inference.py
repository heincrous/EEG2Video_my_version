# ==========================================
# Full Inference (Video generation using semantic predictor embeddings)
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
OUTPUT_DIR         = "/content/drive/MyDrive/EEG2Video_outputs/full_inference"
BLIP_TEXT_PATH     = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/BLIP_text.npy"
SEM_PATH           = "/content/drive/MyDrive/EEG2Video_outputs/semantic_embeddings/embeddings_semantic_predictor_DE_sub1_subset0-2-4-10-11-12-22-26-29-37.npy"

# Toggle between vanilla or finetuned diffusion
USE_FINETUNED = True   # set True to use your fine-tuned UNet

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === MEMORY CONFIG ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect(); torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load captions + semantic predictor embeddings ===
blip_text      = np.load(BLIP_TEXT_PATH, allow_pickle=True)   # (7,40,5)
sem_preds_all  = np.load(SEM_PATH)                           # (N,77,768)
num_classes    = len(CLASS_SUBSET)
trials_per_class = 5
test_block     = 6

# === Build empty-string negative embedding ===
tokenizer    = CLIPTokenizer.from_pretrained(PRETRAINED_SD_PATH, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(PRETRAINED_SD_PATH, subfolder="text_encoder").to(device)
with torch.no_grad():
    empty_inputs = tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
    empty_emb    = text_encoder(empty_inputs.input_ids.to(device))[0]
neg_embeddings = empty_emb.to(torch.float16).to(device)

# === Load pipeline ===
unet_path = FINETUNED_SD_PATH if USE_FINETUNED else PRETRAINED_SD_PATH
unet_sub  = "unet" if USE_FINETUNED else "unet"

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
# Run inference over semantic embeddings
# ==========================================
def run_inference():
    video_length, fps = 6, 3
    sem_preds = sem_preds_all.reshape(num_classes, trials_per_class, 77, 768)

    for trial in range(trials_per_class):
        for ci, class_id in enumerate(CLASS_SUBSET):
            emb     = sem_preds[ci, trial]
            caption = str(blip_text[test_block, class_id, trial])
            semantic_emb = torch.tensor(emb, dtype=torch.float16).unsqueeze(0).to(device)

            video = pipe(
                prompt=semantic_emb,
                negative_prompt=neg_embeddings,
                video_length=video_length,
                height=288,
                width=512,
                num_inference_steps=100,
                guidance_scale=12.5,
            ).videos

            safe_caption = re.sub(r'[^a-zA-Z0-9_-]', '_', caption)
            if len(safe_caption) > 120:
                safe_caption = safe_caption[:120]

            out_gif = os.path.join(OUTPUT_DIR, f"{safe_caption}.gif")
            save_videos_grid(video, out_gif, fps=fps)
            print(f"Saved: {out_gif}")

run_inference()
