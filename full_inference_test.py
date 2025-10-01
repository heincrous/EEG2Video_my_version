# ==========================================
# Full Inference (EEG → Video, but using BLIP text → CLIP embeddings, subset mean negative)
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
FINETUNED_SD_PATH  = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final_subset1-10-12-16-19-23-25-31-34-39"
OUTPUT_DIR         = "/content/drive/MyDrive/EEG2Video_outputs/test_full_inference"
BLIP_TEXT_PATH     = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/BLIP_text.npy"

CLASS_SUBSET       = [1, 10, 12, 16, 19, 23, 25, 31, 34, 39]  # your chosen subset

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect(); torch.cuda.empty_cache()

# === Load BLIP captions ===
blip_text = np.load(BLIP_TEXT_PATH, allow_pickle=True)  # shape (7,40,5)
caption = blip_text[6, 1, 0]   # block 7, class=1, clip=0
print("Using caption:", caption)

# === Load tokenizer + text encoder ===
tokenizer    = CLIPTokenizer.from_pretrained(PRETRAINED_SD_PATH, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(PRETRAINED_SD_PATH, subfolder="text_encoder").to(device)

# --- Encode target caption ---
text_inputs = tokenizer(caption, padding="max_length", max_length=77, return_tensors="pt")
input_ids   = text_inputs.input_ids.to(device)
with torch.no_grad():
    clip_embeddings = text_encoder(input_ids)[0]  # (1,77,768)

# --- Build NEGATIVE embedding: mean over subset captions ---
all_embeddings = []
with torch.no_grad():
    for cls in CLASS_SUBSET:
        for clip_idx in range(5):  # 5 clips per class
            cap = blip_text[6, cls, clip_idx]   # block 7, subset class
            cap_inputs = tokenizer(cap, padding="max_length", max_length=77, return_tensors="pt")
            cap_ids    = cap_inputs.input_ids.to(device)
            emb = text_encoder(cap_ids)[0]  # (1,77,768)
            all_embeddings.append(emb)

neg_embeddings = torch.mean(torch.cat(all_embeddings, dim=0), dim=0, keepdim=True)  # (1,77,768)
print("Negative embedding shape:", neg_embeddings.shape)

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
    video_length, fps = 6, 3  # 2 seconds
    video = pipe(
        model=None,
        eeg=clip_embeddings,         # BLIP→CLIP embedding for chosen caption
        negative_eeg=None, # mean subset embedding
        latents=None,
        video_length=video_length,
        height=288,
        width=512,
        num_inference_steps=100,
        guidance_scale=12.5,
    ).videos

    out_path = os.path.join(OUTPUT_DIR, "test_blip.gif")
    save_videos_grid(video, out_path, fps=fps)
    print("Saved video:", out_path)

    with open(os.path.join(OUTPUT_DIR, "test_blip.txt"), "w") as f:
        f.write(caption + "\n")

run_inference()
