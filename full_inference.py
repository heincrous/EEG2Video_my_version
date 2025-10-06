# ==========================================
# Full Inference (Video generation using Semantic embeddings)
# ==========================================
import os, gc, re, shutil, torch, numpy as np
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from core.unet import UNet3DConditionModel
from pipelines.my_pipeline import TuneAVideoPipeline
from core.util import save_videos_grid
from einops import rearrange


# ==========================================
# Config
# ==========================================
FEATURE_FUSION     = []  # leave empty for single feature
SEM_FEATURE_TYPE   = "EEG_DE_1per2s"  # fallback single feature if fusion empty
SUBJECT_NAME       = "sub1.npy"
CLASS_SUBSET       = [0, 9, 11, 15, 18, 22, 24, 30, 33, 38]
SUBSET_ID          = "1"

# Feature fusion tag
SEM_TAG = "_".join(FEATURE_FUSION) if FEATURE_FUSION else SEM_FEATURE_TYPE

PRETRAINED_SD_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
FINETUNED_SD_PATH  = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final_subset0-2-4-10-11-12-22-26-29-37_variants"
OUTPUT_ROOT        = "/content/drive/MyDrive/EEG2Video_outputs/full_inference"
BLIP_TEXT_PATH     = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/BLIP_text.npy"
SEM_PATH           = f"/content/drive/MyDrive/EEG2Video_outputs/semantic_embeddings/pred_embeddings_{SEM_TAG}_sub1_subset{SUBSET_ID}.npy"

NEGATIVE_MODE      = "mean_sem"  # or "empty"
USE_FINETUNED      = False


# ==========================================
# Inference Enhancements (toggles)
# ==========================================
USE_NORMALIZATION       = True     # normalize semantic embeddings to unit norm
USE_TEMPERATURE_SCALING = False     # rescale embeddings to sharpen distribution
TEMPERATURE_VALUE       = 0.7      # lower → sharper (0.5–0.9 typical)
USE_PROTOTYPE_BLENDING  = False     # blend with per-class mean
BLEND_ALPHA             = 0.6      # 0.4–0.7 typical blend ratio


# ==========================================
# Output Directory
# ==========================================
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, f"{SEM_TAG}_subset{SUBSET_ID}")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# Cleanup Utility
# ==========================================
def cleanup_previous_outputs():
    deleted = 0
    for f in os.listdir(OUTPUT_DIR):
        path = os.path.join(OUTPUT_DIR, f)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
                deleted += 1
            elif os.path.isdir(path):
                shutil.rmtree(path)
                deleted += 1
        except Exception as e:
            print(f"⚠️ Failed to delete {path}: {e}")
    print(f"🧹 Deleted {deleted} previous generation file(s) from {OUTPUT_DIR}.")


# ==========================================
# Memory Config
# ==========================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect(); torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# Load Captions + Semantic Predictor Embeddings
# ==========================================
print(f"Loading semantic embeddings from: {SEM_PATH}")
blip_text     = np.load(BLIP_TEXT_PATH, allow_pickle=True)
sem_preds_all = np.load(SEM_PATH)
num_classes   = len(CLASS_SUBSET)
trials_per_class = 5
test_block    = 6


# ==========================================
# Build Negative Embedding
# ==========================================
if NEGATIVE_MODE == "empty":
    tokenizer    = CLIPTokenizer.from_pretrained(PRETRAINED_SD_PATH, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(PRETRAINED_SD_PATH, subfolder="text_encoder").to(device)
    with torch.no_grad():
        empty_inputs = tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
        empty_emb    = text_encoder(empty_inputs.input_ids.to(device))[0]
    neg_embeddings = empty_emb.to(torch.float16).to(device)
    print("Using EMPTY STRING negative embedding.")
elif NEGATIVE_MODE == "mean_sem":
    mean_sem = sem_preds_all.mean(axis=0, keepdims=True)
    neg_embeddings = torch.tensor(mean_sem, dtype=torch.float16).to(device)
    print("Using MEAN of produced semantic embeddings as negative embedding.")
else:
    raise ValueError(f"Unknown NEGATIVE_MODE: {NEGATIVE_MODE}")


# ==========================================
# Diffusion Pipeline
# ==========================================
unet_path = FINETUNED_SD_PATH if USE_FINETUNED else PRETRAINED_SD_PATH
pipe = TuneAVideoPipeline(
    vae=AutoencoderKL.from_pretrained(PRETRAINED_SD_PATH, subfolder="vae", torch_dtype=torch.float16),
    text_encoder=CLIPTextModel.from_pretrained(PRETRAINED_SD_PATH, subfolder="text_encoder", torch_dtype=torch.float16),
    tokenizer=CLIPTokenizer.from_pretrained(PRETRAINED_SD_PATH, subfolder="tokenizer"),
    unet=UNet3DConditionModel.from_pretrained_2d(unet_path, subfolder="unet"),
    scheduler=DDIMScheduler.from_pretrained(PRETRAINED_SD_PATH, subfolder="scheduler"),
).to(device)
pipe.unet.to(torch.float16)
pipe.enable_vae_slicing()


# ==========================================
# Inference Function
# ==========================================
def run_inference():
    video_length, fps = 6, 3
    sem_preds = sem_preds_all.reshape(num_classes, trials_per_class, 77, 768)

    # Compute per-class prototypes if blending enabled
    if USE_PROTOTYPE_BLENDING:
        class_means = sem_preds.mean(axis=1, keepdims=True)
        print("Using per-class prototype blending.")

    for trial in range(trials_per_class):
        for ci, class_id in enumerate(CLASS_SUBSET):
            emb = sem_preds[ci, trial]

            # Prototype blending
            if USE_PROTOTYPE_BLENDING:
                emb = BLEND_ALPHA * emb + (1 - BLEND_ALPHA) * class_means[ci, 0]

            # Normalization
            if USE_NORMALIZATION:
                emb = emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8)

            # Temperature scaling
            if USE_TEMPERATURE_SCALING:
                emb = emb / TEMPERATURE_VALUE

            caption = str(blip_text[test_block, class_id, trial])
            semantic_emb = torch.tensor(emb, dtype=torch.float16).unsqueeze(0).to(device)

            video = pipe(
                prompt=semantic_emb,
                negative_prompt=neg_embeddings,
                latents=None,
                video_length=video_length,
                height=288,
                width=512,
                num_inference_steps=50,
                guidance_scale=25,  # 12.5
            ).videos

            safe_caption = re.sub(r"[^a-zA-Z0-9_-]", "_", caption)
            if len(safe_caption) > 120:
                safe_caption = safe_caption[:120]

            out_gif = os.path.join(OUTPUT_DIR, f"{safe_caption}.gif")
            save_videos_grid(video, out_gif, fps=fps)
            print(f"Saved: {out_gif}")


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    print(f"Running inference for SEM={SEM_TAG}, subset={SUBSET_ID}")
    print(f"Settings → Norm: {USE_NORMALIZATION}, Temp: {USE_TEMPERATURE_SCALING}({TEMPERATURE_VALUE}), "
          f"ProtoBlend: {USE_PROTOTYPE_BLENDING}({BLEND_ALPHA})")
    cleanup_previous_outputs()
    run_inference()
