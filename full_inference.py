# ==========================================
# Full Inference (Video generation using Semantic + Seq2Seq embeddings)
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
SEM_FEATURE_TYPE     = "EEG_DE_1per2s"          # used for semantic predictor embeddings
SEQ2SEQ_FEATURE_TYPE = "EEG_windows_100"        # used for Seq2Seq latent features
SUBJECT_NAME         = "sub1.npy"
CLASS_SUBSET         = [0, 9, 11, 15, 18, 22, 24, 30, 33, 38]
SUBSET_ID            = "1"

USE_SEQ2SEQ          = True
USE_DANA             = False                   # placeholder toggle for noisy latents

PRETRAINED_SD_PATH   = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
FINETUNED_SD_PATH    = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final_subset0-2-4-10-11-12-22-26-29-37_variants"
OUTPUT_ROOT          = "/content/drive/MyDrive/EEG2Video_outputs/full_inference"
BLIP_TEXT_PATH       = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/BLIP_text.npy"

# Semantic predictor outputs
SEM_PATH = f"/content/drive/MyDrive/EEG2Video_outputs/semantic_embeddings/pred_embeddings_{SEM_FEATURE_TYPE}_sub1_subset{SUBSET_ID}.npy"

# Seq2Seq latent outputs
SEQ2SEQ_LATENT_PATH = f"/content/drive/MyDrive/EEG2Video_outputs/seq2seq_latents/latent_out_{SEQ2SEQ_FEATURE_TYPE}_sub1_subset{SUBSET_ID}.npy"

# Negative embedding toggle: "empty" or "mean_sem"
NEGATIVE_MODE = "empty"

# Toggle between vanilla or finetuned diffusion
USE_FINETUNED = False


# ==========================================
# Dynamic Output Directory Naming
# ==========================================
mode_tag = "FullModel" if USE_SEQ2SEQ else "woSeq2Seq"
OUTPUT_DIR = os.path.join(
    OUTPUT_ROOT, f"{SEM_FEATURE_TYPE}_sem__{SEQ2SEQ_FEATURE_TYPE}_seq2seq_subset{SUBSET_ID}_{mode_tag}"
)
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
print(f"Loading semantic embeddings from {SEM_PATH}")
blip_text     = np.load(BLIP_TEXT_PATH, allow_pickle=True)   # (7, 40, 5)
sem_preds_all = np.load(SEM_PATH)                            # (N,77,768)
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
    raise ValueError(f"Unknown NEGATIVE_MODE {NEGATIVE_MODE}")


# ==========================================
# Load Diffusion Pipeline
# ==========================================
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
# Load Seq2Seq Latents (Optional)
# ==========================================
if USE_SEQ2SEQ:
    print(f"Loading Seq2Seq latents from {SEQ2SEQ_LATENT_PATH}")
    latents_seq2seq = np.load(SEQ2SEQ_LATENT_PATH, allow_pickle=True)  # (N,6,4,36,64)
    latents_seq2seq = torch.from_numpy(latents_seq2seq).half().to(device)
    latents_seq2seq = rearrange(latents_seq2seq, "b f c h w -> b c f h w")
else:
    latents_seq2seq = None


# ==========================================
# Inference Function
# ==========================================
def run_inference():
    video_length, fps = 6, 3
    sem_preds = sem_preds_all.reshape(num_classes, trials_per_class, 77, 768)

    for trial in range(trials_per_class):
        for ci, class_id in enumerate(CLASS_SUBSET):
            emb     = sem_preds[ci, trial]
            caption = str(blip_text[test_block, class_id, trial])
            semantic_emb = torch.tensor(emb, dtype=torch.float16).unsqueeze(0).to(device)

            # pick Seq2Seq latent if available
            if USE_SEQ2SEQ:
                idx = ci * trials_per_class + trial
                lat_input = latents_seq2seq[idx:idx+1]
            else:
                lat_input = None

            video = pipe(
                prompt=semantic_emb,
                negative_prompt=neg_embeddings,
                latents=lat_input,
                video_length=video_length,
                height=288,
                width=512,
                num_inference_steps=100,
                guidance_scale=12.5,
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
    print(f"Running inference for SEM={SEM_FEATURE_TYPE}, SEQ={SEQ2SEQ_FEATURE_TYPE}, subset={SUBSET_ID}")
    cleanup_previous_outputs()
    run_inference()
