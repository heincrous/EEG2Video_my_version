# ==========================================
# EEG2Video – Semantic-Only Inference (Predictions → Video)
# ==========================================
import os, gc, re, shutil, torch, numpy as np
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from core.unet import UNet3DConditionModel
from pipelines.my_pipeline import TuneAVideoPipeline
from core.util import save_videos_grid


# ==========================================
# Config
# ==========================================
CLASS_SUBSET       = [0, 11, 24, 30, 33]
SEM_PATH           = f"/content/drive/MyDrive/EEG2Video_results/semantic_predictor/predictions/{'_'.join(map(str, CLASS_SUBSET))}.npy"
BLIP_TEXT_PATH     = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/BLIP_text.npy"
PRETRAINED_SD_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
OUTPUT_ROOT        = "/content/drive/MyDrive/EEG2Video_results/inference"

NUM_INFERENCE      = 50
GUIDANCE_SCALE     = 5


# ==========================================
# Output Directory
# ==========================================
subset_name = "_".join(map(str, CLASS_SUBSET))
OUTPUT_DIR  = os.path.join(OUTPUT_ROOT, subset_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# Cleanup
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
    print(f"🧹 Deleted {deleted} previous file(s) from {OUTPUT_DIR}.")


# ==========================================
# Memory Config
# ==========================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# Load Predictions + Captions
# ==========================================
print(f"Loading semantic predictor outputs from: {SEM_PATH}")
sem_preds_all = np.load(SEM_PATH, allow_pickle=True)   # [num_classes, 5, 77, 768]
blip_text     = np.load(BLIP_TEXT_PATH, allow_pickle=True)

num_classes      = sem_preds_all.shape[0]
trials_per_class = sem_preds_all.shape[1]
test_block       = 6
total_samples    = num_classes * trials_per_class

print(f"Loaded predictions shape: {sem_preds_all.shape}, total samples: {total_samples}")


# ==========================================
# Negative Embedding (Mean of Predictions)
# ==========================================
# Flatten [num_classes, trials_per_class, 77, 768] → [num_classes*trials_per_class, 77, 768]
flat_preds = sem_preds_all.reshape(-1, 77, 768)
mean_sem   = flat_preds.mean(axis=0)                # average over all samples → [77,768]
neg_embeddings = torch.tensor(mean_sem, dtype=torch.float16).unsqueeze(0).to(device)
print(f"Using mean of all predictions as negative embedding. Shape: {tuple(neg_embeddings.shape)}")


# ==========================================
# Diffusion Pipeline (Pretrained Only)
# ==========================================
pipe = TuneAVideoPipeline(
    vae=AutoencoderKL.from_pretrained(PRETRAINED_SD_PATH, subfolder="vae", torch_dtype=torch.float16),
    text_encoder=CLIPTextModel.from_pretrained(PRETRAINED_SD_PATH, subfolder="text_encoder", torch_dtype=torch.float16),
    tokenizer=CLIPTokenizer.from_pretrained(PRETRAINED_SD_PATH, subfolder="tokenizer"),
    unet=UNet3DConditionModel.from_pretrained_2d(PRETRAINED_SD_PATH, subfolder="unet"),
    scheduler=DDIMScheduler.from_pretrained(PRETRAINED_SD_PATH, subfolder="scheduler"),
).to(device)
pipe.unet.to(torch.float16)
pipe.enable_vae_slicing()


# ==========================================
# Inference
# ==========================================
def run_inference():
    video_length, fps = 6, 3
    flat_sem_preds = sem_preds_all.reshape(total_samples, 77, 768)
    print(f"Flattened semantic embeddings: {flat_sem_preds.shape}")

    sample_idx = 0
    for ci, class_id in enumerate(CLASS_SUBSET):
        print(f"\n[CLASS {class_id}] ------------------------------")
        for trial in range(trials_per_class):
            emb = flat_sem_preds[sample_idx]
            semantic_emb = torch.tensor(emb, dtype=torch.float16)
            if semantic_emb.ndim == 2:
                semantic_emb = semantic_emb.unsqueeze(0)
            semantic_emb = semantic_emb.to(device)
            caption = str(blip_text[test_block, class_id, trial])

            # Generate video
            video = pipe(
                prompt=semantic_emb,
                negative_prompt=neg_embeddings,
                video_length=video_length,
                height=288,
                width=512,
                num_inference_steps=NUM_INFERENCE,
                guidance_scale=GUIDANCE_SCALE,
            ).videos

            # Save GIF
            safe_caption = re.sub(r"[^a-zA-Z0-9_-]", "_", caption)
            if len(safe_caption) > 120:
                safe_caption = safe_caption[:120]

            out_gif = os.path.join(OUTPUT_DIR, f"class{class_id}_trial{trial}_{safe_caption}.gif")
            save_videos_grid(video, out_gif, fps=fps)
            print(f"Saved {out_gif}")

            sample_idx += 1


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    print(f"Running semantic-only inference for subset {subset_name}")
    print(f"Using pretrained Stable Diffusion | Guidance={GUIDANCE_SCALE}")
    cleanup_previous_outputs()
    run_inference()
