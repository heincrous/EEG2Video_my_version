# # ==========================================
# # Full Inference (Video generation using semantic predictor embeddings)
# # ==========================================
# import os, gc, torch, numpy as np, re, shutil
# from diffusers import AutoencoderKL, DDIMScheduler
# from transformers import CLIPTokenizer, CLIPTextModel
# from core.unet import UNet3DConditionModel
# from pipelines.my_pipeline import TuneAVideoPipeline
# from core.util import save_videos_grid

# # ==========================================
# # Config
# # ==========================================
# FEATURE_TYPE       = "EEG_DE_1per2s"
# CLASS_SUBSET       = [0, 9, 11, 15, 18, 22, 24, 30, 33, 38]
# SUBSET_ID          = "1"

# PRETRAINED_SD_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
# FINETUNED_SD_PATH  = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final_subset0-2-4-10-11-12-22-26-29-37_variants"
# OUTPUT_ROOT        = "/content/drive/MyDrive/EEG2Video_outputs/full_inference"
# BLIP_TEXT_PATH     = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/BLIP_text.npy"

# # Semantic predictor outputs (identified by feature type and subset)
# SEM_PATH = f"/content/drive/MyDrive/EEG2Video_outputs/semantic_embeddings/pred_embeddings_{FEATURE_TYPE}_sub1_subset{SUBSET_ID}.npy"

# # Negative embedding toggle: "empty" or "mean_sem"
# NEGATIVE_MODE      = "mean_sem"

# # Toggle between vanilla or finetuned diffusion
# USE_FINETUNED = False

# # Output directory includes both feature type and subset
# OUTPUT_DIR = os.path.join(OUTPUT_ROOT, f"{FEATURE_TYPE}_subset_{SUBSET_ID}")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ==========================================
# # Cleanup Utility
# # ==========================================
# def cleanup_previous_outputs():
#     deleted = 0
#     for f in os.listdir(OUTPUT_DIR):
#         path = os.path.join(OUTPUT_DIR, f)
#         if os.path.isfile(path) or os.path.islink(path):
#             os.remove(path)
#             deleted += 1
#         elif os.path.isdir(path):
#             shutil.rmtree(path)
#             deleted += 1
#     print(f"🧹 Deleted {deleted} previous generation file(s) from {OUTPUT_DIR}.")

# # ==========================================
# # Memory Config
# # ==========================================
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# gc.collect(); torch.cuda.empty_cache()
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # ==========================================
# # Load captions + semantic predictor embeddings
# # ==========================================
# print(f"Loading embeddings from {SEM_PATH}")
# blip_text      = np.load(BLIP_TEXT_PATH, allow_pickle=True)   # (7,40,5)
# sem_preds_all  = np.load(SEM_PATH)                           # (N,77,768)
# num_classes    = len(CLASS_SUBSET)
# trials_per_class = 5
# test_block     = 6

# # ==========================================
# # Build negative embedding
# # ==========================================
# if NEGATIVE_MODE == "empty":
#     tokenizer    = CLIPTokenizer.from_pretrained(PRETRAINED_SD_PATH, subfolder="tokenizer")
#     text_encoder = CLIPTextModel.from_pretrained(PRETRAINED_SD_PATH, subfolder="text_encoder").to(device)
#     with torch.no_grad():
#         empty_inputs = tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
#         empty_emb    = text_encoder(empty_inputs.input_ids.to(device))[0]
#     neg_embeddings = empty_emb.to(torch.float16).to(device)
#     print("Using EMPTY STRING negative embedding.")

# elif NEGATIVE_MODE == "mean_sem":
#     mean_sem = sem_preds_all.mean(axis=0, keepdims=True)   # (1,77,768)
#     neg_embeddings = torch.tensor(mean_sem, dtype=torch.float16).to(device)
#     print("Using MEAN of produced semantic embeddings as negative embedding.")

# else:
#     raise ValueError(f"Unknown NEGATIVE_MODE {NEGATIVE_MODE}")

# # ==========================================
# # Load diffusion pipeline
# # ==========================================
# unet_path = FINETUNED_SD_PATH if USE_FINETUNED else PRETRAINED_SD_PATH
# unet_sub  = "unet" if USE_FINETUNED else "unet"

# pipe = TuneAVideoPipeline(
#     vae=AutoencoderKL.from_pretrained(PRETRAINED_SD_PATH, subfolder="vae", torch_dtype=torch.float16),
#     text_encoder=CLIPTextModel.from_pretrained(PRETRAINED_SD_PATH, subfolder="text_encoder", torch_dtype=torch.float16),
#     tokenizer=CLIPTokenizer.from_pretrained(PRETRAINED_SD_PATH, subfolder="tokenizer"),
#     unet=UNet3DConditionModel.from_pretrained_2d(unet_path, subfolder=unet_sub),
#     scheduler=DDIMScheduler.from_pretrained(PRETRAINED_SD_PATH, subfolder="scheduler"),
# ).to(device)
# pipe.unet.to(torch.float16)
# pipe.enable_vae_slicing()

# # ==========================================
# # Inference Function
# # ==========================================
# def run_inference():
#     video_length, fps = 6, 3
#     sem_preds = sem_preds_all.reshape(num_classes, trials_per_class, 77, 768)

#     for trial in range(trials_per_class):
#         for ci, class_id in enumerate(CLASS_SUBSET):
#             emb     = sem_preds[ci, trial]
#             caption = str(blip_text[test_block, class_id, trial])
#             semantic_emb = torch.tensor(emb, dtype=torch.float16).unsqueeze(0).to(device)

#             video = pipe(
#                 prompt=semantic_emb,
#                 negative_prompt=neg_embeddings,
#                 video_length=video_length,
#                 height=288,
#                 width=512,
#                 num_inference_steps=100,
#                 guidance_scale=12.5,
#             ).videos

#             safe_caption = re.sub(r'[^a-zA-Z0-9_-]', '_', caption)
#             if len(safe_caption) > 120:
#                 safe_caption = safe_caption[:120]

#             out_gif = os.path.join(OUTPUT_DIR, f"{safe_caption}.gif")
#             save_videos_grid(video, out_gif, fps=fps)
#             print(f"Saved: {out_gif}")

# # ==========================================
# # Main
# # ==========================================
# if __name__ == "__main__":
#     print(f"Running inference for {FEATURE_TYPE}, subset {SUBSET_ID}...")
#     cleanup_previous_outputs()
#     run_inference()

# ==========================================
# Full Inference (Video generation using semantic predictor embeddings)
# ==========================================
import os, gc, torch, numpy as np, re, shutil
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from core.unet import UNet3DConditionModel
from pipelines.my_pipeline import TuneAVideoPipeline
from core.util import save_videos_grid

# ==========================================
# Config
# ==========================================
FEATURE_TYPE       = "EEG_DE_1per2s"
CLASS_SUBSET       = [0, 9, 11, 15, 18, 22, 24, 30, 33, 38]
SUBSET_ID          = "1"

PRETRAINED_SD_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
FINETUNED_SD_PATH  = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final_subset0-2-4-10-11-12-22-26-29-37_variants"
OUTPUT_ROOT        = "/content/drive/MyDrive/EEG2Video_outputs/full_inference"
BLIP_TEXT_PATH     = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/BLIP_text.npy"

SEM_PATH = f"/content/drive/MyDrive/EEG2Video_outputs/semantic_embeddings/pred_embeddings_{FEATURE_TYPE}_sub1_subset{SUBSET_ID}.npy"

USE_FINETUNED = False

OUTPUT_DIR = os.path.join(OUTPUT_ROOT, f"{FEATURE_TYPE}_subset_{SUBSET_ID}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Cleanup Utility
# ==========================================
def cleanup_previous_outputs():
    deleted = 0
    for f in os.listdir(OUTPUT_DIR):
        path = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
            deleted += 1
        elif os.path.isdir(path):
            shutil.rmtree(path)
            deleted += 1
    print(f"🧹 Deleted {deleted} previous generation file(s) from {OUTPUT_DIR}.")

# ==========================================
# Memory Config
# ==========================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect(); torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# Load captions + semantic predictor embeddings
# ==========================================
print(f"Loading embeddings from {SEM_PATH}")
blip_text      = np.load(BLIP_TEXT_PATH, allow_pickle=True)   # (7,40,5)
sem_preds_all  = np.load(SEM_PATH)                            # (N,77,768)
num_classes    = len(CLASS_SUBSET)
trials_per_class = 5
test_block     = 6

# ==========================================
# Load diffusion pipeline
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
pipe.vae.to(torch.float16)
pipe.enable_vae_slicing()

# ==========================================
# Inference Function (biasing + stability fixes)
# ==========================================
def run_inference():
    video_length, fps = 8, 3
    sem_preds = sem_preds_all.reshape(num_classes, trials_per_class, 77, 768)

    # === Precompute class centroids ===
    class_means = sem_preds.mean(axis=1)
    class_means_t = torch.tensor(class_means, dtype=torch.float16, device=device)

    for trial in range(trials_per_class):
        for ci, class_id in enumerate(CLASS_SUBSET):
            emb = sem_preds[ci, trial]
            caption = str(blip_text[test_block, class_id, trial])

            # ==========================================
            # Weighted semantic conditioning (temperature + top-k)
            # ==========================================
            emb_t = torch.tensor(emb, dtype=torch.float16, device=device)
            sims = torch.nn.functional.cosine_similarity(
                emb_t.flatten().unsqueeze(0).to(torch.float32),
                class_means_t.flatten(start_dim=1).to(torch.float32),
                dim=1
            )

            tau = 2.5
            k = 3
            weights = torch.softmax(sims * tau, dim=0)
            topk = torch.topk(weights, k)
            mask = torch.zeros_like(weights)
            mask[topk.indices] = weights[topk.indices]
            mask = mask / mask.sum()

            bias_emb = torch.sum(mask[:, None, None] * class_means_t.to(torch.float32), dim=0)
            bias_emb = bias_emb.to(torch.float16)

            alpha = 0.15
            semantic_emb = (1 - alpha) * emb_t + alpha * bias_emb
            semantic_emb = semantic_emb.unsqueeze(0)

            # ==========================================
            # Class-conditioned negative embedding
            # ==========================================
            neg_class_mean = torch.mean(
                torch.stack([m for i, m in enumerate(class_means_t) if i != ci]),
                dim=0
            )
            neg_embeddings = neg_class_mean.unsqueeze(0)

            # ==========================================
            # Adaptive guidance based on confidence (clamped)
            # ==========================================
            conf = sims.max().item()
            guidance_scale = max(7.0, min(8.0 + 8.0 * conf, 10.0))

            # ==========================================
            # Diffusion generation (reduced steps)
            # ==========================================
            video = pipe(
                prompt=semantic_emb,
                negative_prompt=neg_embeddings,
                video_length=video_length,
                height=288,
                width=512,
                num_inference_steps=40,
                guidance_scale=guidance_scale,
            ).videos

            # ==========================================
            # Save output
            # ==========================================
            safe_caption = re.sub(r'[^a-zA-Z0-9_-]', '_', caption)
            if len(safe_caption) > 120:
                safe_caption = safe_caption[:120]

            out_gif = os.path.join(OUTPUT_DIR, f"{safe_caption}.gif")
            save_videos_grid(video, out_gif, fps=fps)
            print(f"Saved: {out_gif}")

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    print(f"Running inference for {FEATURE_TYPE}, subset {SUBSET_ID}...")
    cleanup_previous_outputs()
    run_inference()
