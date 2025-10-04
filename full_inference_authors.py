# ==========================================
# Full Inference (Video generation using semantic predictor embeddings, GT label aligned)
# ==========================================
import os, gc, torch, numpy as np, re
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from core.unet import UNet3DConditionModel
from pipelines.my_pipeline import TuneAVideoPipeline
from core.util import save_videos_grid

# ==========================================
# Config
# ==========================================
CLASS_SUBSET       = [1, 3, 5, 11, 12, 13, 23, 27, 30, 38]  # already GT label numbering
PRETRAINED_SD_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
FINETUNED_SD_PATH  = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final_subset0-2-4-10-11-12-22-26-29-37_variants"
OUTPUT_DIR         = "/content/drive/MyDrive/EEG2Video_outputs/full_inference"
BLIP_TEXT_PATH     = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/BLIP_text.npy"
SEM_PATH           = "/content/drive/MyDrive/EEG2Video_outputs/semantic_embeddings/embeddings_subset1-3-5-11-12-13-23-27-30-38_block6.npy"

# Negative embedding toggle: "empty", "mean_eeg", "mean_sem"
NEGATIVE_MODE      = "empty"
USE_FINETUNED      = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === MEMORY CONFIG ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect(); torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# Load captions + semantic predictor embeddings
# ==========================================
blip_text      = np.load(BLIP_TEXT_PATH, allow_pickle=True)   # (7,40,5)
sem_preds_all  = np.load(SEM_PATH)                            # (subset_classes*5, 77, 768)
num_classes    = len(CLASS_SUBSET)
trials_per_class = 5
test_block     = 6

# ==========================================
# GT label matrix (authors' original)
# ==========================================
GT_label = np.array([
    [23,22,9,6,18,14,5,36,25,19,28,35,3,16,24,40,15,27,38,33,34,4,39,17,1,26,20,29,13,32,37,2,11,12,30,31,8,21,7,10],
    [27,33,22,28,31,12,38,4,18,17,35,39,40,5,24,32,15,13,2,16,34,25,19,30,23,3,8,29,7,20,11,14,37,6,21,1,10,36,26,9],
    [15,36,31,1,34,3,37,12,4,5,21,24,14,16,39,20,28,29,18,32,2,27,8,19,13,10,30,40,17,26,11,9,33,25,35,7,38,22,23,6],
    [16,28,23,1,39,10,35,14,19,27,37,31,5,18,11,25,29,13,20,24,7,34,26,4,40,12,8,22,21,30,17,2,38,9,3,36,33,6,32,15],
    [18,29,7,35,22,19,12,36,8,15,28,1,34,23,20,13,37,9,16,30,2,33,27,21,14,38,10,17,31,3,24,39,11,32,4,25,40,5,26,6],
    [29,16,1,22,34,39,24,10,8,35,27,31,23,17,2,15,25,40,3,36,26,6,14,37,9,12,19,30,5,28,32,4,13,18,21,20,7,11,33,38],
    [38,34,40,10,28,7,1,37,22,9,16,5,12,36,20,30,6,15,35,2,31,26,18,24,8,3,23,19,14,13,21,4,25,11,32,17,39,29,33,27]
])

# ==========================================
# Build negative embedding
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
    print("Using MEAN semantic embedding as negative prompt.")
else:
    raise ValueError(f"Unknown NEGATIVE_MODE {NEGATIVE_MODE}")

# ==========================================
# Load pipeline
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
# Run inference (GT label aligned)
# ==========================================
def run_inference():
    video_length, fps = 6, 3
    sem_preds = sem_preds_all.reshape(num_classes, trials_per_class, 77, 768)

    for ci, class_id in enumerate(CLASS_SUBSET):
        # find the correct index for this class in GT_label order
        true_idx = list(GT_label[test_block]).index(class_id)
        for trial in range(trials_per_class):
            emb = sem_preds[ci, trial]
            caption = str(blip_text[test_block, true_idx, trial])
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

            out_gif = os.path.join(OUTPUT_DIR, f"{class_id:02d}_{safe_caption}.gif")
            save_videos_grid(video, out_gif, fps=fps)
            print(f"Saved: {out_gif}")

run_inference()
