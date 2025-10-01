# ==========================================
# Full Inference (EEG → Video via semantic predictor + diffusion)
# Generate videos for ALL embeddings in .npy file
# ==========================================
import os, gc, torch, numpy as np
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

from core.unet import UNet3DConditionModel
from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from core.util import save_videos_grid  # same helper they use


# ==========================================
# Config
# ==========================================
SUBJECT       = "sub1.npy"
FEATURE_TYPES = ["DE"]

CLASS_SUBSET     = [1, 10, 12, 16, 19, 23, 25, 31, 34, 39]

PRETRAINED_SD_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
FINETUNED_SD_PATH  = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final_subset1-10-12-16-19-23-25-31-34-39"
OUTPUT_DIR         = "/content/drive/MyDrive/EEG2Video_outputs/test_full_inference"
BLIP_TEXT_PATH     = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/BLIP_text.npy"
SEM_PATH           = "/content/drive/MyDrive/EEG2Video_outputs/semantic_embeddings/embeddings_semantic_predictor_DE_sub1_subset1-10-12-16-19-23-25-31-34-39.npy"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect(); torch.cuda.empty_cache()

# === Load caption array (optional reference) ===
blip_text = np.load(BLIP_TEXT_PATH, allow_pickle=True)  # shape (7,40,5)

# === Load pipeline ===
unet = UNet3DConditionModel.from_pretrained(
    FINETUNED_SD_PATH,
    subfolder="unet",
    torch_dtype=torch.float16
).to(device)

pipe = TuneAVideoPipeline.from_pretrained(
    PRETRAINED_SD_PATH,
    unet=unet,
    torch_dtype=torch.float16
).to(device)

pipe.enable_vae_slicing()

# ==========================================
# Negative embedding = mean of all predicted embeddings in subset
# ==========================================
sem_preds_all = np.load(SEM_PATH)   # shape (N,77,768)
negative_embedding = torch.tensor(
    sem_preds_all.mean(axis=0, keepdims=True),
    dtype=torch.float16
).to(device)
print("Negative embedding shape:", negative_embedding.shape)


# ==========================================
# Run inference
# ==========================================
def run_inference():
    sem_preds = np.load(SEM_PATH)   # shape (N,77,768)
    video_length, fps = 6, 3

    num_classes = len(CLASS_SUBSET)
    trials_per_class = 5
    sem_preds = sem_preds.reshape(num_classes, trials_per_class, 77, 768)

    test_block = 6  # block index for captions

    idx = 0
    for trial in range(trials_per_class):
        for ci, class_id in enumerate(CLASS_SUBSET):
            emb = sem_preds[ci, trial]
            caption = blip_text[test_block, class_id, trial]  # string already

            semantic_pred = torch.tensor(emb, dtype=torch.float16).unsqueeze(0).to(device)

            video = pipe(
                model=None,
                eeg=semantic_pred,
                negative_eeg=negative_embedding,
                latents=None,
                video_length=video_length,
                height=288,
                width=512,
                num_inference_steps=100,
                guidance_scale=12.5,
            ).videos

            out_path = os.path.join(OUTPUT_DIR, f"class{class_id}_trial{trial+1}.gif")
            save_videos_grid(video, out_path, fps=fps)

            print(f"Saved: {out_path}")
            print(f"Caption: {caption}")
            idx += 1

run_inference()
