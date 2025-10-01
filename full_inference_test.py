# # ==========================================
# # Full Inference (EEG → Video, but using BLIP text → CLIP embeddings, subset mean negative)
# # ==========================================
# import os, gc, torch, numpy as np
# from diffusers import AutoencoderKL, DDIMScheduler
# from transformers import CLIPTokenizer, CLIPTextModel

# from core.unet import UNet3DConditionModel
# from pipelines.my_pipeline import TuneAVideoPipeline
# from core.util import save_videos_grid  # helper they use

# # ==========================================
# # Config
# # ==========================================
# PRETRAINED_SD_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
# FINETUNED_SD_PATH  = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final_subset1-10-12-16-19-23-25-31-34-39"
# OUTPUT_DIR         = "/content/drive/MyDrive/EEG2Video_outputs/test_full_inference"
# BLIP_TEXT_PATH     = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/BLIP_text.npy"

# CLASS_SUBSET       = [1, 10, 12, 16, 19, 23, 25, 31, 34, 39]  # your chosen subset

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # === MEMORY CONFIG ===
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# gc.collect(); torch.cuda.empty_cache()
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # === Load BLIP captions ===
# blip_text = np.load(BLIP_TEXT_PATH, allow_pickle=True)  # shape (7,40,5)
# caption = blip_text[6, 1, 0]   # block 7, class=1, clip=0
# print("Using caption:", caption)

# # === Load tokenizer + text encoder ===
# tokenizer    = CLIPTokenizer.from_pretrained(PRETRAINED_SD_PATH, subfolder="tokenizer")
# text_encoder = CLIPTextModel.from_pretrained(PRETRAINED_SD_PATH, subfolder="text_encoder").to(device)

# # --- Encode target caption ---
# text_inputs = tokenizer(caption, padding="max_length", max_length=77, return_tensors="pt")
# input_ids   = text_inputs.input_ids.to(device)
# with torch.no_grad():
#     clip_embeddings = text_encoder(input_ids)[0]  # (1,77,768)

# # --- Build NEGATIVE embedding: mean over subset captions ---
# all_embeddings = []
# with torch.no_grad():
#     for cls in CLASS_SUBSET:
#         for clip_idx in range(5):  # 5 clips per class
#             cap = blip_text[6, cls, clip_idx]   # block 7, subset class
#             cap_inputs = tokenizer(cap, padding="max_length", max_length=77, return_tensors="pt")
#             cap_ids    = cap_inputs.input_ids.to(device)
#             emb = text_encoder(cap_ids)[0]  # (1,77,768)
#             all_embeddings.append(emb)

# neg_embeddings = torch.mean(torch.cat(all_embeddings, dim=0), dim=0, keepdim=True)  # (1,77,768)
# print("Negative (mean) embedding shape:", neg_embeddings.shape)

# # === Save both negative variants ===
# save_dir = "/content/drive/MyDrive/EEG2Video_checkpoints"
# os.makedirs(save_dir, exist_ok=True)

# # Save mean embedding
# np.save(os.path.join(save_dir, "negative_mean.npy"), neg_embeddings.detach().cpu().numpy())

# # Save empty-prompt embedding
# with torch.no_grad():
#     empty_inputs = tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
#     empty_ids    = empty_inputs.input_ids.to(device)
#     empty_emb    = text_encoder(empty_ids)[0]  # (1,77,768)
# np.save(os.path.join(save_dir, "negative_empty.npy"), empty_emb.detach().cpu().numpy())

# print("Saved negative_mean.npy and negative_empty.npy to", save_dir)

# # === Load pipeline ===
# pipe = TuneAVideoPipeline(
#     vae=AutoencoderKL.from_pretrained(PRETRAINED_SD_PATH, subfolder="vae", torch_dtype=torch.float16),
#     text_encoder=CLIPTextModel.from_pretrained(PRETRAINED_SD_PATH, subfolder="text_encoder", torch_dtype=torch.float16),
#     tokenizer=CLIPTokenizer.from_pretrained(PRETRAINED_SD_PATH, subfolder="tokenizer"),
#     unet=UNet3DConditionModel.from_pretrained_2d(FINETUNED_SD_PATH, subfolder="unet"),
#     scheduler=DDIMScheduler.from_pretrained(PRETRAINED_SD_PATH, subfolder="scheduler"),
# )
# pipe.unet.to(torch.float16)
# pipe.enable_vae_slicing()
# pipe = pipe.to(device)

# def run_inference():
#     fps = 3  # 2 seconds
#     video = pipe(
#         prompt=clip_embeddings.to(device).to(torch.float16),          # precomputed [1,77,768]
#         negative_prompt=empty_emb.to(device).to(torch.float16),       # unconditional [1,77,768]
#         video_length=6,
#         height=288,
#         width=512,
#         num_inference_steps=100,
#         guidance_scale=12.5,
#     ).videos

#     out_path = os.path.join(OUTPUT_DIR, "test_blip.gif")
#     save_videos_grid(video, out_path, fps=fps)
#     print("Saved video:", out_path)

#     with open(os.path.join(OUTPUT_DIR, "test_blip.txt"), "w") as f:
#         f.write(caption + "\n")

# run_inference()

# ==========================================
# Full Inference (EEG → Video, but using precomputed CLIP embeddings, subset mean negative)
# ==========================================
import os, gc, torch, numpy as np
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

from core.unet import UNet3DConditionModel
from pipelines.my_pipeline import TuneAVideoPipeline
from core.util import save_videos_grid

# ==========================================
# Config
# ==========================================
CLASS_SUBSET       = [1, 10, 12, 16, 19, 23, 25, 31, 34, 39]

PRETRAINED_SD_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
FINETUNED_SD_PATH  = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final_subset1-10-12-16-19-23-25-31-34-39"
OUTPUT_DIR         = "/content/drive/MyDrive/EEG2Video_outputs/test_full_inference"
BLIP_TEXT_PATH     = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/BLIP_text.npy"
CLIP_EMB_PATH      = "/content/drive/MyDrive/EEG2Video_data/processed/CLIP_embeddings/CLIP_embeddings.npy"   # precomputed CLIP embeddings
NEG_PATH           = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_negative.npy"     # optional saved negative

NEG_MODE = "empty"   # choose: "empty", "mean", or "eeg"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === MEMORY CONFIG ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect(); torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load BLIP captions (for reference) ===
blip_text = np.load(BLIP_TEXT_PATH, allow_pickle=True)  # shape (7,40,5)
test_block = 6

# === Load precomputed CLIP embeddings ===
# expected shape: (7,40,5,77,768)
clip_embs_all = np.load(CLIP_EMB_PATH)  
print("Loaded CLIP embeddings shape:", clip_embs_all.shape)

# === Load pipeline ===
pipe = TuneAVideoPipeline(
    vae=AutoencoderKL.from_pretrained(PRETRAINED_SD_PATH, subfolder="vae", torch_dtype=torch.float16),
    text_encoder=CLIPTextModel.from_pretrained(PRETRAINED_SD_PATH, subfolder="text_encoder", torch_dtype=torch.float16),
    tokenizer=CLIPTokenizer.from_pretrained(PRETRAINED_SD_PATH, subfolder="tokenizer"),
    unet=UNet3DConditionModel.from_pretrained_2d(FINETUNED_SD_PATH, subfolder="unet"),
    scheduler=DDIMScheduler.from_pretrained(PRETRAINED_SD_PATH, subfolder="scheduler"),
)
pipe.unet.to(torch.float16)
pipe.enable_vae_slicing()
pipe = pipe.to(device)

# === Build negative embedding ===
if NEG_MODE == "mean":
    neg_embeddings = torch.tensor(
        clip_embs_all[test_block, CLASS_SUBSET].mean(axis=(0,1), keepdims=True),
        dtype=torch.float16
    ).to(device)
    print("Negative embedding mode: MEAN")

elif NEG_MODE == "empty":
    tokenizer = CLIPTokenizer.from_pretrained(PRETRAINED_SD_PATH, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(PRETRAINED_SD_PATH, subfolder="text_encoder").to(device)
    with torch.no_grad():
        empty_inputs = tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
        empty_ids    = empty_inputs.input_ids.to(device)
        empty_emb    = text_encoder(empty_ids)[0]
    neg_embeddings = empty_emb.to(torch.float16).to(device)
    print("Negative embedding mode: EMPTY STRING")

elif NEG_MODE == "eeg":
    neg_np = np.load(NEG_PATH)  # shape (1,77,768)
    neg_embeddings = torch.tensor(neg_np, dtype=torch.float16).to(device)
    print(f"Negative embedding mode: EEG FILE → {NEG_PATH}")

else:
    raise ValueError("NEG_MODE must be 'empty', 'mean', or 'eeg'.")

# ==========================================
# Run inference over all CLIP embeddings
# ==========================================
def run_inference():
    video_length, fps = 6, 3
    trials_per_class = 5

    for trial in range(trials_per_class):
        for class_id in CLASS_SUBSET:
            emb = clip_embs_all[test_block, class_id, trial]  # (77,768)
            caption = blip_text[test_block, class_id, trial]

            clip_pred = torch.tensor(emb, dtype=torch.float16).unsqueeze(0).to(device)

            video = pipe(
                prompt=clip_pred,
                negative_prompt=neg_embeddings,
                video_length=video_length,
                height=288,
                width=512,
                num_inference_steps=100,
                guidance_scale=12.5,
            ).videos

            out_path = os.path.join(OUTPUT_DIR, f"class{class_id}_trial{trial+1}_{NEG_MODE}.gif")
            save_videos_grid(video, out_path, fps=fps)

            print(f"Saved: {out_path}")
            print(f"Caption: {caption}")

run_inference()
