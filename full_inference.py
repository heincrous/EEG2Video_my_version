# ==========================================
# Full Inference: EEG2Video (all subset videos)
# Save outputs as block_class_clip_sample.mp4
# ==========================================
import os, gc, torch, numpy as np, imageio
from einops import rearrange
from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTokenizer
from core.unet import UNet3DConditionModel


# ==========================================
# Paths
# ==========================================
SEMANTIC_DIR = "/content/drive/MyDrive/EEG2Video_outputs/semantic_embeddings"
SEQ2SEQ_DIR  = "/content/drive/MyDrive/EEG2Video_outputs/seq2seq_latents"
DANA_DIR     = "/content/drive/MyDrive/EEG2Video_outputs/dana_latents"
PIPELINE_DIR = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints"
BLIP_TEXT    = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/BLIP_text.npy"

SAVE_ROOT    = "/content/drive/MyDrive/EEG2Video_outputs/final_videos"
os.makedirs(SAVE_ROOT, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
gc.collect(); torch.cuda.empty_cache()


# ==========================================
# Select semantic embeddings file
# ==========================================
sem_files = [f for f in os.listdir(SEMANTIC_DIR) if f.endswith(".npy")]
print("Available semantic embeddings:")
for i,f in enumerate(sem_files):
    print(f"[{i}] {f}")
sem_choice = int(input("Select semantic embeddings index: "))
sem_file   = sem_files[sem_choice]
sem_path   = os.path.join(SEMANTIC_DIR, sem_file)

# deduce subject + subset + pipeline
subject_tag = next((p for p in sem_file.replace(".npy","").split("_") if p.startswith("sub")), None)
if "subset" in sem_file:
    subset_str   = sem_file.split("subset")[1].replace(".npy","")
    class_subset = [int(x) for x in subset_str.split("-")]
    pipeline_path = os.path.join(PIPELINE_DIR, f"pipeline_final_subset{subset_str}")
    pipeline_tag  = f"pipeline_subset{subset_str}"
else:
    class_subset = list(range(40))
    pipeline_path = os.path.join(PIPELINE_DIR, "pipeline_final")
    pipeline_tag  = "pipeline_full"

print("Semantic file:", sem_file)
print("Subject:", subject_tag)
print("Class subset:", class_subset)
print("Pipeline path:", pipeline_path)


# ==========================================
# Select Seq2Seq and DANA latents
# ==========================================
lat_files = [f for f in os.listdir(SEQ2SEQ_DIR) if subject_tag in f]
print("Available Seq2Seq latents:")
for i,f in enumerate(lat_files): print(f"[{i}] {f}")
lat_choice = int(input("Select Seq2Seq latents index: "))
lat_path   = os.path.join(SEQ2SEQ_DIR, lat_files[lat_choice])

dana_files = [f for f in os.listdir(DANA_DIR) if subject_tag in f]
print("Available DANA latents:")
for i,f in enumerate(dana_files): print(f"[{i}] {f}")
dana_choice = int(input("Select DANA latents index: "))
dana_path   = os.path.join(DANA_DIR, dana_files[dana_choice])


# ==========================================
# Load semantic embeddings
# ==========================================
eeg_embeds = np.load(sem_path)  # (N,77,768)
eeg_embeds = torch.from_numpy(eeg_embeds).to(device, dtype=torch.float32)
negative   = eeg_embeds.mean(dim=0, keepdim=True)

# ==========================================
# Load latents
# ==========================================
latents = np.load(lat_path)   # (B,F,C,H,W)
latents = np.repeat(latents, 2, axis=0)
latents = torch.from_numpy(latents).to(device, dtype=torch.float32).permute(0,2,1,3,4)

latents_add_noise = np.load(dana_path)   # (B,F,C,H,W)
latents_add_noise = np.repeat(latents_add_noise, 2, axis=0)  # double samples
latents_add_noise = torch.from_numpy(latents_add_noise).to(device, dtype=torch.float32).permute(0,2,1,3,4)

assert eeg_embeds.shape[0] == latents.shape[0] == latents_add_noise.shape[0]


# ==========================================
# Load pipeline (pretrained backbone + finetuned UNet)
# ==========================================
PRETRAINED_MODEL_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"

vae       = AutoencoderKL.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="vae").to(device, dtype=torch.float32)
scheduler = DDIMScheduler.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="tokenizer")

unet = UNet3DConditionModel.from_pretrained(
    pipeline_path,
    subfolder="unet",
    torch_dtype=torch.float32
).to(device)

pipe = TuneAVideoPipeline.from_pretrained(
    PRETRAINED_MODEL_PATH,
    vae=vae,
    unet=unet,
    scheduler=scheduler,
    tokenizer=tokenizer,
    torch_dtype=torch.float32
).to(device)

pipe.enable_vae_slicing()


# ==========================================
# Captions
# ==========================================
blip_text = np.load(BLIP_TEXT, allow_pickle=True)  # (7,40,5)
block7_caps = blip_text[6]  # (40,5)


# ==========================================
# Inference
# ==========================================
woSeq2Seq = False
woDANA    = False

for i in range(len(eeg_embeds)):
    cls_index    = i // 10
    clip_index   = (i % 10) // 2
    sample_index = i % 2
    class_id     = class_subset[cls_index]
    block_id     = 7
    caption      = block7_caps[class_id, clip_index]

    print(f"[Block {block_id} | Class {class_id} | Clip {clip_index} | Sample {sample_index}] Caption: {caption}")

    # enforce float32 every loop
    eeg_input = eeg_embeds[i:i+1].to(device, dtype=torch.float32)
    neg_input = negative.to(device, dtype=torch.float32)

    if woSeq2Seq:
        video = pipe(
            None,
            eeg_input,
            negative_eeg=neg_input,
            latents=None,
            video_length=6, height=288, width=512,
            num_inference_steps=100, guidance_scale=12.5
        ).videos
        save_dir = os.path.join(SAVE_ROOT, f"{pipeline_tag}_woSeq2Seq")

    elif woDANA:
        video = pipe(
            None,
            eeg_input,
            negative_eeg=neg_input,
            latents=latents[i:i+1].to(dtype=torch.float32),
            video_length=6, height=288, width=512,
            num_inference_steps=100, guidance_scale=12.5
        ).videos
        save_dir = os.path.join(SAVE_ROOT, f"{pipeline_tag}_woDANA")

    else:
        video = pipe(
            None,
            eeg_input,
            negative_eeg=neg_input,
            latents=latents_add_noise[i:i+1].to(dtype=torch.float32),
            video_length=6, height=288, width=512,
            num_inference_steps=100, guidance_scale=12.5
        ).videos
        save_dir = os.path.join(SAVE_ROOT, f"{pipeline_tag}_FullModel")

    os.makedirs(save_dir, exist_ok=True)

    # convert tensor to numpy frames
    frames = (video[0] * 255).clamp(0,255).to(torch.uint8).permute(0,2,3,1).cpu().numpy()

    # safeguard: enforce 3-channel RGB
    if frames.shape[-1] > 3:
        frames = frames[..., :3]
    elif frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)

    fps = 3
    save_path = os.path.join(
        save_dir,
        f"block{block_id}_class{class_id}_clip{clip_index}_sample{sample_index}.mp4"
    )

    writer = imageio.get_writer(save_path, fps=fps, codec="libx264")
    for f in frames:
        writer.append_data(f)
    writer.close()

    print("Saved:", save_path)
