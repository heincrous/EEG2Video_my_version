"""
EEG2Video Inference (One EEG → One MP4, with modes)
---------------------------------------------------
Modes:
  woSeq2Seq : Only semantic predictor, no Seq2Seq latents
  woDANA    : Semantic predictor + Seq2Seq latents (no DANA)
  full      : Semantic predictor + Seq2Seq latents + DANA
"""

import os, sys, gc, imageio, torch
import numpy as np
from einops import rearrange

# === Repo imports ===
repo_root = "/content/EEG2Video_my_version"
sys.path.append(os.path.join(repo_root, "pipelines"))
from pipeline_tuneeeg2video import TuneAVideoPipeline

sys.path.append(os.path.join(repo_root, "core_files"))
from unet import UNet3DConditionModel

# === Paths ===
drive_root            = "/content/drive/MyDrive/EEG2Video_data/processed"
pretrained_model_path = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
finetuned_model_path  = "/content/drive/MyDrive/EEG2Video_outputs"
semantic_ckpt         = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_predictor.pt"
seq2seq_ckpt          = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_checkpoint.pt"
dana_latents_path     = "/content/drive/MyDrive/EEG2Video_checkpoints/dana_latents.npy"

eeg_feat_list_path    = os.path.join(drive_root, "EEG_features/test_list.txt")
eeg_win_list_path     = os.path.join(drive_root, "EEG_windows/test_list.txt")

output_dir            = "/content/drive/MyDrive/EEG2Video_outputs/EEG_inference"
os.makedirs(output_dir, exist_ok=True)

# === Memory config ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()
device = "cuda"

# ==========================================
# Load pipeline (fine-tuned UNet + SD backbone)
# ==========================================
unet = UNet3DConditionModel.from_pretrained(
    finetuned_model_path, subfolder="unet", torch_dtype=torch.float16
).to(device)

pipe = TuneAVideoPipeline.from_pretrained(
    pretrained_model_path,
    unet=unet,
    torch_dtype=torch.float16,
).to(device)

pipe.enable_vae_slicing()

# ==========================================
# Load Semantic Predictor
# ==========================================
from training.train_semantic import SemanticPredictor
semantic_model = SemanticPredictor().to(device)
semantic_model.load_state_dict(torch.load(semantic_ckpt)["state_dict"])
semantic_model.eval()

# ==========================================
# Load Seq2Seq Transformer
# ==========================================
from training.train_seq2seq import MyTransformer
seq2seq_model = MyTransformer(d_model=512, pred_frames=24).to(device)
seq2seq_model.load_state_dict(torch.load(seq2seq_ckpt)["state_dict"])
seq2seq_model.eval()

# ==========================================
# Pick one EEG sample (features + windows)
# ==========================================
with open(eeg_feat_list_path, "r") as f:
    eeg_feat_files = [line.strip() for line in f]
with open(eeg_win_list_path, "r") as f:
    eeg_win_files = [line.strip() for line in f]

# first sample
eeg_feat_file = os.path.join(drive_root, "EEG_features", eeg_feat_files[0])
eeg_win_file  = os.path.join(drive_root, "EEG_windows",  eeg_win_files[0])

print("Chosen EEG feature file:", eeg_feat_file)
print("Chosen EEG window file:", eeg_win_file)

# EEG feature for semantic predictor
eeg_feat = np.load(eeg_feat_file).reshape(-1)   # shape [310]
eeg_feat_tensor = torch.tensor(eeg_feat, dtype=torch.float32).unsqueeze(0).to(device)

# EEG window for seq2seq
eeg_win = np.load(eeg_win_file)  # [7,62,100]
eeg_win_tensor = torch.tensor(eeg_win, dtype=torch.float32).unsqueeze(0).to(device)

# ==========================================
# Semantic Predictor → embedding [77,768]
# ==========================================
with torch.no_grad():
    semantic_pred = semantic_model(eeg_feat_tensor)
semantic_pred = semantic_pred.view(1, 77, 768).half().to(device)
print("Semantic embedding shape:", semantic_pred.shape)

# Negative EEG embedding = mean vector
negative = semantic_pred.mean(dim=0, keepdim=True)

# ==========================================
# Seq2Seq → latents [1,24,4,36,64]
# ==========================================
zero_frame = torch.zeros((1,1,4,36,64), device=device)
with torch.no_grad():
    seq2seq_latents = seq2seq_model.generate(eeg_win_tensor, zero_frame)
seq2seq_latents = seq2seq_latents.half().to(device)
print("Seq2Seq latents shape:", seq2seq_latents.shape)

# ==========================================
# DANA latents (precomputed, aligned to same sample)
# ==========================================
if os.path.exists(dana_latents_path):
    dana_latents = np.load(dana_latents_path)
    dana_latents = torch.from_numpy(dana_latents).unsqueeze(0).half().to(device)
    print("Loaded DANA latents:", dana_latents.shape)
else:
    dana_latents = None
    print("Warning: No DANA latents found, full mode unavailable")

# ==========================================
# Modes
# ==========================================
def run_inference(mode="full"):
    if mode == "woSeq2Seq":
        latents = None
    elif mode == "woDANA":
        latents = seq2seq_latents
    elif mode == "full":
        if dana_latents is None:
            raise RuntimeError("DANA latents not available")
        latents = dana_latents
    else:
        raise ValueError("Mode must be one of: woSeq2Seq, woDANA, full")

    video = pipe(
        model=None,
        eeg=semantic_pred,         
        negative_eeg=negative,     
        latents=latents,           
        video_length=24,
        height=288,
        width=512,
        num_inference_steps=100,   # updated to 100
        guidance_scale=12.5,
    ).videos

    print(f"[{mode}] Generated video tensor:", video.shape)

    # Save MP4
    frames = (video[0] * 255).clamp(0, 255).to(torch.uint8)
    frames = frames.permute(0, 2, 3, 1).cpu().numpy()
    if frames.shape[-1] > 3:
        frames = frames[..., :3]
    elif frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)

    mp4_path = os.path.join(output_dir, f"sample_eeg2video_{mode}.mp4")
    writer = imageio.get_writer(mp4_path, fps=12, codec="libx264")
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"[{mode}] Video saved to:", mp4_path)


# === Run all three modes (one EEG only) ===
run_inference("woSeq2Seq")
run_inference("woDANA")
if dana_latents is not None:
    run_inference("full")
