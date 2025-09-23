import os, sys, gc, glob, imageio, torch
import numpy as np
import joblib
from einops import rearrange

# === Repo imports ===
from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from core_files.unet import UNet3DConditionModel

# === Import DANA (add_noise) ===
repo_root = "/content/EEG2Video_my_version"
sys.path.append(os.path.join(repo_root, "core_files"))
from add_noise import Diffusion

# === Paths ===
drive_root            = "/content/drive/MyDrive/EEG2Video_data/processed"
pretrained_model_path = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
finetuned_model_path  = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final"
semantic_ckpt_dir     = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_predictor"
seq2seq_ckpt_dir      = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_checkpoints"

eeg_feat_list_path    = os.path.join(drive_root, "EEG_DE/test_list.txt")
eeg_win_list_path     = os.path.join(drive_root, "EEG_windows/test_list.txt")

output_dir            = "/content/drive/MyDrive/EEG2Video_outputs/test_full_inference"
os.makedirs(output_dir, exist_ok=True)

# === Memory config ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()
device = "cuda"

# ==========================================
# Helper: checkpoint selection
# ==========================================
def select_ckpt(ckpt_dir, name="checkpoint"):
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    print(f"\nAvailable {name} checkpoints:")
    for idx, path in enumerate(ckpts):
        print(f"  [{idx}] {os.path.basename(path)}")
    choice = int(input(f"Select {name} checkpoint index: "))
    return ckpts[choice]

semantic_ckpt = select_ckpt(semantic_ckpt_dir, "semantic")
seq2seq_ckpt  = select_ckpt(seq2seq_ckpt_dir, "seq2seq")
print("Using:", semantic_ckpt, seq2seq_ckpt)

# ==========================================
# Load scalers (paired with checkpoints)
# ==========================================
semantic_tag = os.path.basename(semantic_ckpt).replace("semantic_predictor_", "").replace(".pt", "")
semantic_scaler = joblib.load(os.path.join(semantic_ckpt_dir, f"scaler_{semantic_tag}.pkl"))

seq2seq_tag = os.path.basename(seq2seq_ckpt).replace("seq2seqmodel_", "").replace(".pt", "")
seq2seq_scaler = joblib.load(os.path.join(seq2seq_ckpt_dir, f"scaler_{seq2seq_tag}.pkl"))

# ==========================================
# Load finetuned pipeline (needs UNet + scheduler too)
# ==========================================
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTokenizer

vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").to(device, dtype=torch.float16)
scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet").to(device, dtype=torch.float16)

pipe = TuneAVideoPipeline.from_pretrained(
    finetuned_model_path,
    vae=vae,
    unet=unet,
    tokenizer=tokenizer,
    scheduler=scheduler,
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
from training.train_seq2seq import myTransformer
seq2seq_model = myTransformer(d_model=512, pred_frames=6).to(device)
seq2seq_model.load_state_dict(torch.load(seq2seq_ckpt)["state_dict"])
seq2seq_model.eval()

# ==========================================
# Init DANA diffusion
# ==========================================
dana_diffusion = Diffusion(time_steps=500)

# ==========================================
# Pick one EEG sample (features + windows)
# ==========================================
with open(eeg_feat_list_path, "r") as f:
    eeg_feat_files = [line.strip() for line in f]
with open(eeg_win_list_path, "r") as f:
    eeg_win_files = [line.strip() for line in f]

# first sample
eeg_feat_file = os.path.join(drive_root, "EEG_DE", eeg_feat_files[0])
eeg_win_file  = os.path.join(drive_root, "EEG_windows",  eeg_win_files[0])

print("Chosen EEG feature file:", eeg_feat_file)
print("Chosen EEG window file:", eeg_win_file)

# === EEG feature for semantic predictor (apply scaler) ===
eeg_feat = np.load(eeg_feat_file)   # (62,5)
if eeg_feat.ndim == 2 and eeg_feat.shape == (62, 5):
    eeg_feat = eeg_feat.reshape(-1)   # (310,)
eeg_feat_scaled = semantic_scaler.transform([eeg_feat])[0]
eeg_feat_tensor = torch.tensor(eeg_feat_scaled, dtype=torch.float32).unsqueeze(0).to(device)

# === EEG window for seq2seq (apply scaler) ===
eeg_win = np.load(eeg_win_file)  # [7,62,100]
eeg_win_flat = eeg_win.reshape(-1, 62 * 100)
eeg_win_scaled = seq2seq_scaler.transform(eeg_win_flat).reshape(eeg_win.shape)
eeg_win_tensor = torch.tensor(eeg_win_scaled, dtype=torch.float32).unsqueeze(0).to(device)

# ==========================================
# Semantic Predictor → embedding [1,77,768]
# ==========================================
with torch.no_grad():
    semantic_pred = semantic_model(eeg_feat_tensor)
semantic_pred = semantic_pred.view(1, 77, 768).half().to(device)
print("Semantic embedding shape:", semantic_pred.shape)

# Negative EEG embedding = mean vector
negative = semantic_pred.mean(dim=0, keepdim=True)

# ==========================================
# Seq2Seq → latents [1,6,4,36,64]
# ==========================================
zero_frame = torch.zeros((1,1,4,36,64), device=device)
with torch.no_grad():
    seq2seq_latents = seq2seq_model.generate(eeg_win_tensor, zero_frame)
seq2seq_latents = seq2seq_latents.half().to(device)
print("Seq2Seq latents shape:", seq2seq_latents.shape)

# ==========================================
# DANA noise injection at inference
# ==========================================
def make_dana_latents(seq2seq_latents, dynamic_beta=0.25):
    with torch.no_grad():
        dana_latents = dana_diffusion.forward(seq2seq_latents, dynamic_beta)
    return dana_latents.half().to(device)

# ==========================================
# Run inference for chosen mode
# ==========================================
def run_inference(mode="full"):
    if mode == "woSeq2Seq":
        latents = None
    elif mode == "woDANA":
        latents = seq2seq_latents
    elif mode == "full":
        latents = make_dana_latents(seq2seq_latents, dynamic_beta=0.25)
    else:
        raise ValueError("Mode must be one of: woSeq2Seq, woDANA, full")

    video = pipe(
        eeg=semantic_pred,
        negative_eeg=negative,
        latents=latents,
        video_length=6,
        height=288,
        width=512,
        num_inference_steps=100,
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
    writer = imageio.get_writer(mp4_path, fps=3, codec="libx264")
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"[{mode}] Video saved to:", mp4_path)

# ==========================================
# User prompt for mode
# ==========================================
print("\nSelect inference mode:")
print("  [0] full (semantic + seq2seq + DANA)")
print("  [1] woSeq2Seq (semantic only)")
print("  [2] woDANA (semantic + seq2seq, no DANA)")
choice = int(input("Enter choice index: "))

modes = ["full", "woSeq2Seq", "woDANA"]
selected_mode = modes[choice]
print(f"\nRunning mode: {selected_mode}\n")

run_inference(selected_mode)
