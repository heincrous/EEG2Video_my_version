# ==========================================
# Full Inference (Subject1, DE features, Block7 test set, 5 samples, scaler refit + captions)
# ==========================================
import os, gc, glob, random, imageio, torch, shutil, numpy as np
from sklearn.preprocessing import StandardScaler
from diffusers import AutoencoderKL, DDIMScheduler

from core_files.unet import UNet3DConditionModel
from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from training.train_semantic_final import ReshapeWrapper, SemanticPredictor

# --- Paths ---
drive_root            = "/content/drive/MyDrive/EEG2Video_data/processed"
pretrained_model_path = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
finetuned_model_path  = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final"
semantic_ckpt_dir     = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"

blip_text_root        = os.path.join(drive_root, "BLIP_text")
video_mp4_root        = os.path.join(drive_root, "Video_mp4")
output_dir            = "/content/drive/MyDrive/EEG2Video_outputs/test_full_inference"
os.makedirs(output_dir, exist_ok=True)

# --- Device + mem cleanup ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()
device = "cuda"

# === Select semantic predictor checkpoint ===
ckpts = sorted(glob.glob(os.path.join(semantic_ckpt_dir, "semantic_predictor_sub1_de*.pt")))
if not ckpts:
    raise FileNotFoundError("No DE checkpoints for sub1 found")
print("Available checkpoints:")
for i, p in enumerate(ckpts):
    print(f"[{i}] {os.path.basename(p)}")
choice = int(input("Select checkpoint index: "))
semantic_ckpt = ckpts[choice]
print("Using checkpoint:", semantic_ckpt)

# === Load diffusion backbone ===
vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").to(device, dtype=torch.float32)
scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet").to(device, dtype=torch.float32)

pipe = TuneAVideoPipeline.from_pretrained(
    finetuned_model_path,
    vae=vae,
    unet=unet,
    scheduler=scheduler,
    torch_dtype=torch.float32,
).to(device)
pipe.enable_vae_slicing()

# === Build semantic predictor ===
in_dim = 62*5
out_shape = (77,768)
model = ReshapeWrapper(SemanticPredictor(in_dim=in_dim, out_shape=out_shape), n_tokens=77).to(device)
model.load_state_dict(torch.load(semantic_ckpt, map_location=device)["state_dict"])
model.eval()

# === Load DE features and rebuild scaler ===
subj = "sub1"
de_path = os.path.join(drive_root, "EEG_DE", f"{subj}.npy")   # shape (7,40,5,62,5)
de = np.load(de_path)

# Fit scaler on blocks 1–6 (indices 0–5)
trainval = de[:6].reshape(-1, 62*5)
scaler = StandardScaler().fit(trainval)

# Prepare block 7 (index 6) as test set
block7 = de[6].reshape(-1, 62*5)
samples = scaler.transform(block7)

# === Helper: run inference for one sample ===
def run_inference(eeg_feat, clip_name, block="Block7", class_idx=None, clip_idx=None):
    eeg_tensor = torch.tensor(eeg_feat, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        semantic_pred = model(eeg_tensor)
    negative = semantic_pred.mean(dim=0, keepdim=True).float().to(device)

    video_length, fps = 6, 3
    video = pipe(
        eeg=semantic_pred,
        negative_eeg=negative,
        latents=None,
        video_length=video_length,
        height=288,
        width=512,
        num_inference_steps=100,
        guidance_scale=12.5,
    ).videos

    frames = (video[0] * 255).clamp(0,255).to(torch.uint8).permute(0,2,3,1).cpu().numpy()
    frames = frames[:video_length]
    if frames.shape[-1] > 3: frames = frames[...,:3]
    elif frames.shape[-1] == 1: frames = np.repeat(frames, 3, axis=-1)

    inf_path = os.path.join(output_dir, f"{clip_name}_inference.mp4")
    writer = imageio.get_writer(inf_path, fps=fps, codec="libx264")
    for f in frames: writer.append_data(f)
    writer.close()

    # Copy ground-truth video and print caption if exists
    if class_idx is not None and clip_idx is not None:
        fname = f"class{class_idx:02d}_clip{clip_idx:02d}"
        gt_path = os.path.join(video_mp4_root, block, fname + ".mp4")
        if os.path.exists(gt_path):
            shutil.copy(gt_path, os.path.join(output_dir, f"{clip_name}_groundtruth.mp4"))

        cap_path = os.path.join(blip_text_root, block, fname + ".txt")
        if os.path.exists(cap_path):
            with open(cap_path,"r") as f:
                caption = f.read().strip()
            print("BLIP caption:", caption)

    print("Saved inference video:", inf_path)

# === Pick 5 random samples ===
for n in range(5):
    idx = random.randrange(len(samples))
    eeg_feat = samples[idx]

    # map idx back to class and clip
    class_idx = idx // 5
    clip_idx  = idx % 5 + 1
    clip_name = f"{subj}_block7_class{class_idx:02d}_clip{clip_idx:02d}"
    run_inference(eeg_feat, clip_name, block="Block7", class_idx=class_idx, clip_idx=clip_idx)
