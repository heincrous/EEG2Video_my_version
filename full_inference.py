# ==========================================
# Full Inference (configurable, EEG → Video via semantic predictor + diffusion)
# ==========================================
import os, gc, glob, random, imageio, torch, shutil, numpy as np
from sklearn.preprocessing import StandardScaler
from diffusers import AutoencoderKL, DDIMScheduler
from einops import rearrange

from core.unet import UNet3DConditionModel
from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from train_semantic_predictor import SemanticPredictor, FusionNet, MODEL_MAP, FEATURE_PATHS

# ==========================================
# Config
# ==========================================
FEATURE_TYPE         = "DE"   # "segments", "DE", "PSD", or "fusion"
SUBJECT              = "sub1.npy"
CHECKPOINT_DIR       = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_predictor"
PRETRAINED_SD_PATH   = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
FINETUNED_SD_PATH    = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final"
DATA_ROOT            = "/content/drive/MyDrive/EEG2Video_data/processed"
OUTPUT_DIR           = "/content/drive/MyDrive/EEG2Video_outputs/test_full_inference"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect(); torch.cuda.empty_cache()

# ==========================================
# Select semantic checkpoint
# ==========================================
ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, f"semantic_predictor_{FEATURE_TYPE}_sub1*.pt")))
if not ckpts:
    raise FileNotFoundError(f"No checkpoints found for {FEATURE_TYPE}")
print("Available checkpoints:")
for i, p in enumerate(ckpts):
    print(f"[{i}] {os.path.basename(p)}")
choice = int(input("Select checkpoint index: "))
semantic_ckpt = ckpts[choice]
print("Using checkpoint:", semantic_ckpt)

ckpt_data = torch.load(semantic_ckpt, map_location=device)
input_dim = ckpt_data["input_dim"]

# ==========================================
# Build semantic predictor
# ==========================================
if FEATURE_TYPE == "fusion":
    encoders = {
        "DE": MODEL_MAP["DE"](),
        "PSD": MODEL_MAP["PSD"](),
        "segments": MODEL_MAP["segments"](),
    }
    encoder = FusionNet(encoders)
else:
    encoder = MODEL_MAP[FEATURE_TYPE]()

model = SemanticPredictor(encoder, input_dim).to(device)
model.load_state_dict(ckpt_data["state_dict"])
model.eval()

# ==========================================
# Load diffusion backbone
# ==========================================
vae = AutoencoderKL.from_pretrained(PRETRAINED_SD_PATH, subfolder="vae").to(device, dtype=torch.float32)
scheduler = DDIMScheduler.from_pretrained(PRETRAINED_SD_PATH, subfolder="scheduler")
unet = UNet3DConditionModel.from_pretrained_2d(PRETRAINED_SD_PATH, subfolder="unet").to(device, dtype=torch.float32)

pipe = TuneAVideoPipeline.from_pretrained(
    FINETUNED_SD_PATH,
    vae=vae,
    unet=unet,
    scheduler=scheduler,
    torch_dtype=torch.float32,
).to(device)
pipe.enable_vae_slicing()

# ==========================================
# Load EEG features + scale
# ==========================================
def load_features(subname, ft):
    path = os.path.join(FEATURE_PATHS[ft], subname)
    arr = np.load(path)
    if ft in ["DE", "PSD"]:
        arr = arr.reshape(-1, 62, 5)   # (2800, 62, 5)
    elif ft == "segments":
        arr = rearrange(arr, "a b c d (w t) -> (a b c w) d t", w=2, t=200)  # (2800, 62, 200)
    return arr

if FEATURE_TYPE == "fusion":
    feats = {ft: load_features(SUBJECT, ft) for ft in FEATURE_PATHS}
    feat_len = next(iter(feats.values())).shape[0]
else:
    feats = load_features(SUBJECT, FEATURE_TYPE)
    feat_len = feats.shape[0]

samples_per_block = 400
train_idx = np.arange(0, 6 * samples_per_block)   # blocks 0-5
test_idx  = np.arange(6 * samples_per_block, 7 * samples_per_block)  # block 6

if FEATURE_TYPE == "fusion":
    feats_test = {}
    for ft, arr in feats.items():
        scaler = StandardScaler().fit(arr[train_idx].reshape(len(train_idx), -1))
        feats_test[ft] = scaler.transform(arr[test_idx].reshape(len(test_idx), -1)).reshape(arr[test_idx].shape)
    test_data = feats_test
else:
    scaler = StandardScaler().fit(feats[train_idx].reshape(len(train_idx), -1))
    test_data = scaler.transform(feats[test_idx].reshape(len(test_idx), -1)).reshape(feats[test_idx].shape)

# ==========================================
# Inference helper
# ==========================================
def run_inference(eeg_feat, clip_name):
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

    inf_path = os.path.join(OUTPUT_DIR, f"{clip_name}_inference.mp4")
    writer = imageio.get_writer(inf_path, fps=fps, codec="libx264")
    for f in frames: writer.append_data(f)
    writer.close()
    print("Saved inference video:", inf_path)

# ==========================================
# Run 5 random samples
# ==========================================
if FEATURE_TYPE == "fusion":
    num_samples = next(iter(test_data.values())).shape[0]
else:
    num_samples = test_data.shape[0]

for n in range(5):
    idx = random.randrange(num_samples)
    if FEATURE_TYPE == "fusion":
        eeg_feat = {ft: test_data[ft][idx] for ft in test_data}
    else:
        eeg_feat = test_data[idx]

    clip_name = f"{SUBJECT.replace('.npy','')}_block7_sample{idx}"
    run_inference(eeg_feat, clip_name)
