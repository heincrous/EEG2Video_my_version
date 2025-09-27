# ==========================================
# Full Inference (EEG → Video via semantic predictor + diffusion)
# Subject + Feature combo specified manually
# ==========================================
import os, gc, random, imageio, torch, numpy as np
from sklearn.preprocessing import StandardScaler
from diffusers import AutoencoderKL, DDIMScheduler
from einops import rearrange

from core.unet import UNet3DConditionModel
from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from train_semantic_predictor import SemanticPredictor, FusionNet, MODEL_MAP, FEATURE_PATHS

# ==========================================
# Config (set these manually)
# ==========================================
SUBJECT       = "sub1.npy"          # e.g. "sub10.npy"
FEATURE_TYPES = ["DE"]       # e.g. ["DE"], ["segments"], ["DE","PSD"], ["segments","DE","PSD"]

CHECKPOINT_DIR     = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"
PRETRAINED_SD_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
FINETUNED_SD_PATH  = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final"
OUTPUT_DIR         = "/content/drive/MyDrive/EEG2Video_outputs/test_full_inference"
BLIP_TEXT_PATH     = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/BLIP_text.npy"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect(); torch.cuda.empty_cache()

# ==========================================
# Load ground truth captions
# ==========================================
blip_text = np.load(BLIP_TEXT_PATH, allow_pickle=True)  # shape (7,40,5)

def get_caption(idx):
    # Block 6 (the 7th block) is test block
    class_id = idx // 10          # 40 classes
    clip_id  = (idx % 10) // 2    # 5 clips, each repeated twice
    return blip_text[6, class_id, clip_id]

# ==========================================
# Load semantic checkpoint
# ==========================================
ft_tag  = "_".join(FEATURE_TYPES)
sub_tag = SUBJECT.replace(".npy", "")
ckpt_pattern = f"semantic_predictor_{ft_tag}_{sub_tag}.pt"
ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_pattern)

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"No checkpoint found: {ckpt_pattern}")

print("Using checkpoint:", ckpt_path)
ckpt_data   = torch.load(ckpt_path, map_location=device)
state_dict  = ckpt_data["state_dict"]
saved_feats = ckpt_data["feature_types"]

if set(saved_feats) != set(FEATURE_TYPES):
    raise ValueError(f"Checkpoint trained with {saved_feats}, "
                     f"but you specified {FEATURE_TYPES}")

# ==========================================
# Build semantic predictor
# ==========================================
dim_map = {"DE": 128, "PSD": 128, "segments": 256}

if len(FEATURE_TYPES) > 1:
    encoders  = {ft: MODEL_MAP[ft]() for ft in FEATURE_TYPES}
    total_dim = sum(dim_map[ft] for ft in FEATURE_TYPES)
    encoder   = FusionNet(encoders, total_dim)
    input_dim = total_dim
    multi     = True
else:
    ft        = FEATURE_TYPES[0]
    encoder   = MODEL_MAP[ft]()
    input_dim = dim_map[ft]
    multi     = False

model = SemanticPredictor(encoder, input_dim).to(device)
model.load_state_dict(state_dict, strict=False)
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
        arr = arr.reshape(-1, 62, 5)
    elif ft == "segments":
        arr = rearrange(arr, "a b c d (w t) -> (a b c w) d t", w=2, t=200)
    return arr

samples_per_block = 400
train_idx = np.arange(0, 6 * samples_per_block)
test_idx  = np.arange(6 * samples_per_block, 7 * samples_per_block)

if multi:
    feats_test = {}
    for ft in FEATURE_TYPES:
        arr = load_features(SUBJECT, ft)
        scaler = StandardScaler().fit(arr[train_idx].reshape(len(train_idx), -1))
        feats_test[ft] = scaler.transform(arr[test_idx].reshape(len(test_idx), -1)).reshape(arr[test_idx].shape)
    test_data = feats_test
else:
    ft = FEATURE_TYPES[0]
    arr = load_features(SUBJECT, ft)
    scaler = StandardScaler().fit(arr[train_idx].reshape(len(train_idx), -1))
    test_data = scaler.transform(arr[test_idx].reshape(len(test_idx), -1)).reshape(arr[test_idx].shape)

# ==========================================
# Inference helper
# ==========================================
def run_inference(eeg_feat, idx):
    if multi:
        eeg_tensor = {ft: torch.tensor(eeg_feat[ft], dtype=torch.float32).unsqueeze(0).to(device) for ft in eeg_feat}
    else:
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

    caption = get_caption(idx)
    inf_base = f"{SUBJECT.replace('.npy','')}_{ft_tag}_sample{idx}"
    inf_path = os.path.join(OUTPUT_DIR, inf_base + ".mp4")
    writer = imageio.get_writer(inf_path, fps=fps, codec="libx264")
    for f in frames: writer.append_data(f)
    writer.close()

    txt_path = os.path.join(OUTPUT_DIR, inf_base + ".txt")
    with open(txt_path, "w") as f:
        f.write(caption + "\n")

    print("Saved inference video:", inf_path)
    print("Ground truth caption:", caption)

# ==========================================
# Run 5 random samples
# ==========================================
if multi:
    num_samples = next(iter(test_data.values())).shape[0]
else:
    num_samples = test_data.shape[0]

for n in range(5):
    idx = random.randrange(num_samples)
    eeg_feat = {ft: test_data[ft][idx] for ft in test_data} if multi else test_data[idx]
    run_inference(eeg_feat, idx)
