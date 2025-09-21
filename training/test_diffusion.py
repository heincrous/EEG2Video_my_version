import sys
import os

# Add your repo root to Python path
repo_root = "/content/EEG2Video_my_version"  # change to your repo folder
sys.path.append(repo_root)

# Now imports will work
from pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from core_files.unet import UNet3DConditionModel
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel


# ---------------- CONFIG ----------------
CHECKPOINT_DIR = "/content/drive/MyDrive/EEG2Video_checkpoints/EEG2Video_diffusion_output"
BLIP_CAP_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_captions"
TEST_SPLIT_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/Split_4train1test/test/Video_latents"
SAVE_DIR = "/content/drive/MyDrive/EEG2Video_inference"
os.makedirs(SAVE_DIR, exist_ok=True)
VIDEO_LENGTH = 6

# ---------------- INLINE GIF SAVE ----------------
def save_videos_grid(videos, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(videos, torch.Tensor):
        videos = videos.cpu().numpy()
    if videos.ndim == 5:
        for i, video in enumerate(videos):
            frames = []
            for frame in video:
                if frame.shape[0] == 1:
                    frame = frame.squeeze(0)
                else:
                    frame = frame.transpose(1,2,0)
                frame = np.clip(frame*255,0,255).astype(np.uint8)
                frames.append(frame)
            imageio.mimsave(f"{path}_{i}.gif", frames, fps=5)
    elif videos.ndim == 4:
        frames = []
        for frame in videos:
            if frame.shape[0] == 1:
                frame = frame.squeeze(0)
            else:
                frame = frame.transpose(1,2,0)
            frame = np.clip(frame*255,0,255).astype(np.uint8)
            frames.append(frame)
        imageio.mimsave(path, frames, fps=5)

# ---------------- LOAD UNET FIRST ----------------
unet_path = os.path.join(CHECKPOINT_DIR, "unet")
unet = UNet3DConditionModel.from_pretrained_2d(unet_path, torch_dtype=torch.float16).to("cuda")

# ---------------- LOAD PIPELINE ----------------
pipeline = TuneAVideoPipeline.from_pretrained(
    CHECKPOINT_DIR,
    unet=unet,               # pass the UNet manually
    torch_dtype=torch.float16
).to("cuda")

pipeline.enable_vae_slicing()

# ---------------- FIND TEST CAPTIONS ----------------
test_blocks = sorted(os.listdir(TEST_SPLIT_DIR))
test_captions = []

for block in test_blocks:
    block_video_dir = os.path.join(TEST_SPLIT_DIR, block)
    block_cap_dir = os.path.join(BLIP_CAP_DIR, block)
    if not os.path.exists(block_cap_dir):
        continue
    video_files = sorted([f for f in os.listdir(block_video_dir) if f.endswith(".npy")])
    for vf in video_files:
        txt_file = vf.replace(".npy",".txt")
        txt_path = os.path.join(block_cap_dir, txt_file)
        if os.path.exists(txt_path):
            test_captions.append((block, vf, txt_path))

print(f"Found {len(test_captions)} test captions matching test video latents")

# ---------------- RUN INFERENCE ----------------
for block, vf, txt_path in test_captions:
    with open(txt_path,"r") as f:
        prompt_text = f.read().strip()

    with torch.no_grad():
        sample = pipeline(prompt_text, generator=None, latents=None, video_length=VIDEO_LENGTH).videos

    save_path = os.path.join(SAVE_DIR, f"{block}_{vf.replace('.npy','.gif')}")
    save_videos_grid(sample, save_path)

print(f"\nAll test GIFs saved to {SAVE_DIR}")
