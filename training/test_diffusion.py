import torch
import os
from pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from save_videos_grid import save_videos_grid  # use your inline function or import

# ---------------- CONFIG ----------------
OUTPUT_DIR = "./EEG2Video_checkpoints/EEG2Video_diffusion_output"  # trained pipeline
BLIP_CAP_DIR = "./EEG2Video_data/processed/BLIP_captions"         # captions
TEST_SPLIT_DIR = "./EEG2Video_data/processed/Split_4train1test/test/Video_latents"  # test latents
SAVE_DIR = "./EEG2Video_inference"  # local repo folder for GIFs

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- LOAD PIPELINE ----------------
pipeline = TuneAVideoPipeline.from_pretrained(OUTPUT_DIR)
pipeline.enable_vae_slicing()
pipeline.to("cuda")

# ---------------- CROSS-CHECK TEST CAPTIONS ----------------
test_blocks = sorted(os.listdir(TEST_SPLIT_DIR))
test_captions = []

for block in test_blocks:
    block_video_dir = os.path.join(TEST_SPLIT_DIR, block)
    block_cap_dir = os.path.join(BLIP_CAP_DIR, block)
    if not os.path.exists(block_cap_dir):
        continue
    video_files = sorted([f for f in os.listdir(block_video_dir) if f.endswith(".npy")])
    for vf in video_files:
        txt_file = vf.replace(".npy", ".txt")
        txt_path = os.path.join(block_cap_dir, txt_file)
        if os.path.exists(txt_path):
            test_captions.append((block, vf, txt_path))

print(f"Found {len(test_captions)} test captions matching video latents")

# ---------------- INFERENCE ----------------
for block, vf, txt_path in test_captions:
    with open(txt_path, "r") as f:
        prompt_text = f.read().strip()

    with torch.no_grad():
        sample = pipeline(prompt_text, generator=None, latents=None, video_length=6).videos

    save_path = os.path.join(SAVE_DIR, f"{block}_{vf.replace('.npy','.gif')}")
    save_videos_grid(sample, save_path)

print(f"\nAll test GIFs saved to {SAVE_DIR}")
