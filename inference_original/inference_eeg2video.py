# EEG2Video Inference (Seq2Seq only, faithful to paper setup)

import os
import torch
import numpy as np

from pipelines_original.pipeline_tuneeeg2video import TuneAVideoPipeline
from models_original.tuneavideo.unet import UNet3DConditionModel
from models_original.tuneavideo.util import save_videos_grid
from models_original.seq2seq import Seq2SeqModel

# ----------------------------------------------------------------
# Load Seq2Seq checkpoint
CKPT_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_subset.pt"
seq2seq = Seq2SeqModel().to("cuda")
seq2seq.load_state_dict(torch.load(CKPT_PATH, map_location="cuda"))
seq2seq.eval()
print("Loaded Seq2Seq checkpoint:", CKPT_PATH)

# Dummy semantic model (placeholder for real semantic predictor)
class DummySemantic(torch.nn.Module):
    def forward(self, x):
        b = x.size(0)
        expanded = torch.zeros((b, 77 * 768), device=x.device)
        expanded[:, :min(x.shape[1], 77 * 768)] = x[:, :min(x.shape[1], 77 * 768)]
        return expanded.view(b, 77, 768)

semantic_model = DummySemantic().to("cuda")
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Load EEG features (DE: [7,40,5,62,5])
eeg_file = "/content/drive/MyDrive/Data/Processed/EEG_DE_1per2s/sub1.npy"
eeg_features = np.load(eeg_file)

# Example clip: block01, class00, clip0
eeg_clip = eeg_features[0, 0, 0]  # shape (5,62,5)
eeg_input = torch.from_numpy(eeg_clip).unsqueeze(0).float().cuda()  # [1,5,62,5]
print(">>> eeg_input shape:", eeg_input.shape)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Build UNet + pipeline
print(">>> Starting UNet initialization")
unet = UNet3DConditionModel(
    sample_size=64,
    in_channels=4,
    out_channels=4,
    layers_per_block=2,
    block_out_channels=(320, 640, 1280),
    down_block_types=("DownBlock3D", "DownBlock3D", "DownBlock3D"),
    up_block_types=("UpBlock3D", "UpBlock3D", "UpBlock3D"),
    cross_attention_dim=768
).to("cuda").to(torch.float32)

pipe = TuneAVideoPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    unet=unet,
    torch_dtype=torch.float32
).to("cuda")
pipe.enable_vae_slicing()
print(">>> Pipeline ready")
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Generate latents from Seq2Seq
with torch.no_grad():
    B, F = 1, 5  # EEG clips have F=5 slices
    dummy_tgt = torch.zeros((B, F, 4, 36, 64), device="cuda")

    pred_latents = seq2seq(eeg_input, dummy_tgt)  # [B,F,9216]
    latents = pred_latents.view(B, F, 4, 36, 64).permute(0, 2, 1, 3, 4).contiguous()

print(">>> latents final shape:", latents.shape)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Inference loop (just 1 example for now)
print(">>> Starting inference loop")
os.makedirs("Results", exist_ok=True)

with torch.no_grad():
    eeg_embed = semantic_model(eeg_input.view(1, -1))  # flatten to [1,D] -> [1,77,768]

    video = pipe(
        None,
        eeg_embed,
        latents=latents,
        video_length=F,
        height=288,
        width=512,
        num_inference_steps=100,
        guidance_scale=12.5
    ).videos

    save_path = "./Results/sample.gif"
    save_videos_grid(video, save_path)
    print(f">>> Saved {save_path}")
# ----------------------------------------------------------------
