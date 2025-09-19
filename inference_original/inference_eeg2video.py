# EEG2Video Inference (patched & simplified for Seq2Seq only)

from pipelines_original.pipeline_tuneeeg2video import TuneAVideoPipeline
from models_original.tuneavideo.unet import UNet3DConditionModel
from models_original.tuneavideo.util import save_videos_grid
from models_original.seq2seq import Seq2SeqModel
# Future: from EEG2Video.EEG2Video_New.Semantic.eeg_text import CLIP or CLIPSmall

import torch
import numpy as np

# ----------------------------------------------------------------
# Load Seq2Seq checkpoint
CKPT_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_dummy.pt"
seq2seq = Seq2SeqModel().to("cuda")
seq2seq.load_state_dict(torch.load(CKPT_PATH, map_location="cuda"))
seq2seq.eval()
print("Loaded Seq2Seq checkpoint:", CKPT_PATH)

# Dummy semantic model (EEG -> [B,77,768])
class DummySemantic(torch.nn.Module):
    def forward(self, x):
        b = x.size(0)
        expanded = torch.zeros((b, 77*768), device=x.device)
        expanded[:, :min(x.shape[1], 77*768)] = x[:, :min(x.shape[1], 77*768)]
        return expanded.view(b, 77, 768)

semantic_model = DummySemantic().to("cuda")
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Load EEG data (sub1 as test subject)
eeg_data_path = "/content/drive/MyDrive/Data/Raw/EEG/sub1.npy"
eegdata = np.load(eeg_data_path)

# Labels (authors’ GT ordering, unchanged)
GT_label = np.array([
    [23, 22, 9, 6, 18, 14, 5, 36, 25, 19, 28, 35, 3, 16, 24, 40, 15, 27, 38, 33,
     34, 4, 39, 17, 1, 26, 20, 29, 13, 32, 37, 2, 11, 12, 30, 31, 8, 21, 7, 10],
    [27, 33, 22, 28, 31, 12, 38, 4, 18, 17, 35, 39, 40, 5, 24, 32, 15, 13, 2, 16,
     34, 25, 19, 30, 23, 3, 8, 29, 7, 20, 11, 14, 37, 6, 21, 1, 10, 36, 26, 9],
    [15, 36, 31, 1, 34, 3, 37, 12, 4, 5, 21, 24, 14, 16, 39, 20, 28, 29, 18, 32,
     2, 27, 8, 19, 13, 10, 30, 40, 17, 26, 11, 9, 33, 25, 35, 7, 38, 22, 23, 6],
    [16, 28, 23, 1, 39, 10, 35, 14, 19, 27, 37, 31, 5, 18, 11, 25, 29, 13, 20, 24,
     7, 34, 26, 4, 40, 12, 8, 22, 21, 30, 17, 2, 38, 9, 3, 36, 33, 6, 32, 15],
    [18, 29, 7, 35, 22, 19, 12, 36, 8, 15, 28, 1, 34, 23, 20, 13, 37, 9, 16, 30,
     2, 33, 27, 21, 14, 38, 10, 17, 31, 3, 24, 39, 11, 32, 4, 25, 40, 5, 26, 6],
    [29, 16, 1, 22, 34, 39, 24, 10, 8, 35, 27, 31, 23, 17, 2, 15, 25, 40, 3, 36,
     26, 6, 14, 37, 9, 12, 19, 30, 5, 28, 32, 4, 13, 18, 21, 20, 7, 11, 33, 38],
    [38, 34, 40, 10, 28, 7, 1, 37, 22, 9, 16, 5, 12, 36, 20, 30, 6, 15, 35, 2,
     31, 26, 18, 24, 8, 3, 23, 19, 14, 13, 21, 4, 25, 11, 32, 17, 39, 29, 33, 27]
])

# Test EEG [clips,timepoints] -> shrink to manageable dim
test_indices = [list(GT_label[6]).index(el) for el in range(1, 41)]
eeg_test = torch.from_numpy(eegdata[6][test_indices, :])  # [40, T]
eeg_test = eeg_test.view(eeg_test.shape[0], -1)[:, :310].float().cuda()
print(">>> eeg_test shape:", eeg_test.shape)
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
    B, F = 1, 6
    dummy_tgt = torch.zeros((B, F, 4, 36, 64), device="cuda")

    eeg_input = torch.zeros((B, F, 62, 100), device="cuda")
    flat = eeg_test[0:1, :min(62*100, eeg_test.shape[1])]
    eeg_input.view(B, -1)[:, :flat.numel()] = flat.view(-1)

    pred_latents = seq2seq(eeg_input, dummy_tgt)  # [B,F,9216]
    latents = pred_latents.view(B, F, 4, 36, 64).permute(0, 2, 1, 3, 4).contiguous()

print(">>> latents final shape:", latents.shape)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Inference loop (2 samples for testing)
print(">>> Starting inference loop")
for i in range(2):
    video = pipe(
        semantic_model,                # stub semantic encoder
        eeg_test[i:i+1, ...],
        latents=latents,
        video_length=6,
        height=288,
        width=512,
        num_inference_steps=100,
        guidance_scale=12.5
    ).videos

    savename = "40_Classes_Test"
    save_videos_grid(video, f"./{savename}/{i}.gif")
    print(f">>> Saved {savename}/{i}.gif")
# ----------------------------------------------------------------
