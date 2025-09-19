# EEG2Video Inference (patched & simplified for Seq2Seq only)

from pipelines_original.pipeline_tuneeeg2video import TuneAVideoPipeline
from models_original.tuneavideo.unet import UNet3DConditionModel
from models_original.tuneavideo.util import save_videos_grid
from models_original.seq2seq import Seq2SeqModel
from utils.gt_labels import GT_LABEL
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

# Semantic adapter (ensures shape [B,77,768])
class SemanticAdapter(torch.nn.Module):
    def forward(self, x):
        b = x.size(0)
        # if already [B,77,768], keep it
        if x.dim() == 3 and x.shape[1:] == (77, 768):
            return x
        flat = x.view(b, -1)
        out = torch.zeros((b, 77 * 768), device=x.device, dtype=x.dtype)
        n = min(flat.size(1), 77 * 768)
        out[:, :n] = flat[:, :n]
        return out.view(b, 77, 768)

semantic_model = SemanticAdapter().to("cuda")
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Load EEG data (sub1 as test subject)
eeg_data_path = "/content/drive/MyDrive/Data/Raw/EEG/sub1.npy"
eegdata = np.load(eeg_data_path)

# Use authors’ GT ordering
GT_label = GT_LABEL
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
        semantic_model,                # adapter ensures [B,77,768]
        eeg_test[i:i+1, ...],          # raw [1,310]
        latents=latents,               # [1,4,6,36,64]
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
