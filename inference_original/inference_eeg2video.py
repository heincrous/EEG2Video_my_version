# EEG2Video Inference + Evaluation (trained vs random)

import os
import torch
import numpy as np
from PIL import Image
import decord
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
import clip

from pipelines_original.pipeline_tuneeeg2video import TuneAVideoPipeline
from models_original.tuneavideo.unet import UNet3DConditionModel
from models_original.tuneavideo.util import save_videos_grid
from models_original.seq2seq import Seq2SeqModel

# ----------------------------------------------------------------
# Dummy semantic model
class DummySemantic(torch.nn.Module):
    def forward(self, x):
        b = x.size(0)
        expanded = torch.zeros((b, 77 * 768), device=x.device)
        expanded[:, :min(x.shape[1], 77 * 768)] = x[:, :min(x.shape[1], 77 * 768)]
        return expanded.view(b, 77, 768)

semantic_model = DummySemantic().to("cuda")
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Load EEG features
eeg_file = "/content/drive/MyDrive/Data/Processed/EEG_timewindows_100/sub1.npy"
eeg_features = np.load(eeg_file)
eeg_clip = eeg_features[0, 0, 0]  # block0, class0, clip0
eeg_input = torch.from_numpy(eeg_clip).unsqueeze(0).float().cuda()  # [1,4,62,100]
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Build UNet + pipeline
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
# ----------------------------------------------------------------

def run_inference(seq2seq_model, save_name):
    with torch.no_grad():
        B, F = eeg_input.size(0), eeg_input.size(1)
        dummy_tgt = torch.zeros((B, F, 4, 36, 64), device="cuda")

        pred_latents = seq2seq_model(eeg_input, dummy_tgt)  # [B,F,9216]
        latents = pred_latents.view(B, F, 4, 36, 64).permute(0, 2, 1, 3, 4).contiguous()

        eeg_embed = semantic_model(eeg_input.view(1, -1))

        video = pipe(
            None,
            eeg_embed,
            latents=latents,
            video_length=F,
            height=288,
            width=512,
            num_inference_steps=50,
            guidance_scale=12.5
        ).videos

        os.makedirs("Results", exist_ok=True)
        save_path = f"./Results/{save_name}.gif"
        save_videos_grid(video, save_path)
        return save_path

# ----------------------------------------------------------------
# Run both models
CKPT_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_subset.pt"
seq2seq_trained = Seq2SeqModel().to("cuda")
seq2seq_trained.load_state_dict(torch.load(CKPT_PATH, map_location="cuda"))
seq2seq_trained.eval()
trained_path = run_inference(seq2seq_trained, "sample_trained")

seq2seq_random = Seq2SeqModel().to("cuda")
seq2seq_random.eval()
random_path = run_inference(seq2seq_random, "sample_random")

# ----------------------------------------------------------------
# Evaluation
# Load ground truth video (block01/class00/clip0) from SEED-DV
gt_video_path = "/content/drive/MyDrive/Data/Raw/Video/1st_10min.mp4"
vr = decord.VideoReader(gt_video_path)
gt_frames = [Image.fromarray(frame.asnumpy()) for frame in vr.get_batch(range(4))]

# Helper to load GIF frames
def load_gif_frames(path):
    gif = Image.open(path)
    frames = []
    try:
        while True:
            frames.append(gif.copy().convert("RGB"))
            gif.seek(len(frames))  # next frame
    except EOFError:
        pass
    return frames

trained_frames = load_gif_frames(trained_path)
random_frames = load_gif_frames(random_path)

# Resize everything
transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
gt_frames = [transform(f) for f in gt_frames]
trained_frames = [transform(f) for f in trained_frames[:len(gt_frames)]]
random_frames = [transform(f) for f in random_frames[:len(gt_frames)]]

# SSIM
def avg_ssim(pred, gt):
    vals = []
    for p, g in zip(pred, gt):
        p_np = p.permute(1,2,0).numpy()
        g_np = g.permute(1,2,0).numpy()
        vals.append(ssim(p_np, g_np, channel_axis=-1, data_range=1.0))
    return np.mean(vals)

print("SSIM (trained vs GT):", avg_ssim(trained_frames, gt_frames))
print("SSIM (random vs GT):", avg_ssim(random_frames, gt_frames))

# CLIP similarity
clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")
def avg_clip_sim(pred, gt):
    vals = []
    for p,g in zip(pred,gt):
        p_emb = clip_model.encode_image((p.unsqueeze(0)*255).byte().cuda())
        g_emb = clip_model.encode_image((g.unsqueeze(0)*255).byte().cuda())
        vals.append(torch.cosine_similarity(p_emb, g_emb).item())
    return np.mean(vals)

print("CLIP sim (trained vs GT):", avg_clip_sim(trained_frames, gt_frames))
print("CLIP sim (random vs GT):", avg_clip_sim(random_frames, gt_frames))
