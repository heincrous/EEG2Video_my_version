import os
import torch
import numpy as np
import decord
from einops import rearrange
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as T
import clip
from PIL import Image

from models_original.seq2seq import Seq2SeqModel
from training_original.train_semantic_predictor import SemanticPredictor

# ----------------------------------------------------------------
# Load EEG features (example clip)
eeg_file = "/content/drive/MyDrive/Data/Processed/EEG_timewindows_100/sub1.npy"
eeg_features = np.load(eeg_file)  # [blocks, classes, clips, 4, 62, 100]
eeg_clip = eeg_features[0, 0, 0]  # block0, class0, clip0
eeg_input_4d = torch.from_numpy(eeg_clip).unsqueeze(0).float().cuda()  # [1,4,62,100]
eeg_flat = rearrange(eeg_input_4d, "b f c t -> b (f c t)")
input_dim = eeg_flat.shape[1]

# ----------------------------------------------------------------
# Semantic predictor
semantic_model = SemanticPredictor(input_dim=input_dim).to("cuda")
semantic_model.load_state_dict(torch.load(
    "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_predictor_full.pt",
    map_location="cuda"
)["state_dict"])
semantic_model.eval()

# ----------------------------------------------------------------
# Seq2Seq checkpoints
CKPT_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_sub1to10_blocks1to4.pt"
seq2seq_trained = Seq2SeqModel().to("cuda")
seq2seq_trained.load_state_dict(torch.load(CKPT_PATH, map_location="cuda"))
seq2seq_trained.eval()

seq2seq_random = Seq2SeqModel().to("cuda")
seq2seq_random.eval()

# ----------------------------------------------------------------
# Ground truth frames (decord)
gt_video = "/content/drive/MyDrive/Data/Raw/Video/1st_10min.mp4"
vr = decord.VideoReader(gt_video)
clip_idx = 0
frame_start = clip_idx * 48   # 2s at 24fps
frame_end = frame_start + 48
gt_frames = [Image.fromarray(frame.asnumpy()) for frame in vr.get_batch(range(frame_start, frame_end))]

transform = T.Compose([
    T.Resize((288,512)),
    T.ToTensor()
])
gt_tensors = torch.stack([transform(f) for f in gt_frames])  # [48,3,288,512]

# Downsample to match our window count (e.g., 6 frames)
gt_tensors = gt_tensors[::8]  # every 8th frame -> 6 frames

# ----------------------------------------------------------------
# Load CLIP
clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")

def evaluate(seq2seq_model, name):
    with torch.no_grad():
        B = eeg_input_4d.size(0)
        F = eeg_input_4d.size(1)
        dummy_tgt = torch.zeros((B, F, 4, 36, 64), device="cuda")
        pred_latents = seq2seq_model(eeg_input_4d, dummy_tgt)

        # Debug
        print(f"[{name}] latents mean={pred_latents.mean().item():.4f}, std={pred_latents.std().item():.4f}")

        # Fake recon as RGB by reshaping latents (for metrics only)
        fake_imgs = pred_latents.view(F, 3, 96, 96)  # coarse reshaping
        fake_imgs = (fake_imgs - fake_imgs.min()) / (fake_imgs.max()-fake_imgs.min()+1e-6)
        fake_imgs = torch.nn.functional.interpolate(fake_imgs, size=(288,512), mode="bilinear")

        # SSIM (mean over frames)
        ssim_vals = []
        for i in range(min(len(fake_imgs), len(gt_tensors))):
            s = ssim(gt_tensors[i].permute(1,2,0).cpu().numpy(),
                     fake_imgs[i].permute(1,2,0).cpu().numpy(),
                     channel_axis=2, data_range=1.0)
            ssim_vals.append(s)
        mean_ssim = np.mean(ssim_vals)

        # CLIP similarity
        gt_feats = clip_model.encode_image(gt_tensors.cuda()).mean(0, keepdim=True)
        fake_feats = clip_model.encode_image((fake_imgs*255).clamp(0,255).byte()).mean(0, keepdim=True)
        sim = torch.nn.functional.cosine_similarity(gt_feats, fake_feats).item()

        print(f"{name}: SSIM={mean_ssim:.4f}, CLIP-sim={sim:.4f}")
        return mean_ssim, sim

# ----------------------------------------------------------------
print("\nEvaluating trained vs random...")
trained_ssim, trained_sim = evaluate(seq2seq_trained, "Trained")
random_ssim, random_sim = evaluate(seq2seq_random, "Random")

