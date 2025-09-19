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
# Load EEG features (subject1, block0, class0, all 5 clips)
eeg_file = "/content/drive/MyDrive/Data/Processed/EEG_timewindows_100/sub1.npy"
eeg_features = np.load(eeg_file)  # [blocks, classes, clips, 4, 62, 100]
block_id, class_id = 0, 0
eeg_clips = eeg_features[block_id, class_id]  # shape [5, 4, 62, 100]

# ----------------------------------------------------------------
# Prepare semantic predictor
eeg_flat = rearrange(torch.from_numpy(eeg_clips[0]).unsqueeze(0).float(), "b f c t -> b (f c t)")
input_dim = eeg_flat.shape[1]

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
# Ground truth video (block0)
gt_video = "/content/drive/MyDrive/Data/Raw/Video/1st_10min.mp4"
vr = decord.VideoReader(gt_video)

transform = T.Compose([
    T.Resize((288,512)),
    T.ToTensor()
])

# ----------------------------------------------------------------
# Load CLIP
clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")

def evaluate(seq2seq_model, name, eeg_clip, clip_idx):
    with torch.no_grad():
        B, F, C, Tt = eeg_clip.shape  # (1,F,62,100)
        dummy_tgt = torch.zeros((B, F, 4, 36, 64), device="cuda")
        pred_latents = seq2seq_model(eeg_clip, dummy_tgt)

        print(f"[{name}] clip {clip_idx} -> shape={tuple(pred_latents.shape)}, "
              f"mean={pred_latents.mean().item():.4f}, std={pred_latents.std().item():.4f}")

        # Flatten + fake RGB
        flat = pred_latents.view(F, -1)
        needed = 3 * 96 * 96
        if flat.shape[1] < needed:
            pad = needed - flat.shape[1]
            flat = torch.nn.functional.pad(flat, (0, pad))
        flat = flat[:, :needed]
        fake_imgs = flat.view(F, 3, 96, 96)

        fake_imgs = (fake_imgs - fake_imgs.min()) / (fake_imgs.max() - fake_imgs.min() + 1e-6)
        fake_imgs = torch.nn.functional.interpolate(fake_imgs, size=(288,512), mode="bilinear")

        # Ground truth frames (48 frames = 2s at 24fps)
        frame_start = clip_idx * 48
        frame_end = frame_start + 48
        batch = vr.get_batch(range(frame_start, frame_end)).asnumpy()
        gt_frames = [Image.fromarray(f) for f in batch]
        gt_tensors = torch.stack([transform(f) for f in gt_frames])[::8]  # downsample to 6 frames

        # SSIM
        ssim_vals = []
        for i in range(min(len(fake_imgs), len(gt_tensors))):
            s = ssim(gt_tensors[i].permute(1,2,0).cpu().numpy(),
                     fake_imgs[i].permute(1,2,0).cpu().numpy(),
                     channel_axis=2, data_range=1.0)
            ssim_vals.append(s)
        mean_ssim = np.mean(ssim_vals)

        # CLIP similarity
        gt_pre = torch.stack([clip_preprocess(T.ToPILImage()(f.cpu())) for f in gt_tensors]).cuda()
        fake_pre = torch.stack([clip_preprocess(T.ToPILImage()(f.cpu())) for f in fake_imgs]).cuda()
        gt_feats = clip_model.encode_image(gt_pre).mean(0, keepdim=True)
        fake_feats = clip_model.encode_image(fake_pre).mean(0, keepdim=True)
        sim = torch.nn.functional.cosine_similarity(gt_feats, fake_feats).item()

        print(f"{name} clip {clip_idx}: SSIM={mean_ssim:.4f}, CLIP-sim={sim:.4f}")
        return mean_ssim, sim

# ----------------------------------------------------------------
print("\nEvaluating 5 clips for block0,class0...")

trained_scores = []
random_scores = []

for i in range(5):
    eeg_clip = torch.from_numpy(eeg_clips[i]).unsqueeze(0).float().cuda()  # (1,F,62,100)
    trained_scores.append(evaluate(seq2seq_trained, "Trained", eeg_clip, i))
    random_scores.append(evaluate(seq2seq_random, "Random", eeg_clip, i))

trained_ssim = np.mean([s for s,_ in trained_scores])
trained_sim = np.mean([c for _,c in trained_scores])
random_ssim = np.mean([s for s,_ in random_scores])
random_sim = np.mean([c for _,c in random_scores])

print("\n=== Average Results (block0,class0,5 clips) ===")
print(f"Trained: SSIM={trained_ssim:.4f}, CLIP-sim={trained_sim:.4f}")
print(f"Random : SSIM={random_ssim:.4f}, CLIP-sim={random_sim:.4f}")
