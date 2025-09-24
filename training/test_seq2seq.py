# ==========================================
# Seq2Seq Inference + Evaluation
# ==========================================

import os
import random
import numpy as np
import torch
import joblib
import imageio
from diffusers.models import AutoencoderKL
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchmetrics

# Import config + model (must match training definition)
from train_seq2seq import CONFIG, myTransformer


if __name__ == "__main__":
    # === Paths ===
    eeg_root    = os.path.join(CONFIG["bundle_dir"], "..", "EEG_windows")
    latent_root = os.path.join(CONFIG["bundle_dir"], "..", "Video_latents")
    test_list   = os.path.join(eeg_root, "test_list.txt")
    ckpt_dir    = os.path.join(CONFIG["save_root"], "seq2seq_checkpoints")
    out_dir     = os.path.join(CONFIG["save_root"], "test_seq2seq")
    vae_dir     = CONFIG.get("vae_dir", None)
    os.makedirs(out_dir, exist_ok=True)

    # === Select checkpoint ===
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    print("\nAvailable checkpoints:")
    for idx, ck in enumerate(ckpts):
        print(f"  [{idx}] {ck}")
    choice = int(input("\nEnter checkpoint index: "))
    ckpt_path = os.path.join(ckpt_dir, ckpts[choice])
    ckname = ckpts[choice]
    tag = ckname.replace("seq2seqmodel_", "").replace(".pt", "")

    scaler_path = os.path.join(ckpt_dir, f"scaler_{tag}.pkl")
    scaler = joblib.load(scaler_path)

    # === Pick a random test sample from test_list.txt ===
    with open(test_list, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    rel_path = random.choice(lines)

    eeg_path = os.path.join(eeg_root, rel_path)
    rel_parts = rel_path.split(os.sep)
    vid_rel = os.path.join(*rel_parts[1:])  # drop subject folder
    vid_path = os.path.join(latent_root, vid_rel)

    print(f"\nTesting sample: {rel_path}")
    eeg = np.load(eeg_path)        # (7,62,100)
    gt_latents = np.load(vid_path) # (F,4,36,64)

    # === Apply scaler ===
    eeg_flat = scaler.transform(eeg.reshape(-1, CONFIG["C"] * CONFIG["T"]))
    eeg = eeg_flat.reshape(eeg.shape)
    eeg = torch.from_numpy(eeg).unsqueeze(0).float().cuda()

    # === Load model ===
    model = myTransformer(CONFIG).cuda()
    state = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()

    # === Predict latents ===
    num_frames = gt_latents.shape[0]
    pred_latents = model(eeg, num_frames).squeeze(0).cpu().numpy()

    if vae_dir is None:
        raise RuntimeError("vae_dir must be set in CONFIG for decoding")
    vae = AutoencoderKL.from_pretrained(vae_dir, subfolder="vae").cuda()

    # === Decode prediction ===
    latents_t = torch.from_numpy(pred_latents).float().cuda() / 0.18215
    with torch.no_grad():
        frames = vae.decode(latents_t).sample
    frames = (frames.clamp(-1, 1) + 1) / 2
    frames = frames.permute(0, 2, 3, 1).cpu().numpy() * 255
    frames = frames.astype(np.uint8)

    # === Decode ground truth ===
    gt_latents_t = torch.from_numpy(gt_latents).float().cuda() / 0.18215
    with torch.no_grad():
        gt_frames = vae.decode(gt_latents_t).sample
    gt_frames = (gt_frames.clamp(-1, 1) + 1) / 2
    gt_frames = gt_frames.permute(0, 2, 3, 1).cpu().numpy() * 255
    gt_frames = gt_frames.astype(np.uint8)

    # === Metrics ===
    ssim_scores, psnr_scores = [], []
    for f in range(min(len(frames), len(gt_frames))):
        ssim_scores.append(ssim(frames[f], gt_frames[f], channel_axis=2, data_range=255))
        psnr_scores.append(psnr(frames[f], gt_frames[f], data_range=255))
    print(f"Mean SSIM: {np.mean(ssim_scores):.4f}")
    print(f"Mean PSNR: {np.mean(psnr_scores):.2f} dB")

    lpips_metric = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type="alex").cuda()
    gen_t = torch.from_numpy(frames).permute(0, 3, 1, 2).float().cuda() / 255.0
    gt_t  = torch.from_numpy(gt_frames).permute(0, 3, 1, 2).float().cuda() / 255.0
    with torch.no_grad():
        lpips_score = lpips_metric(gen_t, gt_t)
    print(f"LPIPS: {lpips_score.item():.4f}")

    # === Save videos ===
    base_name = os.path.splitext(os.path.basename(rel_path))[0]
    pred_path = os.path.join(out_dir, f"{base_name}_pred.mp4")
    gt_path   = os.path.join(out_dir, f"{base_name}_gt.mp4")
    imageio.mimsave(pred_path, frames, fps=3)
    imageio.mimsave(gt_path, gt_frames, fps=3)
    print(f"Saved predicted video to {pred_path}")
    print(f"Saved ground truth video to {gt_path}")
