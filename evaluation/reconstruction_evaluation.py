# ==========================================
# TRUE vs NON-MATCH EVALUATION (GIF VERSION)
# ==========================================
# Input:
#   results/       → generated 6-frame .gif videos
#   ground_truth/  → real 6-frame .gif videos (same names)
#
# Process:
#   - True-match: reconstruction vs correct GT
#   - Non-match: reconstruction vs all other GTs (average)
#   - Compute SSIM, PSNR, FID, LPIPS for both
#
# Output:
#   Per-clip and average true/non-match metrics
# ==========================================

import os, re, torch, numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance


cfg = {
    "base": "/content/EEG2Video_my_version",
    "size": (256, 256),
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
cfg["res"] = os.path.join(cfg["base"], "results")
cfg["gt"]  = os.path.join(cfg["base"], "ground_truth")


# ==========================================
# Helpers
# ==========================================
def list_gifs(p):
    return sorted([f for f in os.listdir(p) if f.endswith(".gif")])

def load_gif(path, size):
    img = Image.open(path)
    frames = []
    try:
        while True:
            frame = img.convert("RGB").resize(size)
            frames.append(np.array(frame))
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return np.array(frames, dtype=np.uint8)

def compute_ssim_psnr(gt, pred):
    s_list, p_list = [], []
    for g, p in zip(gt, pred):
        g_gray = g.mean(-1)
        p_gray = p.mean(-1)
        s_list.append(ssim(g_gray, p_gray, data_range=255))
        p_list.append(psnr(g_gray, p_gray, data_range=255))
    return np.mean(s_list), np.mean(p_list)

def compute_lpips_fid(gt, pred, lp, fid, tr):
    g = torch.stack([tr(f) for f in gt]).to(cfg["device"])
    p = torch.stack([tr(f) for f in pred]).to(cfg["device"])
    l = lp(g, p).mean().item()
    fid.update(g, real=True)
    fid.update(p, real=False)
    f = fid.compute().item()
    fid.reset()
    return l, f

def evaluate_pair(gt_path, pred_path, lp, fid, tr):
    gt = load_gif(gt_path, cfg["size"])
    pr = load_gif(pred_path, cfg["size"])
    m = min(len(gt), len(pr))
    gt, pr = gt[:m], pr[:m]
    s, p = compute_ssim_psnr(gt, pr)
    l, f = compute_lpips_fid(gt, pr, lp, fid, tr)
    return s, p, f, l


# ==========================================
# Main
# ==========================================
def main():
    lp = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(cfg["device"])
    fid = FrechetInceptionDistance(feature=64).to(cfg["device"])
    tr = transforms.Compose([transforms.ToTensor()])

    gt_files = list_gifs(cfg["gt"])
    pred_files = list_gifs(cfg["res"])

    true_vals, non_vals = [], []

    for pf in tqdm(pred_files, desc="Evaluating GIFs"):
        base = re.sub(r"(_\d+)?\.gif$", ".gif", pf)
        pred_path = os.path.join(cfg["res"], pf)
        true_path = os.path.join(cfg["gt"], base)

        if not os.path.exists(true_path):
            continue

        # True match
        true_metrics = evaluate_pair(true_path, pred_path, lp, fid, tr)
        true_vals.append(true_metrics)

        # Non-match: average over all *other* GTs
        temp = []
        for gtf in gt_files:
            if gtf == base:
                continue
            temp.append(evaluate_pair(os.path.join(cfg["gt"], gtf), pred_path, lp, fid, tr))
        non_vals.append(np.mean(np.array(temp), axis=0))

        print(f"{pf:35s}  TRUE SSIM={true_metrics[0]:.4f}  NON={non_vals[-1][0]:.4f}")

    if true_vals:
        t = np.mean(np.array(true_vals), axis=0)
        n = np.mean(np.array(non_vals), axis=0)
        print("\n==========================================")
        print("          AVERAGE METRICS")
        print("==========================================")
        print(f"True     → SSIM={t[0]:.4f}  PSNR={t[1]:.2f}  FID={t[2]:.2f}  LPIPS={t[3]:.4f}")
        print(f"Non-match→ SSIM={n[0]:.4f}  PSNR={n[1]:.2f}  FID={n[2]:.2f}  LPIPS={n[3]:.4f}")
        print("==========================================")
    else:
        print("No valid GIF pairs found.")


if __name__ == "__main__":
    main()
