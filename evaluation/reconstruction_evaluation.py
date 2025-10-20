# ==========================================
# TRUE vs NON-MATCH VIDEO EVALUATION
# ==========================================
# Computes SSIM, PSNR, FID, LPIPS for:
#   (1) each reconstructed → correct ground truth
#   (2) reconstructed → all *other* ground truths (averaged)
# ==========================================

import os, re, cv2, torch, numpy as np
from tqdm import tqdm
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance

cfg = {
    "base": "/content/EEG2Video_my_version",
    "w": 256, "h": 256,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
cfg["results"] = os.path.join(cfg["base"], "results")
cfg["gt"] = os.path.join(cfg["base"], "ground_truth")

def list_videos(p): return sorted([f for f in os.listdir(p) if f.endswith(".mp4")])

def load_video(p, size):
    cap = cv2.VideoCapture(p); frames=[]
    while True:
        r,f = cap.read()
        if not r: break
        f = cv2.resize(f,size); f = cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
        frames.append(f)
    cap.release(); return np.array(frames,dtype=np.uint8)

def compute_ssim_psnr(gt,pred):
    s_list,p_list=[],[]
    for g,p in zip(gt,pred):
        g=cv2.cvtColor(g,cv2.COLOR_RGB2GRAY)
        p=cv2.cvtColor(p,cv2.COLOR_RGB2GRAY)
        s_list.append(ssim(g,p,data_range=255))
        p_list.append(psnr(g,p,data_range=255))
    return np.mean(s_list),np.mean(p_list)

def compute_lpips_fid(gt,pred,lp,fid,tr):
    g=torch.stack([tr(f) for f in gt]).to(cfg["device"])
    p=torch.stack([tr(f) for f in pred]).to(cfg["device"])
    l=lp(g,p).mean().item()
    fid.update(g,real=True); fid.update(p,real=False)
    f=fid.compute().item(); fid.reset()
    return l,f

def evaluate_pair(gt_path,pred_path,lp,fid,tr):
    gt=load_video(gt_path,(cfg["w"],cfg["h"]))
    pr=load_video(pred_path,(cfg["w"],cfg["h"]))
    m=min(len(gt),len(pr)); gt,pr=gt[:m],pr[:m]
    s,p=compute_ssim_psnr(gt,pr)
    l,f=compute_lpips_fid(gt,pr,lp,fid,tr)
    return s,p,f,l

def main():
    lp=LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(cfg["device"])
    fid=FrechetInceptionDistance(feature=64).to(cfg["device"])
    tr=transforms.Compose([transforms.ToTensor()])

    gt_files=list_videos(cfg["gt"]); pred_files=list_videos(cfg["results"])
    true_vals=[]; non_vals=[]

    for pf in tqdm(pred_files,desc="Evaluating"):
        base=re.sub(r"(_\d+)?\.mp4$",".mp4",pf)
        pred_path=os.path.join(cfg["results"],pf)
        true_path=os.path.join(cfg["gt"],base)
        if not os.path.exists(true_path): continue

        # True match
        true_metrics=evaluate_pair(true_path,pred_path,lp,fid,tr)
        true_vals.append(true_metrics)

        # Non-match average
        tmp=[]
        for gtf in gt_files:
            if gtf==base: continue
            tmp.append(evaluate_pair(os.path.join(cfg["gt"],gtf),pred_path,lp,fid,tr))
        non_vals.append(np.mean(np.array(tmp),axis=0))

        print(f"{pf:35s} True SSIM={true_metrics[0]:.4f}  Non-match SSIM={non_vals[-1][0]:.4f}")

    if true_vals:
        t=np.mean(np.array(true_vals),axis=0)
        n=np.mean(np.array(non_vals),axis=0)
        print("\n==========================================")
        print("          AVERAGE METRICS")
        print("==========================================")
        print(f"True   → SSIM={t[0]:.4f}  PSNR={t[1]:.2f}  FID={t[2]:.2f}  LPIPS={t[3]:.4f}")
        print(f"NonMatch→ SSIM={n[0]:.4f}  PSNR={n[1]:.2f}  FID={n[2]:.2f}  LPIPS={n[3]:.4f}")
        print("==========================================")
    else:
        print("No valid pairs found.")

if __name__=="__main__": main()
