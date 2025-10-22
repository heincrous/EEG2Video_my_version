# ==========================================
# EEG2VIDEO SEMANTIC-ONLY INFERENCE
# ==========================================
# Input: Predicted CLIP embeddings (.npy)
# Process: Generate videos using Tune-A-Video pipeline
# Output: .gif videos per class/trial
# ==========================================

import os, gc, re, shutil, torch, numpy as np
from diffusers import AutoencoderKL, DDIMScheduler, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from core.unet import UNet3DConditionModel
from EEG2Video_pipeline import TuneAVideoPipeline
from core.save_video_grid import save_videos_grid


# ==========================================
# Default Configuration
# ==========================================
"""
DEFAULT INFERENCE CONFIGURATION

Model:
Pretrained Stable Diffusion (v1-4)
Scheduler: DDIMScheduler
Guidance scale: 8
Inference steps: 100
Video length: 6 frames (3 FPS)
"""

# ==========================================
# Experiment Settings
# ==========================================
SUBSETS = {
    "subset_A": [0, 5, 10, 11, 21],
    "subset_B": [23, 26, 27, 30, 36],
    "subset_C": [1, 4, 7, 12, 13],
    "subset_D": [22, 24, 26, 28, 38],
    "subset_E": [2, 5, 6, 8, 10],
    "subset_F": [21, 23, 29, 30, 37],
    "subset_G": [0, 3, 7, 9, 11],
    "subset_H": [22, 25, 26, 27, 39],
    "subset_I": [1, 2, 4, 6, 8],
    "subset_J": [12, 21, 24, 28, 36],
    "subset_K": [3, 5, 7, 10, 11],
    "subset_L": [13, 22, 26, 30, 38],
    "subset_M": [0, 4, 6, 8, 9],
    "subset_N": [12, 21, 23, 27, 37],
    "subset_O": [1, 2, 5, 7, 10],
    "subset_P": [11, 22, 24, 28, 39],
    "subset_Q": [3, 4, 6, 12, 36],
    "subset_R": [13, 21, 25, 29, 38],
    "subset_S": [0, 2, 5, 7, 9],
    "subset_T": [10, 22, 23, 27, 30],
}

CONFIG = {
    "num_inference": 50,
    "guidance_scale": 8,
    "use_dpm_solver": True,
    "video_length": 6,
    "fps": 3,
    "test_block": 6,
    "pretrained_sd_path": "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4",
    "pred_root": "/content/drive/MyDrive/EEG2Video_results/semantic_predictor/predictions",
    "blip_text_path": "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/BLIP_text.npy",
    "output_root": "/content/drive/MyDrive/EEG2Video_results/inference",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "only_correct_predictions": True,  # Reconstruct only correctly predicted samples
}


# ==========================================
# Memory Config
# ==========================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()


# ==========================================
# Diffusion Pipeline
# ==========================================
def load_pipeline(cfg):
    scheduler_class = (
        DPMSolverMultistepScheduler if cfg["use_dpm_solver"] else DDIMScheduler
    )
    pipe = TuneAVideoPipeline(
        vae=AutoencoderKL.from_pretrained(
            cfg["pretrained_sd_path"], subfolder="vae", torch_dtype=torch.float16
        ),
        text_encoder=CLIPTextModel.from_pretrained(
            cfg["pretrained_sd_path"], subfolder="text_encoder", torch_dtype=torch.float16
        ),
        tokenizer=CLIPTokenizer.from_pretrained(
            cfg["pretrained_sd_path"], subfolder="tokenizer"
        ),
        unet=UNet3DConditionModel.from_pretrained_2d(
            cfg["pretrained_sd_path"], subfolder="unet"
        ),
        scheduler=scheduler_class.from_pretrained(
            cfg["pretrained_sd_path"], subfolder="scheduler"
        ),
    ).to(cfg["device"])

    pipe.unet.to(torch.float16)
    pipe.enable_vae_slicing()

    print(f"Scheduler: {'DPM-Solver-Multistep' if cfg['use_dpm_solver'] else 'DDIM'}")
    return pipe


# ==========================================
# Inference
# ==========================================
def run_inference_for_subset(subset_name, class_subset, cfg, pipe):
    print(f"\n=== Running inference for {subset_name}: {class_subset} ===")

    # Paths
    sem_path = os.path.join(
        cfg["pred_root"], f"{'_'.join(map(str, class_subset))}.npy"
    )
    subset_name_num = "_".join(map(str, class_subset))
    out_dir = os.path.join(
        cfg["output_root"],
        subset_name_num + ("_correct_only" if cfg.get("only_correct_predictions", False) else "")
    )
    os.makedirs(out_dir, exist_ok=True)

    # Cleanup
    cleanup_previous_outputs(out_dir)

    # Load predictions + captions
    print(f"Loading semantic predictions: {sem_path}")
    sem_preds_all = np.load(sem_path, allow_pickle=True)

    # Detect if correctness is embedded in the same array (shape [..., 769])
    if sem_preds_all.shape[-1] == 769:
        print("Detected combined predictions with correctness channel.")
        correct_flags = sem_preds_all[..., -1]  # shape: (num_classes, 5, 77)
        correct_flags = correct_flags[..., 0].reshape(-1).astype(int)
        sem_preds_all = sem_preds_all[..., :768]
    else:
        correct_flags = None
        print("No correctness channel detected — using full predictions.")

    # Filter only correct predictions if enabled
    if cfg.get("only_correct_predictions", False) and correct_flags is not None:
        print("Filtering to only correctly predicted samples...")
        flat_preds = sem_preds_all.reshape(-1, 77, 768)
        kept_indices = np.where(correct_flags == 1)[0]
        sem_preds_all = flat_preds[kept_indices]
        print(f"Filtered predictions shape: {sem_preds_all.shape} | Kept {len(kept_indices)} samples.")
    else:
        kept_indices = np.arange(sem_preds_all.reshape(-1, 77, 768).shape[0])
        print(f"Loaded prediction array shape: {sem_preds_all.shape}")

    # Load captions
    blip_text = np.load(cfg["blip_text_path"], allow_pickle=True)

    # Determine structure
    if cfg.get("only_correct_predictions", False) and sem_preds_all.ndim == 3:
        total_samples = sem_preds_all.shape[0]
    else:
        num_classes, trials_per_class = sem_preds_all.shape[:2]
        total_samples = num_classes * trials_per_class

    print(f"Predictions shape: {sem_preds_all.shape} | Total samples: {total_samples}")

    # Negative embedding
    mean_sem = sem_preds_all.reshape(-1, 77, 768).mean(axis=0)
    neg_embeddings = torch.tensor(mean_sem, dtype=torch.float16).unsqueeze(0).to(cfg["device"])
    print(f"Negative embedding shape: {tuple(neg_embeddings.shape)}")

    # Generate videos
    test_block = cfg["test_block"]

    for i, emb in enumerate(sem_preds_all):
        semantic_emb = torch.tensor(emb, dtype=torch.float16).unsqueeze(0).to(cfg["device"])

        # Map index back to class/trial
        idx = kept_indices[i]
        class_idx = idx // 5
        trial_idx = idx % 5
        class_id = class_subset[class_idx]

        caption = str(blip_text[test_block, class_id, trial_idx])
        safe_caption = re.sub(r"[^a-zA-Z0-9_-]", "_", caption)[:120]

        video = pipe(
            prompt=semantic_emb,
            negative_prompt=neg_embeddings,
            video_length=cfg["video_length"],
            height=288,
            width=512,
            num_inference_steps=cfg["num_inference"],
            guidance_scale=cfg["guidance_scale"],
        ).videos

        out_path = os.path.join(out_dir, f"class{class_id}_clip{trial_idx + 1}_{safe_caption}.gif")
        save_videos_grid(video, out_path, fps=cfg["fps"])
        print(f"Saved {out_path}")


# ==========================================
# Cleanup
# ==========================================
def cleanup_previous_outputs(folder):
    deleted = 0
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
                deleted += 1
            elif os.path.isdir(path):
                shutil.rmtree(path)
                deleted += 1
        except Exception as e:
            print(f"⚠️ Failed to delete {path}: {e}")
    print(f"🧹 Deleted {deleted} previous file(s) from {folder}.")


# ==========================================
# Main
# ==========================================
def main():
    print("=== EEG2Video Semantic-Only Inference ===")
    print(f"Device: {CONFIG['device']}")
    print(f"Guidance scale: {CONFIG['guidance_scale']}")
    pipe = load_pipeline(CONFIG)

    for subset_name, class_subset in SUBSETS.items():
        run_inference_for_subset(subset_name, class_subset, CONFIG, pipe)

    print("\nAll subset inferences completed successfully.")


if __name__ == "__main__":
    main()
