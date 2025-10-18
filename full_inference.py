# ==========================================
# Inference
# ==========================================
import os, gc, re, shutil, torch, numpy as np
from diffusers import AutoencoderKL, DDIMScheduler, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from core.unet import UNet3DConditionModel
from pipelines.my_pipeline import TuneAVideoPipeline
from core.util import save_videos_grid


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
    "subset_A": [2, 26, 24, 23, 38],
    "subset_B": [37, 12, 10, 6, 3],
    "subset_C": [8, 36, 13, 22, 5],
    "subset_D": [1, 25, 39, 9, 29],
    "subset_E": [28, 0, 4, 7, 11],
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
    out_dir = os.path.join(cfg["output_root"], subset_name_num)
    os.makedirs(out_dir, exist_ok=True)

    # Cleanup
    cleanup_previous_outputs(out_dir)

    # Load predictions + captions
    print(f"Loading semantic predictions: {sem_path}")
    sem_preds_all = np.load(sem_path, allow_pickle=True)
    blip_text = np.load(cfg["blip_text_path"], allow_pickle=True)

    num_classes, trials_per_class = sem_preds_all.shape[:2]
    total_samples = num_classes * trials_per_class
    print(f"Predictions shape: {sem_preds_all.shape} | Total samples: {total_samples}")

    # Negative embedding
    mean_sem = sem_preds_all.reshape(-1, 77, 768).mean(axis=0)
    neg_embeddings = torch.tensor(mean_sem, dtype=torch.float16).unsqueeze(0).to(cfg["device"])
    print(f"Negative embedding shape: {tuple(neg_embeddings.shape)}")

    # Generate videos
    flat_preds = sem_preds_all.reshape(total_samples, 77, 768)
    test_block = cfg["test_block"]
    sample_idx = 0

    for ci, class_id in enumerate(class_subset):
        print(f"\n[CLASS {class_id}] ------------------------------")
        for trial in range(trials_per_class):
            emb = flat_preds[sample_idx]
            semantic_emb = torch.tensor(emb, dtype=torch.float16)
            if semantic_emb.ndim == 2:
                semantic_emb = semantic_emb.unsqueeze(0)
            semantic_emb = semantic_emb.to(cfg["device"])

            caption = str(blip_text[test_block, class_id, trial])
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

            out_path = os.path.join(out_dir, f"class{class_id}_trial{trial}_{safe_caption}.gif")
            save_videos_grid(video, out_path, fps=cfg["fps"])
            print(f"Saved {out_path}")

            sample_idx += 1


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
