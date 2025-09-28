# ==========================================
# Diffusion Training (Final Minimal Version)
# ==========================================

# === Standard libraries ===
import os
from typing import Optional

# === Third-party libraries ===
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from einops import rearrange

from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed

# === Repo imports ===
from pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from core.unet import UNet3DConditionModel


# ==========================================
# Paths
# ==========================================
pretrained_model_path = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
data_root             = "/content/drive/MyDrive/EEG2Video_data/processed"
save_root             = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints"

os.makedirs(save_root, exist_ok=True)


# ==========================================
# Dataset (Video_latents.npy + BLIP_text.npy)
# ==========================================
class LatentsTextDataset(Dataset):
    def __init__(self, latents_path, text_path, tokenizer):
        self.latents = np.load(latents_path, allow_pickle=True)  # (N,F,C,H,W)
        self.texts   = np.load(text_path, allow_pickle=True)     # (N,)
        assert len(self.latents) == len(self.texts), "Mismatch between latents and text length"
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        latents = self.latents[idx]  # (F,C,H,W)
        text    = str(self.texts[idx])
        prompt_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        return {
            "latents": torch.from_numpy(latents).float(),
            "prompt_ids": prompt_ids,
        }


# ==========================================
# Main training
# ==========================================
def main(
    pretrained_model_path: str,
    train_batch_size: int = 2,
    learning_rate: float = 3e-5,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    mixed_precision: Optional[str] = "fp16",
    max_grad_norm: float = 1.0,
    num_epochs: int = 1,
    seed: Optional[int] = None,
):
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    if seed is not None:
        set_seed(seed)

    # === Load pretrained models ===
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer       = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder    = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae             = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet            = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(True)

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

    # === Dataset ===
    latents_path = os.path.join(data_root, "Video_latents/Video_latents.npy")
    text_path    = os.path.join(data_root, "BLIP_text/BLIP_text.npy")
    dataset = LatentsTextDataset(latents_path, text_path, tokenizer)
    print(f"Loaded {len(dataset)} samples")

    train_dataloader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_epochs * len(train_dataloader),
    )

    # Prepare with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # === Training loop ===
    for epoch in range(1, num_epochs + 1):
        unet.train()
        total_loss = 0.0
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
            for batch in train_dataloader:
                with accelerator.accumulate(unet):
                    latents = batch["latents"].to(accelerator.device, dtype=weight_dtype)
                    latents = rearrange(latents, "b f c h w -> b c f h w")
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (latents.size(0),), device=latents.device).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    encoder_hidden_states = text_encoder(batch["prompt_ids"].to(accelerator.device))[0]
                    target = noise
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item()
                pbar.update(1)

        avg_loss = total_loss / len(train_dataloader)
        print(f"[Epoch {epoch}] Avg loss: {avg_loss:.6f}")

    accelerator.wait_for_everyone()

    # === Save final pipeline only ===
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = TuneAVideoPipeline.from_pretrained(
            pretrained_model_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(os.path.join(save_root, "pipeline_final"))
        print(f"Saved final pipeline to {os.path.join(save_root, 'pipeline_final')}")

    accelerator.end_training()


# ==========================================
# Entrypoint
# ==========================================
if __name__ == "__main__":
    main(pretrained_model_path=pretrained_model_path, num_epochs=20)
