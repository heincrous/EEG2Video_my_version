# ==========================================
# Diffusion Training
# ==========================================
import os
from typing import Optional
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

from pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from core.unet import UNet3DConditionModel


# ==========================================
# Config
# ==========================================
train_batch_size       = 2
num_epochs             = 10
learning_rate          = 3e-5
gradient_accumulation  = 1
gradient_checkpointing = True
mixed_precision        = "fp16"   # "fp16", "bf16", or None
max_grad_norm          = 1.0
seed                   = 42

# restrict to certain classes (0–39); set to None for all
CLASS_SUBSET           = [1, 10, 12, 16, 19, 23, 25, 31, 34, 39]

# paths
PRETRAINED_MODEL_PATH  = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
DATA_ROOT              = "/content/drive/MyDrive/EEG2Video_data/processed"
SAVE_ROOT              = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints"

os.makedirs(SAVE_ROOT, exist_ok=True)


# ==========================================
# Dataset
# ==========================================
class LatentsTextDataset(Dataset):
    def __init__(self, latents_path, text_path, tokenizer, class_subset=None, split="train"):
        latents = np.load(latents_path, allow_pickle=True)  # (7,40,5,6,4,36,64)
        texts   = np.load(text_path, allow_pickle=True)     # (7,40,5)

        # reshape
        latents = latents.reshape(7, 40, 5, 6, 4, 36, 64)
        texts   = texts.reshape(7, 40, 5)

        # split by block
        if split == "train":
            latents = latents[:6].reshape(-1, 6, 4, 36, 64)  # blocks 0–5
            texts   = texts[:6].reshape(-1)
            block_count = 6
        elif split == "test":
            latents = latents[6:].reshape(-1, 6, 4, 36, 64)  # block 6 only
            texts   = texts[6:].reshape(-1)
            block_count = 1
        else:
            raise ValueError(f"Unknown split: {split}")

        # build class labels
        labels_block = np.repeat(np.arange(40), 5*2)      # 400 per block
        labels_all   = np.tile(labels_block, block_count) # 2400 train or 400 test

        # apply class subset filter
        if class_subset is not None:
            mask = np.isin(labels_all, class_subset)
            latents, texts, labels_all = latents[mask], texts[mask], labels_all[mask]

        self.latents   = latents
        self.texts     = texts
        self.labels    = labels_all
        self.tokenizer = tokenizer

        print(f"Dataset prepared: {self.latents.shape[0]} clips "
              f"(split={split}, subset={class_subset})")

    def __len__(self): 
        return len(self.latents)

    def __getitem__(self, idx):
        latents = self.latents[idx]
        text    = str(self.texts[idx])
        prompt_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        return {"latents": torch.from_numpy(latents).float(),
                "prompt_ids": prompt_ids}


# ==========================================
# Main training
# ==========================================
def main():
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation,
        mixed_precision=mixed_precision,
    )
    if seed is not None:
        set_seed(seed)

    # load pretrained models
    noise_scheduler = DDPMScheduler.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="scheduler")
    tokenizer       = CLIPTokenizer.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="tokenizer")
    text_encoder    = CLIPTextModel.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="text_encoder")
    vae             = AutoencoderKL.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="vae")
    unet            = UNet3DConditionModel.from_pretrained_2d(PRETRAINED_MODEL_PATH, subfolder="unet")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(True)
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

    # datasets
    latents_path = os.path.join(DATA_ROOT, "Video_latents/Video_latents.npy")
    text_path    = os.path.join(DATA_ROOT, "BLIP_text/BLIP_text.npy")
    train_dataset = LatentsTextDataset(latents_path, text_path, tokenizer,
                                       class_subset=CLASS_SUBSET, split="train")
    test_dataset  = LatentsTextDataset(latents_path, text_path, tokenizer,
                                       class_subset=CLASS_SUBSET, split="test")
    print(f"Loaded {len(train_dataset)} train and {len(test_dataset)} test samples")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_epochs * len(train_dataloader),
    )

    # accelerator prep
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # training loop
    for epoch in range(1, num_epochs + 1):
        unet.train()
        total_loss = 0.0
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
            for batch in train_dataloader:
                with accelerator.accumulate(unet):
                    latents = batch["latents"].to(accelerator.device, dtype=weight_dtype)
                    latents = rearrange(latents, "b f c h w -> b c f h w")
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0, noise_scheduler.num_train_timesteps, 
                        (latents.size(0),), device=latents.device
                    ).long()
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

    # save final pipeline
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = TuneAVideoPipeline.from_pretrained(
            PRETRAINED_MODEL_PATH,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
        )
        subset_tag = ""
        if CLASS_SUBSET is not None:
            subset_tag = "_subset" + "-".join(str(c) for c in CLASS_SUBSET)
        save_dir = os.path.join(SAVE_ROOT, f"pipeline_final{subset_tag}")
        pipeline.save_pretrained(save_dir)
        print(f"Saved final pipeline to {save_dir}")

    accelerator.end_training()


# ==========================================
# Entrypoint
# ==========================================
if __name__ == "__main__":
    main()
