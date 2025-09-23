import logging
import inspect
import math
import os
import sys
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
from einops import rearrange

repo_root = "/content/EEG2Video_my_version"
sys.path.append(os.path.join(repo_root, "pipelines"))
from pipeline_tuneavideo import TuneAVideoPipeline

sys.path.append(os.path.join(repo_root, "core_files"))
from unet import UNet3DConditionModel
from dataset import TuneMultiVideoDataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ------------------------------
# Helper: Batched VAE encoding
# ------------------------------
def encode_frames_in_batches(vae, pixel_values, video_length, batch_size=4):
    """
    Encode frames through VAE in batches to save VRAM.
    Args:
        vae: AutoencoderKL
        pixel_values: [B*F, 3, H, W] tensor
        video_length: number of frames per video
        batch_size: number of frames per VAE forward
    Returns:
        latents: [B, C, F, H', W']
    """
    all_latents = []
    for i in range(0, pixel_values.shape[0], batch_size):
        batch = pixel_values[i:i+batch_size]
        with torch.no_grad():
            latent = vae.encode(batch).latent_dist.sample()
            latent = latent * 0.18215
        all_latents.append(latent)
    latents = torch.cat(all_latents, dim=0)
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    return latents


def main(
    pretrained_model_path: str,
    output_dir: str,
    train_data: Dict,
    validation_data: Dict,
    validation_steps: int = 100,
    trainable_modules: Tuple[str] = (
        "attn1.to_q",
        "attn2.to_q",
        "attn_temp",
    ),
    train_batch_size: int = 10,
    max_train_steps: int = 1200000,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = False,
    seed: Optional[int] = None,
    num_train_epochs: int = 200,
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

    if seed is not None:
        set_seed(seed)

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    # ---- Load pretrained components ----
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")

    # Freeze non-trainable parts
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    for name, module in unet.named_modules():
        if name.endswith(tuple(trainable_modules)):
            for params in module.parameters():
                params.requires_grad = True

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if accelerator.is_main_process:
        print("Trainable modules:")
        for name, module in unet.named_modules():
            if any(name.endswith(tm) for tm in trainable_modules):
                print(" -", name)

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # ---- Dataset ----
    train_dataset = TuneMultiVideoDataset(**train_data)

    video_latents_root = "/content/drive/MyDrive/EEG2Video_data/processed/Video_latents"
    blip_text_root     = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text"

    latent_list_path = os.path.join(video_latents_root, "train_list.txt")
    text_list_path   = os.path.join(blip_text_root, "train_list.txt")

    with open(latent_list_path, "r") as f:
        train_dataset.video_path = [os.path.join(video_latents_root, line.strip()) for line in f]

    with open(text_list_path, "r") as f:
        train_dataset.prompt = [os.path.join(blip_text_root, line.strip()) for line in f]

    print("video_path_length:", len(train_dataset.video_path))
    print("prompt_length:", len(train_dataset.prompt))

    train_dataset.prompt_ids = tokenizer(
        [open(p).read().strip() for p in train_dataset.prompt],
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        persistent_workers=True,
    )

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=num_train_epochs * len(train_dataloader) * gradient_accumulation_steps,
    )

    # prepare with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Sanity check
    first_batch = next(iter(train_dataloader))
    if "latents" in first_batch:
        latents = first_batch["latents"]
        inferred_video_length = latents.shape[2]
    elif "pixel_values" in first_batch:
        pixel_values = first_batch["pixel_values"]
        inferred_video_length = pixel_values.shape[1]
    else:
        raise ValueError("Dataset did not return 'latents' or 'pixel_values'")

    validation_data["video_length"] = min(24, inferred_video_length)
    print(f"[Validation] Using video_length={validation_data['video_length']}")

    # val_text_list_path = os.path.join(blip_text_root, "test_list.txt")
    # with open(val_text_list_path, "r") as f:
    #     all_prompts = [os.path.join(blip_text_root, line.strip()) for line in f]

    # config["validation_data"]["prompts"] = [
    #     open(random.choice(all_prompts)).read().strip()
    # ]

    # validation_pipeline = TuneAVideoPipeline(
    #     vae=vae,
    #     text_encoder=text_encoder,
    #     tokenizer=tokenizer,
    #     unet=unet,
    #     scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler"),
    # )
    # validation_pipeline.enable_vae_slicing()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    global_step = 0
    first_epoch = 1

    for epoch in range(first_epoch, num_train_epochs + 1):
        unet.train()
        total_loss = 0.0

        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}/{num_train_epochs}", leave=True) as pbar:
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    if "latents" in batch:
                        latents = batch["latents"].to(weight_dtype)
                    else:
                        pixel_values = batch["pixel_values"].to(weight_dtype)
                        video_length = pixel_values.shape[1]
                        pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                        latents = encode_frames_in_batches(vae, pixel_values, video_length, batch_size=4)

                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0, noise_scheduler.num_train_timesteps, (latents.size(0),), device=latents.device
                    ).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]
                    target = noise if noise_scheduler.prediction_type == "epsilon" else noise_scheduler.get_velocity(latents, noise, timesteps)

                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    accelerator.backward(loss)

                    # ✅ Only clip when gradients are being synced
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item()
                global_step += 1
                pbar.update(1)

        avg_loss = total_loss / len(train_dataloader)
        current_lr = lr_scheduler.get_last_lr()[0]
        print(f"[Epoch {epoch}] Avg loss: {avg_loss:.6f} | LR: {current_lr:.2e}")

        # if epoch % 2 == 0 and accelerator.is_main_process:
        #     generator = torch.Generator(device=latents.device).manual_seed(seed)
        #     for prompt in config["validation_data"]["prompts"]:
        #         _ = validation_pipeline(
        #             prompt,
        #             video_length=validation_data["video_length"],
        #             num_inference_steps=20,
        #             generator=generator
        #         )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = TuneAVideoPipeline.from_pretrained(
            pretrained_model_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    config = dict(
        pretrained_model_path="/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4",
        output_dir="/content/drive/MyDrive/EEG2Video_outputs",
        train_data=dict(video_path=None, prompt=None),
        validation_data=dict(prompts=None, num_inv_steps=20, use_inv_latent=False),
        train_batch_size=1,
        learning_rate=3e-5,
        num_train_epochs=1,
        mixed_precision="fp16",
        gradient_accumulation_steps=4,
        enable_xformers_memory_efficient_attention=False,
        seed=42,
    )
    main(**config)
