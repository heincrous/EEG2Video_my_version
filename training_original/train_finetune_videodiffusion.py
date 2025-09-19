import argparse
import datetime
import logging
import inspect
import math
import os
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from models_original.tuneavideo.unet import UNet3DConditionModel
from models_original.tuneavideo.dataset import TuneAVideoDataset, TuneMultiVideoDataset
from pipelines_original.pipeline_tuneeeg2video import TuneAVideoPipeline
from models_original.tuneavideo.util import save_videos_grid, ddim_inversion
from einops import rearrange

# --- DANA import ---
from models_original.dana.dana import Diffusion
# -------------------

os.environ["PYTORCH_CUDA_ALLOC_conf"] = "max_split_size_mb:24"

check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


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
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None,
    num_train_epochs: int = 20,          # <— ADD THIS
    dana_beta: float = 0.3
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if seed is not None:
        set_seed(seed)

    # Handle output dir
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer, models
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")

    # Freeze VAE + text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet.requires_grad_(False)
    for name, module in unet.named_modules():
        if name.endswith(tuple(trainable_modules)):
            for params in module.parameters():
                params.requires_grad = True

    # Memory optimizations
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            logger.info("Using xformers memory efficient attention.")
        else:
            logger.warning("xformers is not available, continuing without it.")

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Optimizer
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Dataset
    train_dataset = TuneMultiVideoDataset(**train_data)
    print("First 10 video paths:", train_dataset.video_path[:10])
    print("video_path_length:", len(train_dataset.video_path))
    print("prompt_length:", len(train_dataset.prompt))

    train_dataset.prompt_ids = tokenizer(
        list(train_dataset.prompt), max_length=tokenizer.model_max_length,
        padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )

    # Validation pipeline
    validation_pipeline = TuneAVideoPipeline(
        vae=vae, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    )
    validation_pipeline.enable_vae_slicing()
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(validation_data.num_inv_steps)
    num_train_epochs = config["num_train_epochs"]

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=num_train_epochs * len(train_dataloader) * gradient_accumulation_steps,
    )

    # Accelerator prep
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # --- Init DANA ---
    dana = Diffusion(time_steps=noise_scheduler.num_train_timesteps)
    # -----------------

    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")

    global_step, first_epoch = 0, 1

    for epoch in tqdm(range(first_epoch, num_train_epochs+1)):
        unet.train()
        train_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_train_epochs}", leave=False)

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(weight_dtype)
                video_length = pixel_values.shape[1]
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215

                # --- Apply DANA noise ---
                dynamic_beta = config.get("dana_beta", dana_beta)
                latents = dana.forward(latents, dynamic_beta)
                # ------------------------

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]

                if noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

        if epoch % 10 == 0 and accelerator.is_main_process:
            samples = []
            generator = torch.Generator(device=latents.device).manual_seed(seed)

            for idx, prompt in enumerate(validation_data.prompts):
                sample = validation_pipeline(prompt, generator=generator, latents=None, **validation_data).videos
                save_videos_grid(sample, f"{output_dir}/samples/sample-{epoch}/{prompt}.gif")
                samples.append(sample)

            samples = torch.concat(samples)
            save_path = f"{output_dir}/samples/sample-{epoch}.gif"
            save_videos_grid(samples, save_path)
            logger.info(f"Saved samples to {save_path}")

            unet = accelerator.unwrap_model(unet)
            pipeline = TuneAVideoPipeline.from_pretrained(
                pretrained_model_path,
                vae=vae,
                unet=unet,
            )
            pipeline.save_pretrained(output_dir)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = TuneAVideoPipeline.from_pretrained(
            pretrained_model_path,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/all_40_video.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))
