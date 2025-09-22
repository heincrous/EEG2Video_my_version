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
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from einops import rearrange

repo_root = "/content/EEG2Video_my_version"
sys.path.append(os.path.join(repo_root, "pipelines"))

from pipeline_tuneavideo import TuneAVideoPipeline

sys.path.append(os.path.join(repo_root, "core_files"))

from unet import UNet3DConditionModel
from dataset import TuneMultiVideoDataset
from util import save_videos_grid, ddim_inversion

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"

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
    num_train_epochs: int = 200,
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

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

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)

    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    for name, module in unet.named_modules():
        if name.endswith(tuple(trainable_modules)):
            for params in module.parameters():
                params.requires_grad = True

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

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

    # dataset
    train_dataset = TuneMultiVideoDataset(**train_data)

    latent_list_path = "/content/drive/MyDrive/EEG2Video_data/processed/Video_latents/train_list.txt"
    text_list_path   = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/train_list.txt"

    with open(latent_list_path, "r") as f:
        train_dataset.video_path = [line.strip() for line in f]

    with open(text_list_path, "r") as f:
        train_dataset.prompt = [line.strip() for line in f]

    print("video_path_length:", len(train_dataset.video_path))
    print("prompt_length:", len(train_dataset.prompt))

    train_dataset.prompt_ids = tokenizer(
        list(train_dataset.prompt),
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )

    # === SHAPE SANITY CHECK ===
    first_batch = next(iter(train_dataloader))
    if "latents" in first_batch:
        latents = first_batch["latents"]
        inferred_video_length = latents.shape[2]
        print("Sanity check: latents shape =", latents.shape)
    elif "pixel_values" in first_batch:
        pixel_values = first_batch["pixel_values"]
        inferred_video_length = pixel_values.shape[1]
        print("Sanity check: pixel_values shape =", pixel_values.shape)
    else:
        raise ValueError("Dataset did not return 'latents' or 'pixel_values'")

    validation_data["video_length"] = inferred_video_length
    print(f"[Validation] Using video_length={inferred_video_length}")

    # restrict validation prompts
    val_text_list_path = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/test_list.txt"
    with open(val_text_list_path, "r") as f:
        all_prompts = [line.strip() for line in f]
    config["validation_data"]["prompts"] = random.sample(all_prompts, k=min(3, len(all_prompts)))

    validation_pipeline = TuneAVideoPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler"),
    )
    validation_pipeline.enable_vae_slicing()

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=num_train_epochs * len(train_dataloader) * gradient_accumulation_steps,
    )

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

    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")

    global_step = 0
    first_epoch = 1

    for epoch in tqdm(range(first_epoch, num_train_epochs + 1)):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                if "latents" in batch:
                    latents = batch["latents"].to(weight_dtype)
                else:
                    pixel_values = batch["pixel_values"].to(weight_dtype)
                    video_length = pixel_values.shape[1]
                    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                    latents = latents * 0.18215

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
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1

        # validate every 2 epochs only
        if epoch % 2 == 0 and accelerator.is_main_process:
            generator = torch.Generator(device=latents.device).manual_seed(seed)
            for prompt in validation_data["prompts"]:
                sample = validation_pipeline(
                    prompt, video_length=validation_data["video_length"], generator=generator
                ).videos
                save_videos_grid(sample, f"{output_dir}/samples/sample-{epoch}/{prompt}.gif")

    accelerator.end_training()

if __name__ == "__main__":
    config = dict(
        pretrained_model_path="/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4",
        output_dir="/content/drive/MyDrive/EEG2Video_outputs",
        train_data=dict(video_path=None, prompt=None),
        validation_data=dict(prompts=None, num_inv_steps=50, use_inv_latent=False),
        train_batch_size=1,
        learning_rate=3e-5,
        num_train_epochs=2,  # demo run
        mixed_precision="fp16",
        gradient_accumulation_steps=4,
        enable_xformers_memory_efficient_attention=False,
        seed=42,
    )
    main(**config)
