# ==========================================
# TUNE-A-VIDEO PIPELINE (EEG-CONDITIONED)
# ==========================================
# Input:
#   CLIP text embeddings or text prompts
#   Pretrained Stable Diffusion VAE, UNet3DConditionModel, and Scheduler
#
# Process:
#   - Encode text prompts or EEG-derived embeddings using CLIP
#   - Run diffusion-based latent denoising using 3D UNet
#   - Decode latents to reconstruct video frames
#
# Output:
#   TuneAVideoPipelineOutput.videos
#       Type: torch.Tensor or np.ndarray
#       Shape: [B, C, F, H, W]
# ==========================================

import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange

from core.unet import UNet3DConditionModel


# ==========================================
# LOGGER
# ==========================================
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# ==========================================
# OUTPUT DATACLASS
# ==========================================
@dataclass
class TuneAVideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


# ==========================================
# MAIN PIPELINE CLASS
# ==========================================
class TuneAVideoPipeline(DiffusionPipeline):
    _optional_components = []

    # ==========================================
    # INITIALIZATION
    # ==========================================
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        # Handle scheduler config updates
        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}."
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set `clip_sample`."
                " It should be set to False for correct results."
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        # Handle UNet config updates
        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "UNet config sample_size < 64 detected; update to 64 for consistency."
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    # ==========================================
    # MEMORY MANAGEMENT
    # ==========================================
    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")
        device = torch.device(f"cuda:{gpu_id}")
        for m in [self.unet, self.text_encoder, self.vae]:
            if m is not None:
                cpu_offload(m, device)

    # ==========================================
    # ENCODING PROMPTS
    # ==========================================
    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for m in self.unet.modules():
            if hasattr(m, "_hf_hook") and hasattr(m._hf_hook, "execution_device") and m._hf_hook.execution_device:
                return torch.device(m._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        if isinstance(prompt, torch.Tensor):
            text_embeddings = prompt.to(device)
            bs, seq_len, dim = text_embeddings.shape
            text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
            text_embeddings = text_embeddings.view(bs * num_videos_per_prompt, seq_len, dim)

            if do_classifier_free_guidance:
                if not isinstance(negative_prompt, torch.Tensor):
                    raise ValueError("When passing embeddings, negative_prompt must also be a tensor.")
                uncond = negative_prompt.to(device).repeat(1, num_videos_per_prompt, 1)
                uncond = uncond.view(bs * num_videos_per_prompt, seq_len, dim)
                text_embeddings = torch.cat([uncond, text_embeddings])
            return text_embeddings

        batch_size = len(prompt) if isinstance(prompt, list) else 1
        text_inputs = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        text_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask.to(device) if getattr(
            self.text_encoder.config, "use_attention_mask", False
        ) else None

        text_embeddings = self.text_encoder(text_ids.to(device), attention_mask=attention_mask)[0]
        bs, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size if negative_prompt is None else negative_prompt
            uncond_inputs = self.tokenizer(
                uncond_tokens, padding="max_length", max_length=text_ids.shape[-1],
                truncation=True, return_tensors="pt"
            )
            mask = uncond_inputs.attention_mask.to(device) if getattr(
                self.text_encoder.config, "use_attention_mask", False
            ) else None
            uncond_emb = self.text_encoder(uncond_inputs.input_ids.to(device), attention_mask=mask)[0]
            uncond_emb = uncond_emb.repeat(1, num_videos_per_prompt, 1)
            uncond_emb = uncond_emb.view(batch_size * num_videos_per_prompt, seq_len, -1)
            text_embeddings = torch.cat([uncond_emb, text_embeddings])
        return text_embeddings

    # ==========================================
    # LATENT DECODING
    # ==========================================
    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = self.vae.decode(latents).sample
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        return video.cpu().float().numpy()

    # ==========================================
    # SAMPLING UTILITIES
    # ==========================================
    def prepare_extra_step_kwargs(self, generator, eta):
        extra = {}
        if "eta" in inspect.signature(self.scheduler.step).parameters:
            extra["eta"] = eta
        if "generator" in inspect.signature(self.scheduler.step).parameters:
            extra["generator"] = generator
        return extra

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device
            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [torch.randn(shape, generator=g, device=rand_device, dtype=dtype) for g in generator]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)
        return latents * self.scheduler.init_noise_sigma

    # ==========================================
    # FORWARD PASS
    # ==========================================
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width  = width  or self.unet.config.sample_size * self.vae_scale_factor
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        do_cfg = guidance_scale > 1.0

        text_embeddings = self._encode_prompt(prompt, device, num_videos_per_prompt, do_cfg, negative_prompt)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt, num_channels_latents,
            video_length, height, width, text_embeddings.dtype, device, generator, latents
        )
        extra_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        num_warmup = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as pbar:
            for i, t in enumerate(timesteps):
                latent_in = torch.cat([latents] * 2) if do_cfg else latents
                latent_in = self.scheduler.scale_model_input(latent_in, t)

                noise_pred = self.unet(latent_in, t, encoder_hidden_states=text_embeddings).sample
                if do_cfg:
                    noise_uncond, noise_text = noise_pred.chunk(2)
                    noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_kwargs).prev_sample

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup and (i + 1) % self.scheduler.order == 0):
                    pbar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        video = self.decode_latents(latents)
        if output_type == "tensor":
            video = torch.from_numpy(video)
        if not return_dict:
            return video
        return TuneAVideoPipelineOutput(videos=video)
