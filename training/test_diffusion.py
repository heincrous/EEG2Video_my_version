import os, sys, random, torch, imageio, gc
from einops import rearrange
from diffusers import DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

repo_root = "/content/EEG2Video_my_version"
sys.path.append(os.path.join(repo_root, "pipelines"))
from pipeline_tuneavideo import TuneAVideoPipeline
from unet import UNet3DConditionModel

# === DIRECTORIES ===
pretrained_model_path = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
trained_output_dir    = "/content/drive/MyDrive/EEG2Video_outputs"
test_text_list        = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/test_list.txt"
save_dir              = os.path.join(trained_output_dir, "test_samples")
os.makedirs(save_dir, exist_ok=True)

# === MEMORY CONFIG ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Clear any cached memory
gc.collect()
torch.cuda.empty_cache()

# === LOAD TRAINED PIPELINE WITH FP16 ===
pipe = TuneAVideoPipeline(
    vae=AutoencoderKL.from_pretrained(trained_output_dir, subfolder="vae", torch_dtype=torch.float16),
    text_encoder=CLIPTextModel.from_pretrained(trained_output_dir, subfolder="text_encoder", torch_dtype=torch.float16),
    tokenizer=CLIPTokenizer.from_pretrained(trained_output_dir, subfolder="tokenizer"),
    unet=UNet3DConditionModel.from_pretrained_2d(trained_output_dir, subfolder="unet"),  # no torch_dtype here
    scheduler=DDIMScheduler.from_pretrained(trained_output_dir, subfolder="scheduler"),
)

# cast UNet after loading
pipe.unet.to(torch.float16)

pipe.enable_vae_slicing()
pipe = pipe.to("cuda")

# === PICK RANDOM PROMPT FROM TEST LIST ===
with open(test_text_list, "r") as f:
    test_prompts = [line.strip() for line in f]

prompt = random.choice(test_prompts)
print("Chosen prompt:", prompt)

# === GENERATE VIDEO ===
generator = torch.Generator(device="cuda").manual_seed(42)
result = pipe(
    prompt,
    video_length=8,          # keep consistent with training validation
    num_inference_steps=20,  # reduce if still OOM
    generator=generator,
)

video_tensor = result.videos  # [1, f, c, h, w]

# === SAVE MP4 ===
# video_tensor: [1, f, c, h, w]
frames = (video_tensor[0] * 255).clamp(0, 255).to(torch.uint8)  # [f, c, h, w]
frames = frames.permute(0, 2, 3, 1).cpu().numpy()               # [f, h, w, c]

# enforce 3 channels only
if frames.shape[-1] > 3:
    frames = frames[..., :3]
elif frames.shape[-1] == 1:
    frames = frames.repeat(3, axis=-1)

print("Final frame shape:", frames.shape)  # should be (f, h, w, 3)

mp4_path = os.path.join(save_dir, "sample_test.mp4")
imageio.mimsave(mp4_path, [f for f in frames], fps=8)

print("Video saved to:", mp4_path)



