# ==========================================
# Full Inference (EEG → Video via semantic predictor + diffusion)
# Subject + Feature combo specified manually
# ==========================================
import os, gc, random, imageio, torch, numpy as np
from sklearn.preprocessing import StandardScaler
from diffusers import AutoencoderKL, DDIMScheduler
from einops import rearrange

from core.unet import UNet3DConditionModel
from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
# from train_semantic_predictor import SemanticPredictor, FusionNet, MODEL_MAP, FEATURE_PATHS

# ==========================================
# Config (set these manually)
# ==========================================
SUBJECT       = "sub1.npy"          # e.g. "sub10.npy"
FEATURE_TYPES = ["DE"]        # e.g. ["DE"], ["segments"], ["DE","PSD"]

# CHECKPOINT_DIR     = "/content/drive/MyDrive/EEG2Video_checkpoints/semantic_checkpoints"
PRETRAINED_SD_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"
FINETUNED_SD_PATH  = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints/pipeline_final"
OUTPUT_DIR         = "/content/drive/MyDrive/EEG2Video_outputs/test_full_inference"
BLIP_TEXT_PATH     = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/BLIP_text.npy"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect(); torch.cuda.empty_cache()

# ==========================================
# Load ground truth captions
# ==========================================
# blip_text = np.load(BLIP_TEXT_PATH, allow_pickle=True)  # shape (7,40,5)

# def get_caption(idx):
#     # Block 6 (the 7th block) is test block
#     class_id = idx // 10          # 40 classes
#     clip_id  = (idx % 10) // 2    # 5 clips, each repeated twice
#     return blip_text[6, class_id, clip_id]

caption= np.load(BLIP_TEXT_PATH, allow_pickle=True)  # shape (7,40,5)

def get_caption(idx=None):
    # 7th block (index 6), class index 1, clip index 0
    return caption[6, 1, 0]

# ==========================================
# Load semantic checkpoint
# ==========================================
# ft_tag  = "_".join(FEATURE_TYPES)
# sub_tag = SUBJECT.replace(".npy", "")
# ckpt_pattern = f"semantic_predictor_{ft_tag}_{sub_tag}.pt"
# ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_pattern)

# if not os.path.exists(ckpt_path):
#     raise FileNotFoundError(f"No checkpoint found: {ckpt_pattern}")

# print("Using checkpoint:", ckpt_path)
# ckpt_data   = torch.load(ckpt_path, map_location=device)
# state_dict  = ckpt_data["state_dict"]
# saved_feats = ckpt_data["feature_types"]

# if set(saved_feats) != set(FEATURE_TYPES):
#     raise ValueError(f"Checkpoint trained with {saved_feats}, "
#                      f"but you specified {FEATURE_TYPES}")

# ==========================================
# Build semantic predictor
# ==========================================
# dim_map = {"DE": 128, "PSD": 128, "segments": 256}

# if len(FEATURE_TYPES) > 1:
#     encoders  = {ft: MODEL_MAP[ft]() for ft in FEATURE_TYPES}
#     total_dim = sum(dim_map[ft] for ft in FEATURE_TYPES)
#     encoder   = FusionNet(encoders, total_dim)
#     input_dim = total_dim
#     multi     = True
# else:
#     ft        = FEATURE_TYPES[0]
#     encoder   = MODEL_MAP[ft]()
#     input_dim = dim_map[ft]
#     multi     = False

# model = SemanticPredictor(encoder, input_dim).to(device)
# model.load_state_dict(state_dict, strict=False)
# model.eval()

# ==========================================
# Load diffusion backbone
# ==========================================
vae = AutoencoderKL.from_pretrained(PRETRAINED_SD_PATH, subfolder="vae").to(device, dtype=torch.float32)
scheduler = DDIMScheduler.from_pretrained(PRETRAINED_SD_PATH, subfolder="scheduler")
unet = UNet3DConditionModel.from_pretrained_2d(PRETRAINED_SD_PATH, subfolder="unet").to(device, dtype=torch.float32)

pipe = TuneAVideoPipeline.from_pretrained(
    FINETUNED_SD_PATH,
    vae=vae,
    unet=unet,
    scheduler=scheduler,
    torch_dtype=torch.float32,
).to(device)
pipe.enable_vae_slicing()

# ==========================================
# Load EEG features + scale
# ==========================================
# def load_features(subname, ft):
#     path = os.path.join(FEATURE_PATHS[ft], subname)
#     arr = np.load(path)
#     if ft in ["DE", "PSD"]:
#         arr = arr.reshape(-1, 62, 5)
#     elif ft == "segments":
#         arr = rearrange(arr, "a b c d (w t) -> (a b c w) d t", w=2, t=200)
#     return arr

# samples_per_block = 400
# train_idx = np.arange(0, 5 * samples_per_block)   # blocks 0–4 only
# test_idx  = np.arange(6 * samples_per_block, 7 * samples_per_block)  # block 6

# if multi:
#     feats_test = {}
#     for ft in FEATURE_TYPES:
#         arr = load_features(SUBJECT, ft)
#         scaler = StandardScaler().fit(arr[train_idx].reshape(len(train_idx), -1))
#         feats_test[ft] = scaler.transform(arr[test_idx].reshape(len(test_idx), -1)).reshape(arr[test_idx].shape)
#     test_data = feats_test
# else:
#     ft = FEATURE_TYPES[0]
#     arr = load_features(SUBJECT, ft)
#     scaler = StandardScaler().fit(arr[train_idx].reshape(len(train_idx), -1))
#     test_data = scaler.transform(arr[test_idx].reshape(len(test_idx), -1)).reshape(arr[test_idx].shape)

# ==========================================
# Inference helper
# ==========================================
# def run_inference(eeg_feat, idx):
#     if multi:
#         eeg_tensor = {ft: torch.tensor(eeg_feat[ft], dtype=torch.float32).unsqueeze(0).to(device) for ft in eeg_feat}
#     else:
#         eeg_tensor = torch.tensor(eeg_feat, dtype=torch.float32).unsqueeze(0).to(device)

#     with torch.no_grad():
#         semantic_pred = model(eeg_tensor)

#     # reshape to (B,77,768)
#     semantic_pred = semantic_pred.view(semantic_pred.size(0), 77, 768)

#     negative = semantic_pred.mean(dim=0, keepdim=True).float().to(device)

#     video_length, fps = 6, 3
#     video = pipe(
#         model=None,
#         eeg=semantic_pred,
#         negative_eeg=negative,
#         latents=None,
#         video_length=video_length,
#         height=288,
#         width=512,
#         num_inference_steps=100,
#         guidance_scale=12.5,
#     ).videos

def run_inference():
    sem_path = "/content/drive/MyDrive/EEG2Video_outputs/semantic_embeddings/embeddings_semantic_predictor_DE_sub1_subset1-10-12-16-19-23-25-31-34-39.npy"

    sem_preds = np.load(sem_path)   # shape (50, 6, 4, 6, 36, 64) or similar
    first_emb = sem_preds.reshape(-1, 77, 768)[0]  # flatten to (N,77,768) then take [0]

    # add batch dimension for pipeline use
    semantic_pred = torch.tensor(first_emb, dtype=torch.float32).unsqueeze(0).to(device)

    # for negative prompt
    negative = semantic_pred.mean(dim=0, keepdim=True).float().to(device)

    video_length, fps = 6, 3
    video = pipe(
        model=None,
        eeg=semantic_pred,
        negative_eeg=negative,
        latents=None,
        video_length=video_length,
        height=288,
        width=512,
        num_inference_steps=100,
        guidance_scale=12.5,
    ).videos

    frames = (video[0] * 255).clamp(0,255).to(torch.uint8).permute(0,2,3,1).cpu().numpy()

    if frames.shape[-1] > 3: frames = frames[...,:3]
    elif frames.shape[-1] == 1: frames = np.repeat(frames, 3, axis=-1)

    # caption = get_caption(idx)
    # inf_base = f"{SUBJECT.replace('.npy','')}_{ft_tag}_sample{idx}"
    # inf_path = os.path.join(OUTPUT_DIR, inf_base + ".mp4")
    # writer = imageio.get_writer(inf_path, fps=fps, codec="libx264")
    # for f in frames: writer.append_data(f)
    # writer.close()

    # txt_path = os.path.join(OUTPUT_DIR, inf_base + ".txt")
    # with open(txt_path, "w") as f:
    #     f.write(caption + "\n")

    # print("Saved inference video:", inf_path)
    # print("Ground truth caption:", caption)

    # save outputs with fixed name "test"
    inf_base = "test"
    inf_path = os.path.join(OUTPUT_DIR, inf_base + ".mp4")
    writer = imageio.get_writer(inf_path, fps=fps, codec="libx264")
    for f in frames:
        writer.append_data(f)
    writer.close()

    txt_path = os.path.join(OUTPUT_DIR, inf_base + ".txt")
    with open(txt_path, "w") as f:
        f.write(caption + "\n")

    print("Saved inference video:", inf_path)
    print("Ground truth caption:", caption)

# ==========================================
# Run 5 random samples
# ==========================================
# if multi:
#     num_samples = next(iter(test_data.values())).shape[0]
# else:
#     num_samples = test_data.shape[0]

# for n in range(5):
#     idx = random.randrange(num_samples)
#     eeg_feat = {ft: test_data[ft][idx] for ft in test_data} if multi else test_data[idx]
#     run_inference(eeg_feat, idx)

run_inference()

# # ==========================================
# # Full Inference: EEG2Video (all subset videos)
# # Save outputs as block_class_clip_sample.mp4
# # ==========================================
# import os, gc, torch, numpy as np, imageio
# from einops import rearrange
# from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
# from diffusers import AutoencoderKL, DDIMScheduler
# from transformers import CLIPTokenizer
# from core.unet import UNet3DConditionModel


# # ==========================================
# # Paths
# # ==========================================
# SEMANTIC_DIR = "/content/drive/MyDrive/EEG2Video_outputs/semantic_embeddings"
# SEQ2SEQ_DIR  = "/content/drive/MyDrive/EEG2Video_outputs/seq2seq_latents"
# DANA_DIR     = "/content/drive/MyDrive/EEG2Video_outputs/dana_latents"
# PIPELINE_DIR = "/content/drive/MyDrive/EEG2Video_checkpoints/diffusion_checkpoints"
# BLIP_TEXT    = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_text/BLIP_text.npy"

# SAVE_ROOT    = "/content/drive/MyDrive/EEG2Video_outputs/final_videos"
# os.makedirs(SAVE_ROOT, exist_ok=True)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# gc.collect(); torch.cuda.empty_cache()


# # ==========================================
# # Select semantic embeddings file
# # ==========================================
# sem_files = [f for f in os.listdir(SEMANTIC_DIR) if f.endswith(".npy")]
# print("Available semantic embeddings:")
# for i,f in enumerate(sem_files):
#     print(f"[{i}] {f}")
# sem_choice = int(input("Select semantic embeddings index: "))
# sem_file   = sem_files[sem_choice]
# sem_path   = os.path.join(SEMANTIC_DIR, sem_file)

# # deduce subject + subset + pipeline
# subject_tag = next((p for p in sem_file.replace(".npy","").split("_") if p.startswith("sub")), None)
# if "subset" in sem_file:
#     subset_str   = sem_file.split("subset")[1].replace(".npy","")
#     class_subset = [int(x) for x in subset_str.split("-")]
#     pipeline_path = os.path.join(PIPELINE_DIR, f"pipeline_final_subset{subset_str}")
#     pipeline_tag  = f"pipeline_subset{subset_str}"
# else:
#     class_subset = list(range(40))
#     pipeline_path = os.path.join(PIPELINE_DIR, "pipeline_final")
#     pipeline_tag  = "pipeline_full"

# print("Semantic file:", sem_file)
# print("Subject:", subject_tag)
# print("Class subset:", class_subset)
# print("Pipeline path:", pipeline_path)


# # ==========================================
# # Select Seq2Seq and DANA latents
# # ==========================================
# lat_files = [f for f in os.listdir(SEQ2SEQ_DIR) if subject_tag in f]
# print("Available Seq2Seq latents:")
# for i,f in enumerate(lat_files): print(f"[{i}] {f}")
# lat_choice = int(input("Select Seq2Seq latents index: "))
# lat_path   = os.path.join(SEQ2SEQ_DIR, lat_files[lat_choice])

# dana_files = [f for f in os.listdir(DANA_DIR) if subject_tag in f]
# print("Available DANA latents:")
# for i,f in enumerate(dana_files): print(f"[{i}] {f}")
# dana_choice = int(input("Select DANA latents index: "))
# dana_path   = os.path.join(DANA_DIR, dana_files[dana_choice])


# # ==========================================
# # Load semantic embeddings
# # ==========================================
# eeg_embeds = np.load(sem_path)  # (N,77,768)
# eeg_embeds = torch.from_numpy(eeg_embeds).to(device, dtype=torch.float32)
# negative   = eeg_embeds.mean(dim=0, keepdim=True)

# # ==========================================
# # Load latents
# # ==========================================
# latents = np.load(lat_path)   # (B,F,C,H,W)
# latents = np.repeat(latents, 2, axis=0)
# latents = torch.from_numpy(latents).to(device, dtype=torch.float32).permute(0,2,1,3,4)

# latents_add_noise = np.load(dana_path)   # (B,F,C,H,W)
# latents_add_noise = np.repeat(latents_add_noise, 2, axis=0)  # double samples
# latents_add_noise = torch.from_numpy(latents_add_noise).to(device, dtype=torch.float32).permute(0,2,1,3,4)

# assert eeg_embeds.shape[0] == latents.shape[0] == latents_add_noise.shape[0]


# # ==========================================
# # Load pipeline (pretrained backbone + finetuned UNet)
# # ==========================================
# PRETRAINED_MODEL_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4"

# vae       = AutoencoderKL.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="vae").to(device, dtype=torch.float32)
# scheduler = DDIMScheduler.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="scheduler")
# tokenizer = CLIPTokenizer.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="tokenizer")

# unet = UNet3DConditionModel.from_pretrained(
#     pipeline_path,
#     subfolder="unet",
#     torch_dtype=torch.float32
# ).to(device)

# pipe = TuneAVideoPipeline.from_pretrained(
#     PRETRAINED_MODEL_PATH,
#     vae=vae,
#     unet=unet,
#     scheduler=scheduler,
#     tokenizer=tokenizer,
#     torch_dtype=torch.float32
# ).to(device)

# pipe.enable_vae_slicing()


# # ==========================================
# # Captions
# # ==========================================
# blip_text = np.load(BLIP_TEXT, allow_pickle=True)  # (7,40,5)
# block7_caps = blip_text[6]  # (40,5)


# # ==========================================
# # Inference
# # ==========================================
# woSeq2Seq = True
# woDANA    = True

# for i in range(len(eeg_embeds)):
#     cls_index    = i // 10
#     clip_index   = (i % 10) // 2
#     sample_index = i % 2
#     class_id     = class_subset[cls_index]
#     block_id     = 7
#     caption      = block7_caps[class_id, clip_index]

#     print(f"[Block {block_id} | Class {class_id} | Clip {clip_index} | Sample {sample_index}] Caption: {caption}")

#     # enforce float32 every loop
#     eeg_input = eeg_embeds[i:i+1].to(device, dtype=torch.float32)
#     neg_input = negative.to(device, dtype=torch.float32)

#     if woSeq2Seq:
#         video = pipe(
#             model=None,
#             eeg=eeg_input,            # (1,77,768)
#             negative_eeg=neg_input,   # (1,77,768)
#             latents=latents_add_noise[i:i+1] if not woDANA else (
#                 latents[i:i+1] if not woSeq2Seq else None
#             ),
#             video_length=6,
#             height=288,
#             width=512,
#             num_inference_steps=100,
#             guidance_scale=12.5,
#         ).videos
#         save_dir = os.path.join(SAVE_ROOT, f"{pipeline_tag}_FullModel")

#     elif woDANA:
#         video = pipe(
#             None,
#             eeg_input,
#             negative_eeg=neg_input,
#             latents=latents[i:i+1].to(dtype=torch.float32),
#             video_length=6, height=288, width=512,
#             num_inference_steps=100, guidance_scale=12.5
#         ).videos
#         save_dir = os.path.join(SAVE_ROOT, f"{pipeline_tag}_woDANA")

#     else:
#         video = pipe(
#             model=None,
#             eeg=eeg_input,            # (1,77,768)
#             negative_eeg=neg_input,   # (1,77,768)
#             latents=latents_add_noise[i:i+1] if not woDANA else (
#                 latents[i:i+1] if not woSeq2Seq else None
#             ),
#             video_length=6,
#             height=288,
#             width=512,
#             num_inference_steps=100,
#             guidance_scale=12.5,
#         ).videos
#         save_dir = os.path.join(SAVE_ROOT, f"{pipeline_tag}_FullModel")

#     os.makedirs(save_dir, exist_ok=True)

#     # convert tensor to numpy frames
#     frames = (video[0] * 255).clamp(0,255).to(torch.uint8).permute(0,2,3,1).cpu().numpy()

#     # safeguard: enforce 3-channel RGB
#     if frames.shape[-1] > 3:
#         frames = frames[..., :3]
#     elif frames.shape[-1] == 1:
#         frames = np.repeat(frames, 3, axis=-1)

#     fps = 3
#     save_path = os.path.join(
#         save_dir,
#         f"block{block_id}_class{class_id}_clip{clip_index}_sample{sample_index}.mp4"
#     )

#     writer = imageio.get_writer(save_path, fps=fps, codec="libx264")
#     for f in frames:
#         writer.append_data(f)
#     writer.close()

#     print("Saved:", save_path)
