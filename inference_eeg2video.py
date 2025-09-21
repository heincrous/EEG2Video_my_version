'''
Description: 
Author: Zhou Tianyi
LastEditTime: 2025-04-11 12:10:33
LastEditors:  
'''
from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from models.unet import UNet3DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from tuneavideo.util import save_videos_grid,ddim_inversion
import torch
from models.train_semantic_predictor import CLIP
import numpy as np
from einops import rearrange
from sklearn import preprocessing
import random
import math
model = None
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
import os
from diffusers.schedulers import (
    DDPMScheduler,
    DDIMScheduler,
)
torch.cuda.set_device(2)
def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')
seed_everything(114514)

eeg = torch.load('',map_location='cpu')

negative = eeg.mean(dim=0)

pretrained_model_path = "Zhoutianyi/huggingface/stable-diffusion-v1-4"
my_model_path = "outputs/40_classes_200_epoch"

unet = UNet3DConditionModel.from_pretrained(my_model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')
pipe = TuneAVideoPipeline.from_pretrained(pretrained_model_path ,unet=unet, torch_dtype=torch.float16).to("cuda")
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()



latents = np.load('Seq2Seq/latent_out_block7_40_classes.npy')
latents = torch.from_numpy(latents).half()
latents = rearrange(latents, 'a b c d e -> a c b d e')
latents = latents.to('cuda')

latents_add_noise = torch.load('DANA/40_classes_latent_add_noise.pt')
latents_add_noise = latents_add_noise.half()
latents_add_noise = rearrange(latents_add_noise, 'a b c d e -> a c b d e')
latents_add_noise = latents_add_noise.to('cuda')
# Ablation, inference w/o Seq2Seq and w/o DANA
woSeq2Seq = False
woDANA = False
for i in range(len(eeg)):
    if woSeq2Seq:
        video = pipe(model, eeg[i:i+1,...],negative_eeg=negative, latents=None, video_length=6, height=288, width=512, num_inference_steps=100, guidance_scale=12.5).videos
        savename = f'40_Classes_woSeq2Seq/EEG2Video/'
        import os
        os.makedirs(savename, exist_ok=True)
    elif woDANA:
        video = pipe(model, eeg[i:i+1,...],negative_eeg=negative, latents=latents[i:i+1,...], video_length=6, height=288, width=512, num_inference_steps=100, guidance_scale=12.5).videos
        savename = f'40_Classes_woDANA/EEG2Video/'
        import os
        os.makedirs(savename, exist_ok=True)
    else:
        video = pipe(model, eeg[i:i+1,...],negative_eeg=negative, latents=latents_add_noise[i:i+1,...], video_length=6, height=288, width=512, num_inference_steps=100, guidance_scale=12.5).videos
        savename = f'40_Classes_Fullmodel/EEG2Video/'
        import os
        os.makedirs(savename, exist_ok=True)
    save_videos_grid(video, f"./{savename}/{i}.gif")
 
# ---------------------------------------------------------------------------------------------------------------
# NEW VERSION
# ---------------------------------------------------------------------------------------------------------------
import os
import torch
import numpy as np
from einops import rearrange
from torch.utils.data import DataLoader
from torchvision.io import write_video
import imageio

from my_autoregressive_transformer import myTransformer, EEGVideoDataset  # import your classes

# -----------------------
# Config
# -----------------------
BASE = "/content/drive/MyDrive/EEG2Video_data/processed/Split_4train1test"
CKPT_PATH = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_checkpoint.pt"
SAVE_DIR = "/content/drive/MyDrive/EEG2Video_checkpoints/seq2seq_inference"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load test data
# -----------------------
test_ds = EEGVideoDataset(
    os.path.join(BASE, "test/EEG_segments"),
    os.path.join(BASE, "test/Video_latents")
)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

# -----------------------
# Load models
# -----------------------
trained_model = myTransformer().to(device)
state = torch.load(CKPT_PATH, map_location=device)
trained_model.load_state_dict(state["state_dict"])
trained_model.eval()

random_model = myTransformer().to(device)
random_model.eval()

# -----------------------
# Inference loop
# -----------------------
criterion = torch.nn.MSELoss()

for i, (eeg, vid) in enumerate(test_loader):
    eeg, vid = eeg.to(device), vid.to(device)  # vid shape: (F, 4, 36, 64) or (1,F,4,36,64)
    b, f, c, h, w = vid.shape
    padded = torch.zeros((b, 1, c, h, w)).to(device)
    full_vid = torch.cat((padded, vid), dim=1)  # (B, F+1, C, H, W)

    # Run trained
    with torch.no_grad():
        _, out_trained = trained_model(eeg, full_vid)
    trained_latents = out_trained[:, :-1, :].cpu().numpy()

    # Run random
    with torch.no_grad():
        _, out_random = random_model(eeg, full_vid)
    random_latents = out_random[:, :-1, :].cpu().numpy()

    # Ground truth
    gt = vid.cpu().numpy()

    # Convert to gifs (using imageio)
    def save_gif(array, path):
        # array shape (frames, 4, 36, 64)
        array = array.squeeze()
        if array.ndim == 4:
            frames = [array[t,0,:,:] for t in range(array.shape[0])]  # take channel 0
        else:
            frames = [array[t] for t in range(array.shape[0])]
        frames = [((f - f.min()) / (f.max() - f.min() + 1e-8) * 255).astype(np.uint8) for f in frames]
        imageio.mimsave(path, frames, fps=4)

    save_gif(gt, os.path.join(SAVE_DIR, f"sample{i}_groundtruth.gif"))
    save_gif(trained_latents, os.path.join(SAVE_DIR, f"sample{i}_trained.gif"))
    save_gif(random_latents, os.path.join(SAVE_DIR, f"sample{i}_random.gif"))

    print(f"Saved gifs for sample {i}")
    if i == 2:  # just a few samples
        break
