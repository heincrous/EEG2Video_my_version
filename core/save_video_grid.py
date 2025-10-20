# ==========================================
# SAVE VIDEO GRID AS GIF
# ==========================================
# Input:
#   videos : torch.Tensor [B, C, T, H, W]
#   path   : output file path (.gif)
#
# Process:
#   - Rearranges tensor into temporal slices
#   - Creates grid across batch dimension
#   - Optionally rescales from [-1,1] → [0,1]
#   - Saves as animated GIF using imageio
#
# Output:
#   Saved GIF at specified path
# ==========================================
import os
import imageio
import numpy as np

import torch
import torchvision
from einops import rearrange


# ==========================================
# Helper: save_videos_grid
# ==========================================
def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=3):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []

    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 → 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

