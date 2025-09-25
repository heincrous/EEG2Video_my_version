# EEG2Video_data/
# └── processed/
#     ├── EEG_segments/                          # segmented raw EEG
#     │   ├── sub1.npy        → [7,40,5,62, 400]   (7 blocks, 40 classes, 5 clips,
#     │   ├── sub2.npy           2s segment with 400 samples at 200 Hz × 62 channels)
#     │   └── ...
#     │
#     ├── EEG_windows/                           # EEG windowed features
#     │   ├── sub1.npy        → [7,40,5,7,62,100] (each 2s segment split into 7 windows
#     │   ├── sub2.npy           of length 100 with 50 overlap, per channel)
#     │   └── ...
#     │
#     ├── EEG_DE/                                # Differential Entropy features
#     │   ├── sub1.npy        → [7,40,5,62,5]     (5 frequency bands per channel)
#     │   ├── sub2.npy
#     │   └── ...
#     │
#     ├── EEG_PSD/                               # Power Spectral Density features
#     │   ├── sub1.npy        → [7,40,5,62,5]     (same 5 bands per channel)
#     │   ├── sub2.npy
#     │   └── ...
#     │
#     ├── Video_mp4/                             # raw video stimuli (per block)
#     │   ├── Block1/class00_clip01.mp4
#     │   ├── Block1/class00_clip02.mp4
#     │   └── ...
#     │
#     ├── Video_latents/
#     │   └── Video_latents.npy → [7,40,5,6,4,36,64]
#     │                           (7 blocks × 40 classes × 5 clips × 6 frames,
#     │                            each frame latent is [4,36,64] from SD VAE)
#     │
#     ├── BLIP_text/
#     │   └── BLIP_text.npy     → [7,40,5] (string captions aligned to each clip)
#     │
#     ├── BLIP_embeddings/
#     │   └── BLIP_embeddings.npy → [7,40,5,77,768]
#     │                             (per-caption CLIP embeddings, 77 tokens × 768 dims)
#     │
#     ├── BLIP_Video_bundle.npz                  # combined bundle (block-level)
#     │       keys:
#     │         "BLIP_text"       → [7,40,5] captions
#     │         "BLIP_embeddings" → [7,40,5,77,768]
#     │         "Video_latents"   → [7,40,5,6,4,36,64]
#     │
#     └── BLIP_EEG_bundle.npz                    # combined bundle (subject-level)
#             keys:
#               "BLIP_text"       → [7,40,5] captions
#               "BLIP_embeddings" → [7,40,5,77,768]
#               "EEG_data"        → dict of subjects:
#                                    EEG_segments[subX] → [7,40,5,62, 400]
#                                    EEG_windows[subX]  → [7,40,5,7,62,100]
#                                    EEG_DE[subX]       → [7,40,5,62,5]
#                                    EEG_PSD[subX]      → [7,40,5,62,5]

import os

# Path to processed dataset
base_path = "/content/drive/MyDrive/EEG2Video_data/processed"

# Walk through and preview structure
for root, dirs, files in os.walk(base_path):
    level = root.replace(base_path, '').count(os.sep)
    indent = ' ' * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 4 * (level + 1)
    for f in files[:10]:  # preview max 10 files per folder
        print(f"{subindent}{f}")
    if len(files) > 10:
        print(f"{subindent}... ({len(files)} files total)")
