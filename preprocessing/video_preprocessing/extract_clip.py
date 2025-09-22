# VIDEO CONVERSION PROCESS (for SEED-DV blocks like 1st_10min.mp4)

# 1. INPUT
# - Raw block videos: 1st_10min.mp4 ... 7th_10min.mp4
# - Properties: ~8 min 40 s, 24 fps, 1920×1080

# 2. SEGMENTATION LOGIC
# - Each block contains 40 classes (concepts).
# - For each class:
#     a) 3 s "rest" period → IGNORE
#     b) Then 5 × 2 s video clips → KEEP
# - Total usable clips per class = 5
# - Total clips per block = 40 × 5 = 200
# - Across 7 blocks = 1400 clips

# 3. CLIP EXTRACTION
# - Extract each 2 s segment (48 frames at 24 fps).
# - Each class contributes 5 × 48 = 240 frames.
# - Skip the 3 s rest (72 frames at 24 fps) before each class.

# 4. NAMING FORMAT
# - Save as: BlockX/classYY_clipZZ.mp4
#     BlockX = block number (1–7)
#     classYY = class index (00–39)
#     clipZZ  = clip index (01–05)
# - Example: Block1/class00_clip01.mp4

# 5. RESOLUTION
# - Resize each clip to 512×512
#   (training dataset loader expects square 512 resolution).

# 6. FRAME COUNT
# - Each 2 s clip = 48 frames.
# - Training loader (TuneMultiVideoDataset) samples 24 frames.
# - Either:
#     a) Keep full 48-frame .mp4 and let loader subsample, OR
#     b) Downsample evenly to 24 frames at preprocessing.

# 7. OUTPUT
# - Organized folders:
#     Block1/class00_clip01.mp4 ... class39_clip05.mp4
#     Block2/class00_clip01.mp4 ... class39_clip05.mp4
#     ...
#     Block7/class39_clip05.mp4
# - Optional: also save .gif versions for visualization.

# 8. ALIGNMENT
# - Captions (BLIP-caption) and meta-info files already follow
#   classYY_clipZZ structure → must match clip naming.

