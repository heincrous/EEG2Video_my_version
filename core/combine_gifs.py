# ==========================================
# GRID GENERATOR (EEG→Video Result Comparison)
# ==========================================
# Input: Generated result GIFs and matching ground-truth GIFs
# Process: Randomly sample matching pairs, extract frames, label rows, and stack
# Output: Combined comparison grid (.png)
# ==========================================

import os
import re
import random
from PIL import Image, ImageSequence, ImageDraw, ImageFont


# ==========================================
# Default Configuration
# ==========================================
"""
DEFAULT CONFIGURATION

Purpose:
Combine randomly sampled EEG→Video result GIFs with
their corresponding ground-truth GIFs into a visual grid.

Structure:
Each pair (Result + Ground Truth) forms two consecutive rows.
A label is drawn on the left side of each row for clarity.

User Controls:
- frames_per_gif: number of frames per row
- num_pairs: number of sampled pairs per grid
- label_width: reserved width (pixels) for text labels
- random_order: shuffle order of pairs before building grid
"""

CONFIG = {
    "base_dir": "/Users/heinrichcrous/Desktop/4022_Code/EEG2Video_my_version",
    "results_subdir": "results",
    "groundtruth_subdir": "ground_truth",
    "output_filename": "combined_grid.png",
    "frames_per_gif": 6,
    "num_pairs": 5,
    "label_width": 220,
    "random_order": True,
}


# ==========================================
# Utility Functions
# ==========================================
def clean_name(filename):
    """Removes duplicate suffixes like _2 or _trial01 for consistent pairing."""
    return re.sub(r'(_\d+)?(\.gif)?$', '', filename)


def extract_frames(path, max_frames):
    """Extracts RGB frames from a GIF."""
    with Image.open(path) as im:
        frames = [f.copy().convert("RGB") for f in ImageSequence.Iterator(im)]
        return frames[:max_frames]


def collect_gifs(cfg):
    """Collects result and ground-truth GIFs, matching them by cleaned filename."""
    results_dir = os.path.join(cfg["base_dir"], cfg["results_subdir"])
    gt_dir = os.path.join(cfg["base_dir"], cfg["groundtruth_subdir"])

    result_gifs = [f for f in os.listdir(results_dir) if f.endswith(".gif")]
    gt_gifs = [f for f in os.listdir(gt_dir) if f.endswith(".gif")]

    result_map = {clean_name(f): f for f in result_gifs}
    gt_map = {clean_name(f): f for f in gt_gifs}

    common_keys = list(set(result_map.keys()) & set(gt_map.keys()))
    if not common_keys:
        raise ValueError("No matching GIF pairs found between results and ground_truth.")

    return results_dir, gt_dir, result_map, gt_map, common_keys


# ==========================================
# Grid Construction
# ==========================================
def build_grid(cfg, results_dir, gt_dir, result_map, gt_map, common_keys):
    """
    Randomly samples matching pairs and builds a visual grid.
    Each pair contributes two rows: [Reconstructed, Ground Truth].
    Left labels are drawn beside each row.
    """
    frames_per_gif = cfg["frames_per_gif"]
    num_pairs = cfg["num_pairs"]
    label_width = cfg["label_width"]

    # Random sampling
    sampled_keys = random.sample(common_keys, min(num_pairs, len(common_keys)))

    # Randomize final order if enabled
    if cfg["random_order"]:
        random.shuffle(sampled_keys)

    rows = []
    for key in sampled_keys:
        res_path = os.path.join(results_dir, result_map[key])
        gt_path = os.path.join(gt_dir, gt_map[key])

        res_frames = extract_frames(res_path, frames_per_gif)
        gt_frames = extract_frames(gt_path, frames_per_gif)
        rows.extend([("Reconstructed", res_frames), ("Ground Truth", gt_frames)])

    # === Dimensions ===
    frame_w, frame_h = rows[0][1][0].size
    cols = frames_per_gif
    total_rows = len(rows)
    grid_w = cols * frame_w + label_width
    grid_h = total_rows * frame_h
    grid = Image.new("RGB", (grid_w, grid_h), (0, 0, 0))

    # === Font ===
    try:
        font = ImageFont.truetype("Arial.ttf", size=int(frame_h * 0.25))
    except:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(grid)

    # === Paste frames and draw labels ===
    for r_idx, (label, frames) in enumerate(rows):
        y_offset = r_idx * frame_h
        text_y = y_offset + frame_h / 2 - font.getbbox(label)[3] / 2
        draw.text((10, text_y), label, fill=(255, 255, 255), font=font)

        for c_idx, frame in enumerate(frames):
            x_offset = label_width + c_idx * frame_w
            grid.paste(frame, (x_offset, y_offset))

    return grid


# ==========================================
# Save Output
# ==========================================
def save_grid(grid, cfg):
    """Saves combined grid image to disk."""
    output_path = os.path.join(cfg["base_dir"], cfg["output_filename"])
    grid.save(output_path)
    print(f"✅ Combined grid saved to: {output_path}")


# ==========================================
# Main
# ==========================================
def main():
    print("\n=== EEG→Video Result Grid Generator ===")
    cfg = CONFIG.copy()

    results_dir, gt_dir, result_map, gt_map, common_keys = collect_gifs(cfg)
    grid = build_grid(cfg, results_dir, gt_dir, result_map, gt_map, common_keys)
    save_grid(grid, cfg)


if __name__ == "__main__":
    main()
