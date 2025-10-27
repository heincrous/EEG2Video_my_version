# ==========================================
# GRID GENERATOR (EEG→Video Grouped Results + Fail Cases)
# ==========================================
# Input: Result GIF groups (group_1, group_2, …) and fail-case GIFs,
#        each matched to Ground-truth GIFs
# Process: Pair each reconstruction with its GT, randomize, and create grids
# Output: Multi-page grids for each group + single combined grid for fail cases
# ==========================================

import os
import re
import math
import random
from PIL import Image, ImageSequence, ImageDraw, ImageFont


# ==========================================
# Default Configuration
# ==========================================
CONFIG = {
    "base_dir": "/Users/heinrichcrous/Desktop/4022_Code/EEG2Video_my_version",
    "results_subdir": "results",
    "fail_subdir": "fail_cases",
    "groundtruth_subdir": "ground_truth",
    "output_prefix_results": "combined_grid_results",
    "output_prefix_fail": "combined_grid_fail",
    "frames_per_gif": 6,
    "num_pairs": 6,      # Pairs per page for each group
    "label_width": 120,
}


# ==========================================
# Helper Functions
# ==========================================
def match_base(name):
    """Extract base identifier for GT matching (e.g., remove '_2', '_3' suffix)."""
    return re.sub(r"_\d+$", "", name.replace(".gif", ""))


def extract_frames(path, max_frames):
    """Extract RGB frames from a GIF file."""
    with Image.open(path) as im:
        frames = [f.copy().convert("RGB") for f in ImageSequence.Iterator(im)]
        return frames[:max_frames]


def collect_pairs(src_dir, gt_dir, tag):
    """Collect and match GIFs (including duplicates) to ground truths."""
    src_gifs = sorted([f for f in os.listdir(src_dir) if f.endswith(".gif")])
    gt_gifs = sorted([f for f in os.listdir(gt_dir) if f.endswith(".gif")])
    gt_index = {match_base(f): f for f in gt_gifs}

    matched, missing_gt = [], []
    for f in src_gifs:
        base = match_base(f)
        if base in gt_index:
            matched.append((f, gt_index[base]))
        else:
            missing_gt.append(f)

    print(f"\n=== Matching Summary ({tag}) ===")
    print(f"Total GIFs in {tag}: {len(src_gifs)}")
    print(f"Matched pairs: {len(matched)}")

    if missing_gt:
        print(f"\n{tag} files without matching GT ({len(missing_gt)}):")
        for x in missing_gt:
            print(f" - {x}")

    return matched


# ==========================================
# Grid Construction
# ==========================================
def build_grid(cfg, src_dir, gt_dir, matched_pairs, prefix_tag, label_text, single_page=False):
    """Build one grid (single or paginated)."""
    frames_per_gif = cfg["frames_per_gif"]
    base_label_width = cfg["label_width"]

    rows = []
    for res_file, gt_file in matched_pairs:
        res_path = os.path.join(src_dir, res_file)
        gt_path = os.path.join(gt_dir, gt_file)
        res_frames = extract_frames(res_path, frames_per_gif)
        gt_frames = extract_frames(gt_path, frames_per_gif)
        rows.extend([(label_text, res_frames), ("GT", gt_frames)])

    if not rows:
        print(f"\n⚠️ No valid pairs found for {prefix_tag}. Skipping.")
        return

    frame_w, frame_h = rows[0][1][0].size
    cols = frames_per_gif
    total_rows = len(rows)

    try:
        font = ImageFont.truetype("Arial.ttf", size=int(frame_h * 0.25))
    except:
        font = ImageFont.load_default()

    draw_tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    max_label_width = max(draw_tmp.textbbox((0, 0), label, font=font)[2] for label, _ in rows)
    label_width = max(base_label_width, max_label_width + 20)

    grid_w = cols * frame_w + label_width
    grid_h = total_rows * frame_h
    grid = Image.new("RGB", (grid_w, grid_h), (0, 0, 0))
    draw = ImageDraw.Draw(grid)

    for r_idx, (label, frames) in enumerate(rows):
        y_offset = r_idx * frame_h
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        text_x = (label_width - text_w) / 2
        text_y = y_offset + (frame_h - text_h) / 2 - 2
        draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)

        for c_idx, frame in enumerate(frames):
            x_offset = label_width + c_idx * frame_w
            grid.paste(frame, (x_offset, y_offset))

    output_path = os.path.join(cfg["base_dir"], f"{prefix_tag}.png")
    grid.save(output_path)
    print(f"✅ Saved grid: {output_path}")


def build_paged_grids(cfg, src_dir, gt_dir, matched_pairs, prefix_tag, label_text):
    """Build multiple paginated grids for a results group."""
    random.shuffle(matched_pairs)
    num_pairs = cfg["num_pairs"]
    num_pages = math.ceil(len(matched_pairs) / num_pairs)
    print(f"\nGenerating {num_pages} grid pages from {len(matched_pairs)} pairs ({prefix_tag})...")

    for i in range(num_pages):
        batch = matched_pairs[i * num_pairs : (i + 1) * num_pairs]
        suffix = f"_{i+1}" if num_pages > 1 else ""
        page_prefix = f"{prefix_tag}{suffix}"
        build_grid(cfg, src_dir, gt_dir, batch, page_prefix, label_text)


# ==========================================
# Main
# ==========================================
def main():
    print("\n=== EEG→Video Grid Generator (Grouped Results + Fail Cases + GT-Only) ===")
    cfg = CONFIG.copy()

    base_dir = cfg["base_dir"]
    results_root = os.path.join(base_dir, cfg["results_subdir"])
    fails_dir = os.path.join(base_dir, cfg["fail_subdir"])
    gt_dir = os.path.join(base_dir, cfg["groundtruth_subdir"])

    # Process only folders named group_X
    all_subdirs = [d for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))]
    groups = sorted([d for d in all_subdirs if re.match(r"^group_\d+$", d)])
    if not groups:
        print("\n⚠️ No valid result groups found (expected folders like group_1, group_2, ...).")
    else:
        print(f"\nFound {len(groups)} valid result groups: {groups}")

    for group in groups:
        group_dir = os.path.join(results_root, group)
        matched = collect_pairs(group_dir, gt_dir, f"results/{group}")
        prefix_tag = f"{cfg['output_prefix_results']}_{group}"
        build_paged_grids(cfg, group_dir, gt_dir, matched, prefix_tag, "Recon.")

    # Process fail cases (single combined grid)
    matched_fails = collect_pairs(fails_dir, gt_dir, "fail_cases")
    print(f"\nGenerating single combined grid for all fail cases ({len(matched_fails)} pairs)...")
    random.shuffle(matched_fails)
    build_grid(cfg, fails_dir, gt_dir, matched_fails, cfg["output_prefix_fail"], "Fail", single_page=True)

    # Build ground-truth-only grids (10 rows per page)
    print("\n=== Building Ground-Truth-Only Grids ===")
    gt_gifs = sorted([f for f in os.listdir(gt_dir) if f.endswith(".gif")])
    random.shuffle(gt_gifs)
    rows_per_page = 10
    frames_per_gif = cfg["frames_per_gif"]

    try:
        sample_frame = extract_frames(os.path.join(gt_dir, gt_gifs[0]), frames_per_gif)[0]
    except Exception:
        print("⚠️ No valid ground-truth GIFs found.")
        return

    frame_w, frame_h = sample_frame.size
    cols = frames_per_gif

    try:
        font = ImageFont.truetype("Arial.ttf", size=int(frame_h * 0.25))
    except:
        font = ImageFont.load_default()

    num_pages = math.ceil(len(gt_gifs) / rows_per_page)
    print(f"Generating {num_pages} ground-truth grid pages from {len(gt_gifs)} clips...")

    for i in range(num_pages):
        batch = gt_gifs[i * rows_per_page : (i + 1) * rows_per_page]
        grid_h = len(batch) * frame_h
        grid_w = cols * frame_w
        grid = Image.new("RGB", (grid_w, grid_h), (0, 0, 0))

        for r_idx, gt_file in enumerate(batch):
            gt_path = os.path.join(gt_dir, gt_file)
            gt_frames = extract_frames(gt_path, frames_per_gif)
            for c_idx, frame in enumerate(gt_frames):
                grid.paste(frame, (c_idx * frame_w, r_idx * frame_h))

        page_tag = f"combined_grid_groundtruth_{i+1}" if num_pages > 1 else "combined_grid_groundtruth"
        output_path = os.path.join(base_dir, f"{page_tag}.png")
        grid.save(output_path)
        print(f"✅ Saved GT-only grid: {output_path}")


if __name__ == "__main__":
    main()