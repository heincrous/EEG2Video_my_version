import os
from PIL import Image, ImageSequence

# === Paths ===
input_dir = "/Users/heinrichcrous/Desktop/4022_Code/EEG2Video_my_version/results"
output_path = os.path.join(input_dir, "combined_grid.png")

# === Config ===
frames_per_gif = 6  # each concept has 6 frames

# === Load all GIFs ===
gif_files = [f for f in sorted(os.listdir(input_dir)) if f.endswith(".gif")]

# === Extract frames and store them per GIF ===
rows = []
for gif_file in gif_files:
    gif_path = os.path.join(input_dir, gif_file)
    with Image.open(gif_path) as im:
        frames = [frame.copy().convert("RGB") for frame in ImageSequence.Iterator(im)]
        frames = frames[:frames_per_gif]  # only keep first 6
        rows.append(frames)

# === Get dimensions from first frame ===
frame_w, frame_h = rows[0][0].size
cols = frames_per_gif
rows_count = len(rows)

# === Create blank canvas ===
grid = Image.new("RGB", (cols * frame_w, rows_count * frame_h), (0, 0, 0))

# === Paste all frames ===
for row_idx, frames in enumerate(rows):
    for col_idx, frame in enumerate(frames):
        grid.paste(frame, (col_idx * frame_w, row_idx * frame_h))

# === Save ===
grid.save(output_path)
print(f"✅ Combined grid saved to: {output_path}")
