import os
from tqdm import tqdm

base_dir = "/content/drive/MyDrive/EEG2Video_data/processed/"

modalities = {
    "EEG_windows": ".npy",
    "EEG_segments": ".npy",
    "EEG_DE": ".npy",
    "EEG_PSD": ".npy",
    "Video_latents": ".npy",
    "BLIP_embeddings": ".npy",
    "BLIP_text": ".txt"
}

# ------------------------------------------------
# Helpers
# ------------------------------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_split(clip_id):
    # clip_id is 1–5
    return "train" if clip_id < 5 else "test"

def parse_clip(relpath):
    """Return (block, class, clip_id) from a relative path"""
    parts = relpath.split("/")
    fname = parts[-1]
    block = [p for p in parts if p.startswith("Block")][0]
    class_id = int(fname.split("_")[0].replace("class", ""))
    clip_id = int(fname.split("_")[1].replace("clip", "").split(".")[0])
    return block, class_id, clip_id

def collect_files(in_dir, ext):
    files = []
    for root, _, fnames in os.walk(in_dir):
        for f in fnames:
            if f.endswith(ext):
                rel = os.path.relpath(os.path.join(root, f), in_dir)
                files.append(rel)
    return sorted(files)

def write_list(path, items):
    with open(path, "w") as f:
        f.write("\n".join(items))

# ------------------------------------------------
# Main split per modality
# ------------------------------------------------
for mod, ext in modalities.items():
    in_dir = os.path.join(base_dir, mod)
    files = collect_files(in_dir, ext)

    train_list, test_list = [], []
    for rel in tqdm(files, desc=mod):
        block, class_id, clip_id = parse_clip(rel)
        split = get_split(clip_id)
        if split == "train":
            train_list.append(rel)
        else:
            test_list.append(rel)

    write_list(os.path.join(in_dir, "train_list.txt"), train_list)
    write_list(os.path.join(in_dir, "test_list.txt"), test_list)

    print(f"[{mod}] train={len(train_list)} test={len(test_list)}")

print("\nSplitting complete. All modalities aligned by Block/Class/Clip.")
