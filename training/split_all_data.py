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

def collect_files(in_dir, ext):
    files = []
    for root, _, fnames in os.walk(in_dir):
        for f in fnames:
            if f.endswith(ext):
                rel = os.path.relpath(os.path.join(root, f), in_dir)
                files.append(rel)
    return sorted(files)

def get_split_from_name(fname):
    """Decide train/test by clip id in filename (classXX_clipYY.*)"""
    parts = fname.split("_")
    clip_id = int(parts[1].replace("clip", "").split(".")[0])
    return "train" if clip_id < 5 else "test"

def write_list(path, items):
    with open(path, "w") as f:
        f.write("\n".join(items))

for mod, ext in modalities.items():
    in_dir = os.path.join(base_dir, mod)
    files = collect_files(in_dir, ext)

    train_list, test_list = [], []
    for rel in tqdm(files, desc=mod):
        fname = os.path.basename(rel)
        split = get_split_from_name(fname)
        if split == "train":
            train_list.append(rel)
        else:
            test_list.append(rel)

    write_list(os.path.join(in_dir, "train_list.txt"), train_list)
    write_list(os.path.join(in_dir, "test_list.txt"), test_list)

    print(f"[{mod}] train={len(train_list)} test={len(test_list)}")

print("\nSplitting complete. Train/test lists created for all modalities.")
