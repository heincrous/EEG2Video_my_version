import os
import numpy as np
from glob import glob

# CONFIG
ROOT = "/content/drive/MyDrive/EEG2Video_data/processed/"
SPLIT_SAVE = os.path.join(ROOT, "splits")
SPLIT_RATIO = 0.8  # 80% train, 20% test
os.makedirs(SPLIT_SAVE, exist_ok=True)

# Subfolders
EEG_DIR = os.path.join(ROOT, "EEG_segments")
LATENT_DIR = os.path.join(ROOT, "Video_latents")
BLIP_DIR = os.path.join(ROOT, "BLIP_embeddings")
META_DIR = os.path.join(ROOT, "Meta_info")

def collect_files_by_key(folder, exts=("npy", "npz")):
    """Collect files and map them by key like class01_clip01"""
    out = {}
    for ext in exts:
        files = glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True)
        for f in files:
            key = os.path.splitext(os.path.basename(f))[0]  # classXX_clipYY
            out[key] = f
    return out

def split_all_data(split_ratio=0.8):
    eeg_map = collect_files_by_key(EEG_DIR, exts=("npy",))
    latent_map = collect_files_by_key(LATENT_DIR, exts=("npy",))
    blip_map = collect_files_by_key(BLIP_DIR, exts=("npy",))
    meta_map = collect_files_by_key(META_DIR, exts=("npz",))

    # Find intersection of keys across all modalities
    common_keys = set(eeg_map.keys()) & set(latent_map.keys()) & set(blip_map.keys()) & set(meta_map.keys())
    common_keys = sorted(common_keys)
    print(f"Found {len(common_keys)} aligned samples across EEG, Latents, BLIP, Meta")

    if not common_keys:
        raise ValueError("No common samples found! Check naming conventions.")

    # Shuffle once
    indices = np.arange(len(common_keys))
    np.random.shuffle(indices)

    split_idx = int(len(common_keys) * split_ratio)
    train_keys = [common_keys[i] for i in indices[:split_idx]]
    test_keys = [common_keys[i] for i in indices[split_idx:]]

    print(f"Split: {len(train_keys)} train, {len(test_keys)} test")

    # Save index arrays
    np.save(os.path.join(SPLIT_SAVE, "train_keys.npy"), train_keys)
    np.save(os.path.join(SPLIT_SAVE, "test_keys.npy"), test_keys)

    # Save human-readable mapping
    with open(os.path.join(SPLIT_SAVE, "split_mapping.csv"), "w") as f:
        header = "key,EEG_segments,Video_latents,BLIP_embeddings,Meta_info,set\n"
        f.write(header)
        for split_name, key_list in [("train", train_keys), ("test", test_keys)]:
            for k in key_list:
                row = [
                    k,
                    eeg_map.get(k, ""),
                    latent_map.get(k, ""),
                    blip_map.get(k, ""),
                    meta_map.get(k, ""),
                    split_name,
                ]
                f.write(",".join(row) + "\n")

    return train_keys, test_keys

if __name__ == "__main__":
    split_all_data(SPLIT_RATIO)
