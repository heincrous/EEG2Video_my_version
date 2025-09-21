import os
import numpy as np
from glob import glob

# CONFIG
ROOT = "/content/drive/MyDrive/EEG2Video_data/processed/"
SPLIT_SAVE = os.path.join(ROOT, "splits")
SPLIT_RATIO = 0.8  # 80% train, 20% test
os.makedirs(SPLIT_SAVE, exist_ok=True)

# Subfolders to include in split
DATA_FOLDERS = {
    "EEG_segments": os.path.join(ROOT, "EEG_segments"),
    "Video_latents": os.path.join(ROOT, "Video_latents"),
    "BLIP_embeddings": os.path.join(ROOT, "BLIP_embeddings"),
    "Meta_info": os.path.join(ROOT, "Meta_info"),
    # Add EEG_features if needed later
}

def collect_files(folder, pattern="*.npy"):
    """Recursively collect .npy files in a folder"""
    return sorted(glob(os.path.join(folder, "**", pattern), recursive=True))

def split_all_data(split_ratio=0.8):
    # Collect files for each data type
    collected = {k: collect_files(v) for k, v in DATA_FOLDERS.items()}

    # Check consistency (all folders must have same count)
    lengths = [len(v) for v in collected.values()]
    if len(set(lengths)) != 1:
        raise ValueError(f"Mismatch in file counts across folders: {lengths}")
    N = lengths[0]
    print(f"Found {N} samples across {len(DATA_FOLDERS)} folders")

    # Shuffle once
    indices = np.arange(N)
    np.random.shuffle(indices)

    split_idx = int(N * split_ratio)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    print(f"Split: {len(train_idx)} train, {len(test_idx)} test")

    # Save index arrays
    np.save(os.path.join(SPLIT_SAVE, "train_indices.npy"), train_idx)
    np.save(os.path.join(SPLIT_SAVE, "test_indices.npy"), test_idx)

    # Save human-readable mapping
    with open(os.path.join(SPLIT_SAVE, "split_mapping.csv"), "w") as f:
        header = "index," + ",".join(DATA_FOLDERS.keys()) + ",set\n"
        f.write(header)
        for split_name, split_indices in [("train", train_idx), ("test", test_idx)]:
            for i in split_indices:
                row = [str(i)]
                for k in DATA_FOLDERS.keys():
                    row.append(os.path.basename(collected[k][i]))
                row.append(split_name)
                f.write(",".join(row) + "\n")

    return train_idx, test_idx

if __name__ == "__main__":
    split_all_data(SPLIT_RATIO)