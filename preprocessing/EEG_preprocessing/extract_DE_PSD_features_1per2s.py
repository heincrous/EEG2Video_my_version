import os
import numpy as np
from DE_PSD import DE_PSD
from tqdm import tqdm

# Parameters
fre = 200
in_dir = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments/"
out_de_dir = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_DE/"
out_psd_dir = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_PSD/"
os.makedirs(out_de_dir, exist_ok=True)
os.makedirs(out_psd_dir, exist_ok=True)

# Subjects
subjects = [s for s in sorted(os.listdir(in_dir)) if os.path.isdir(os.path.join(in_dir, s))]

print("\nAvailable subjects:")
for idx, subj in enumerate(subjects):
    print(f"{idx}: {subj}")

choices = input("\nEnter subject indices to process (comma separated): ")
choices = [int(c.strip()) for c in choices.split(",") if c.strip().isdigit()]

for idx in choices:
    subj = subjects[idx]
    subj_path = os.path.join(in_dir, subj)

    print(f"\nProcessing subject {subj}...")

    DE_data = np.empty((0, 40, 5, 62, 5))
    PSD_data = np.empty((0, 40, 5, 62, 5))

    for block_id, block in enumerate(sorted(os.listdir(subj_path))):
        block_path = os.path.join(subj_path, block)
        if not os.path.isdir(block_path):
            continue
        print("block:", block_id)

        de_block_data = np.empty((0, 5, 62, 5))
        psd_block_data = np.empty((0, 5, 62, 5))

        # Expecting 40 classes per block
        classes = sorted([f for f in os.listdir(block_path) if f.endswith(".npy")])
        class_map = {}
        for f in classes:
            parts = f.replace(".npy", "").split("_")
            class_id = int(parts[0].replace("class", ""))
            clip_id = int(parts[1].replace("clip", ""))
            if class_id not in class_map:
                class_map[class_id] = {}
            class_map[class_id][clip_id] = os.path.join(block_path, f)

        for class_id in tqdm(range(40)):
            de_class_data = np.empty((0, 62, 5))
            psd_class_data = np.empty((0, 62, 5))
            for i in range(1, 6):
                seg = np.load(class_map[class_id][i])
                # [400,62] → reshape to (62, 400)
                seg = seg.T
                de, psd = DE_PSD(seg, fre, 2)
                de_class_data = np.concatenate((de_class_data, de.reshape(1, 62, 5)))
                psd_class_data = np.concatenate((psd_class_data, psd.reshape(1, 62, 5)))
            de_block_data = np.concatenate((de_block_data, de_class_data.reshape(1, 5, 62, 5)))
            psd_block_data = np.concatenate((psd_block_data, psd_class_data.reshape(1, 5, 62, 5)))

        DE_data = np.concatenate((DE_data, de_block_data.reshape(1, 40, 5, 62, 5)))
        PSD_data = np.concatenate((PSD_data, psd_block_data.reshape(1, 40, 5, 62, 5)))

    np.save(os.path.join(out_de_dir, subj + ".npy"), DE_data)
    np.save(os.path.join(out_psd_dir, subj + ".npy"), PSD_data)

print("\nProcessing complete.")
