import numpy as np
import os
from tqdm import tqdm

# Segment EEG data into 2-second windows
# Input:  (7, 62, 520s*fre)  →  7 blocks of continuous EEG
# Output: (7, 40, 5, 62, 2*fre) → 7 blocks, 40 concepts, 5 clips, 62 channels, 400 samples (2s @ 200 Hz)

fre = 200

def get_files_names_in_directory(directory):
    return [f for f in os.listdir(directory) if f.endswith(".npy")]

# Input/output paths on Google Drive
input_dir = "/content/drive/MyDrive/Data/Raw/EEG/"
output_dir = "/content/drive/MyDrive/Data/Processed/EEG_segments/"
os.makedirs(output_dir, exist_ok=True)

sub_list = get_files_names_in_directory(input_dir)

for subname in sub_list:
    print(f"Processing {subname} ...")
    npydata = np.load(os.path.join(input_dir, subname))  # shape (7,62,104000)

    save_data = np.empty((0, 40, 5, 62, 2 * fre))

    for block_id in range(7):
        print(" Block:", block_id)
        now_data = npydata[block_id]  # (62,104000)
        l = 0
        block_data = np.empty((0, 5, 62, 2 * fre))
        for class_id in tqdm(range(40)):
            l += 3 * fre  # skip 3s hint
            class_data = np.empty((0, 62, 2 * fre))
            for i in range(5):
                clip = now_data[:, l:l + 2 * fre].reshape(1, 62, 2 * fre)
                class_data = np.concatenate((class_data, clip))
                l += 2 * fre
            block_data = np.concatenate((block_data, class_data.reshape(1, 5, 62, 2 * fre)))
        save_data = np.concatenate((save_data, block_data.reshape(1, 40, 5, 62, 2 * fre)))

    save_path = os.path.join(output_dir, subname)
    np.save(save_path, save_data)
    print(f"Saved segmented EEG: {save_path}")
