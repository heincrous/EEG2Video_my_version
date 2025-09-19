import numpy as np
import os
from DE_PSD import DE_PSD
from tqdm import tqdm

fre = 200

# Paths
input_dir = "/content/drive/MyDrive/Data/Processed/EEG_segments/"
output_dir_de = "/content/drive/MyDrive/Data/Processed/EEG_DE_1per1s/"
output_dir_psd = "/content/drive/MyDrive/Data/Processed/EEG_PSD_1per1s/"
os.makedirs(output_dir_de, exist_ok=True)
os.makedirs(output_dir_psd, exist_ok=True)

def get_files_names_in_directory(directory):
    return [f for f in os.listdir(directory) if f.endswith(".npy")]

sub_list = get_files_names_in_directory(input_dir)

for subname in sub_list:
    input_path = os.path.join(input_dir, subname)
    loaded_data = np.load(input_path)  # shape (7,40,5,62,400)
    print(f"Loaded {subname}, shape {loaded_data.shape}")

    DE_data = np.empty((0, 40, 5, 2, 62, 5))
    PSD_data = np.empty((0, 40, 5, 2, 62, 5))

    for block_id in range(7):
        print(" Block:", block_id)
        now_data = loaded_data[block_id]  # (40,5,62,400)
        de_block_data = np.empty((0, 5, 2, 62, 5))
        psd_block_data = np.empty((0, 5, 2, 62, 5))

        for class_id in tqdm(range(40)):
            de_class_data = np.empty((0, 2, 62, 5))
            psd_class_data = np.empty((0, 2, 62, 5))

            for i in range(5):
                # First half (0–1s)
                de1, psd1 = DE_PSD(now_data[class_id, i, :, :200], fre, 1)
                # Second half (1–2s)
                de2, psd2 = DE_PSD(now_data[class_id, i, :, 200:], fre, 1)

                # Stack [2,62,5] and add a clip dimension
                de_pair = np.stack([de1, de2], axis=0).reshape(1, 2, 62, 5)
                psd_pair = np.stack([psd1, psd2], axis=0).reshape(1, 2, 62, 5)

                de_class_data = np.concatenate((de_class_data, de_pair))
                psd_class_data = np.concatenate((psd_class_data, psd_pair))

            de_block_data = np.concatenate((de_block_data, de_class_data.reshape(1, 5, 2, 62, 5)))
            psd_block_data = np.concatenate((psd_block_data, psd_class_data.reshape(1, 5, 2, 62, 5)))

        DE_data = np.concatenate((DE_data, de_block_data.reshape(1, 40, 5, 2, 62, 5)))
        PSD_data = np.concatenate((PSD_data, psd_block_data.reshape(1, 40, 5, 2, 62, 5)))

    np.save(os.path.join(output_dir_de, subname), DE_data)
    np.save(os.path.join(output_dir_psd, subname), PSD_data)
    print(f"Saved DE/PSD (1s window) for {subname}")