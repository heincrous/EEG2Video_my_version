import numpy as np
from DE_PSD import DE_PSD
from tqdm import tqdm
import os

fre = 200

# Paths
input_dir = "/content/drive/MyDrive/Data/Processed/EEG_segments/"
output_dir_de = "/content/drive/MyDrive/Data/Processed/EEG_DE_1per2s/"
output_dir_psd = "/content/drive/MyDrive/Data/Processed/EEG_PSD_1per2s/"
os.makedirs(output_dir_de, exist_ok=True)
os.makedirs(output_dir_psd, exist_ok=True)

# Loop through subjects
for sub_id in range(1, 21):  # sub1.npy ... sub20.npy
    subname = f"sub{sub_id}.npy"
    input_path = os.path.join(input_dir, subname)

    if not os.path.exists(input_path):
        print(f"Skipping {subname} (not found)")
        continue

    loaded_data = np.load(input_path)  # shape (7,40,5,62,2*fre)
    print(f"Loaded {subname}, shape {loaded_data.shape}")

    DE_data = np.empty((0, 40, 5, 62, 5))
    PSD_data = np.empty((0, 40, 5, 62, 5))

    for block_id in range(7):
        print(" Block:", block_id)
        now_data = loaded_data[block_id]  # (40,5,62,400)
        de_block_data = np.empty((0, 5, 62, 5))
        psd_block_data = np.empty((0, 5, 62, 5))

        for class_id in tqdm(range(40)):
            de_class_data = np.empty((0, 62, 5))
            psd_class_data = np.empty((0, 62, 5))
            for i in range(5):
                eeg_clip = now_data[class_id, i, :, :]  # (62,400)
                de, psd = DE_PSD(eeg_clip, fre, 2)     # window = 2s
                de_class_data = np.concatenate((de_class_data, de.reshape(1, 62, 5)))
                psd_class_data = np.concatenate((psd_class_data, psd.reshape(1, 62, 5)))

            de_block_data = np.concatenate((de_block_data, de_class_data.reshape(1, 5, 62, 5)))
            psd_block_data = np.concatenate((psd_block_data, psd_class_data.reshape(1, 5, 62, 5)))

        DE_data = np.concatenate((DE_data, de_block_data.reshape(1, 40, 5, 62, 5)))
        PSD_data = np.concatenate((PSD_data, psd_block_data.reshape(1, 40, 5, 62, 5)))

    np.save(os.path.join(output_dir_de, f"sub{sub_id}.npy"), DE_data)
    np.save(os.path.join(output_dir_psd, f"sub{sub_id}.npy"), PSD_data)
    print(f"Saved DE/PSD features for {subname}")
