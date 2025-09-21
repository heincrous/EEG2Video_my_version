# import numpy as np
# from DE_PSD import DE_PSD
# from tqdm import tqdm

# # Extract DE or PSD features with a 2-second window, that is, for each 2-second EEG segment, we extract a DE or PSD feature.
# # Input the shape of (7 * 40 * 5 * 62 * 2s*fre), meaning 7 blocks, 40 concepts, 5 video clips, 62 channels, and 2s*fre time-points.
# # Output the DE or PSD feature with (7 * 40 * 5 * 62 * 5), the last 5 indicates the frequency bands' number.

# fre = 200

# for subname in range(1,21):

#     loaded_data = np.load('data/EEG2Video/Segmented_Rawf_200Hz_2s/sub'+ str(subname) + '.npy')
#     # (7 * 40 * 5 * 62 * 2*fre)

#     print("Successfully loaded .npy file.")
#     print("Loaded data:")

#     DE_data = np.empty((0, 40, 5, 62, 5))
#     PSD_data = np.empty((0, 40, 5, 62, 5))

#     for block_id in range(7):
#         print("block: ", block_id)
#         now_data = loaded_data[block_id]
#         de_block_data = np.empty((0, 5, 62, 5))
#         psd_block_data = np.empty((0, 5, 62, 5))
#         for class_id in tqdm(range(40)):
#             de_class_data = np.empty((0, 62, 5))
#             psd_class_data = np.empty((0, 62, 5))
#             for i in range(5):
#                 de, psd = DE_PSD(now_data[class_id, i, :, :].reshape(62, 2*fre), fre, 2)
#                 de_class_data = np.concatenate((de_class_data, de.reshape(1, 62, 5)))
#                 psd_class_data = np.concatenate((psd_class_data, psd.reshape(1, 62, 5)))
#             de_block_data = np.concatenate((de_block_data, de_class_data.reshape(1, 5, 62, 5)))
#             psd_block_data = np.concatenate((psd_block_data, psd_class_data.reshape(1, 5, 62, 5)))
#         DE_data = np.concatenate((DE_data, de_block_data.reshape(1, 40, 5, 62, 5)))
#         PSD_data = np.concatenate((PSD_data, psd_block_data.reshape(1, 40, 5, 62, 5)))

#     np.save("data/EEG2Video/DE_1per2s/" + subname +".npy", DE_data)
#     np.save("data/EEG2Video/PSD_1per2s/" + subname + ".npy", PSD_data)

# ---------------------------------------------------------------------------------------------------------------
# NEW VERSION
# ---------------------------------------------------------------------------------------------------------------
import numpy as np
import os
import re
from tqdm import tqdm
from DE_PSD import DE_PSD

fre = 200  # Hz

# CONFIG
SEGMENTED_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments/"
SAVE_DE_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_features/DE_1per2s/"
SAVE_PSD_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_features/PSD_1per2s/"

os.makedirs(SAVE_DE_DIR, exist_ok=True)
os.makedirs(SAVE_PSD_DIR, exist_ok=True)

def get_subjects(directory):
    return sorted([f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))])

# -----------------------------
# Ask user which subjects to process
# -----------------------------
all_subjects = get_subjects(SEGMENTED_DIR)
print("Available subjects:", all_subjects)

user_input = input("Enter subject folders to process (comma separated, e.g. sub1,sub2): ")
sub_list = [f.strip() for f in user_input.split(",") if f.strip() in all_subjects]

if not sub_list:
    raise ValueError("No valid subjects selected!")

processed_count = 0

for subname in sub_list:
    subject_dir = os.path.join(SEGMENTED_DIR, subname)
    print(f"\nProcessing subject {subname}")

    for block_name in sorted(os.listdir(subject_dir)):
        block_dir = os.path.join(subject_dir, block_name)
        if not os.path.isdir(block_dir):
            continue

        save_de_block = os.path.join(SAVE_DE_DIR, subname, block_name)
        save_psd_block = os.path.join(SAVE_PSD_DIR, subname, block_name)
        os.makedirs(save_de_block, exist_ok=True)
        os.makedirs(save_psd_block, exist_ok=True)

        clip_files = sorted([f for f in os.listdir(block_dir) if f.endswith(".npy")])
        for clip_file in tqdm(clip_files, desc=f"{subname}/{block_name}"):
            clip_path = os.path.join(block_dir, clip_file)
            eeg_clip = np.load(clip_path)  # shape (62,400)

            de, psd = DE_PSD(eeg_clip, fre, 2)  # (62,5), (62,5)

            base_name = os.path.splitext(clip_file)[0]  # drop .npy
            np.save(os.path.join(save_de_block, base_name + ".npy"), de)
            np.save(os.path.join(save_psd_block, base_name + ".npy"), psd)

    processed_count += 1
    print(f"Finished subject {subname}")

print(f"\nSummary: {processed_count} subjects processed")