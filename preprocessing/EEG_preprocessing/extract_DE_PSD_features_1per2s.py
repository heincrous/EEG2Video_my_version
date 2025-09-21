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

fre = 200

# CONFIG
SEGMENTED_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments/"
SAVE_DE_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_features/DE_1per2s/"
SAVE_PSD_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_features/PSD_1per2s/"
LOG_FILE = "/content/drive/MyDrive/EEG2Video_data/processed/processed_log.txt"
PROCESS_TAG = "[DE_PSD]"
DEPENDENCY_TAG = "[SEGMENT]"   # only process files that were segmented

os.makedirs(SAVE_DE_DIR, exist_ok=True)
os.makedirs(SAVE_PSD_DIR, exist_ok=True)

def get_files_names_in_directory(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".npy")]
    return sorted(files, key=lambda x: int(re.findall(r'\d+', x)[0]))

def load_processed_log():
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    return set(lines)

def update_processed_log(filename):
    entry = f"{PROCESS_TAG} {filename}"
    if entry not in load_processed_log():
        with open(LOG_FILE, "a") as f:
            f.write(entry + "\n")

# Load segmented files
segmented_files = get_files_names_in_directory(SEGMENTED_DIR)

# Load log
processed = load_processed_log()

processed_count = 0
skipped_count = 0

for subname in segmented_files:
    dep_entry = f"{DEPENDENCY_TAG} {subname}"
    my_entry = f"{PROCESS_TAG} {subname}"

    # Only process if segmentation is done and DE/PSD not yet done
    if dep_entry not in processed:
        print(f"Skipping {subname} (segmentation not found in log)")
        skipped_count += 1
        continue
    if my_entry in processed:
        print(f"Skipping {subname} (already processed for {PROCESS_TAG})")
        skipped_count += 1
        continue

    print(f"Processing {subname}")
    loaded_data = np.load(os.path.join(SEGMENTED_DIR, subname))

    DE_data = np.empty((0, 40, 5, 62, 5))
    PSD_data = np.empty((0, 40, 5, 62, 5))

    for block_id in range(7):
        print(" block:", block_id)
        now_data = loaded_data[block_id]
        de_block_data = np.empty((0, 5, 62, 5))
        psd_block_data = np.empty((0, 5, 62, 5))
        for class_id in tqdm(range(40), desc="Classes"):
            de_class_data = np.empty((0, 62, 5))
            psd_class_data = np.empty((0, 62, 5))
            for i in range(5):
                de, psd = DE_PSD(now_data[class_id, i, :, :].reshape(62, 2*fre), fre, 2)
                de_class_data = np.concatenate((de_class_data, de.reshape(1, 62, 5)))
                psd_class_data = np.concatenate((psd_class_data, psd.reshape(1, 62, 5)))
            de_block_data = np.concatenate((de_block_data, de_class_data.reshape(1, 5, 62, 5)))
            psd_block_data = np.concatenate((psd_block_data, psd_class_data.reshape(1, 5, 62, 5)))
        DE_data = np.concatenate((DE_data, de_block_data.reshape(1, 40, 5, 62, 5)))
        PSD_data = np.concatenate((PSD_data, psd_block_data.reshape(1, 40, 5, 62, 5)))

    np.save(os.path.join(SAVE_DE_DIR, subname), DE_data)
    np.save(os.path.join(SAVE_PSD_DIR, subname), PSD_data)
    update_processed_log(subname)
    processed_count += 1
    print(f"Finished {subname}, saved DE/PSD and logged under {PROCESS_TAG}")

print(f"\nSummary: {processed_count} processed, {skipped_count} skipped")