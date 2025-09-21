# import numpy as np
# import os
# from tqdm import tqdm

# # segment a EEG data numpy array with the shape of (7 * 62 * 520s*fre) into 2-sec EEG segments
# # segment it into a new array with the shape of (7 * 40 * 5 * 62 * 2s*fre), 
# # meaning 7 blocks, 40 concepts, 5 video clips, 62 channels, and 2s*fre time-points.

# fre = 200

# def get_files_names_in_directory(directory):
#     files_names = []
#     for root, _, filenames in os.walk(directory):
#         for filename in filenames:
#             files_names.append(filename)
#     return files_names

# sub_list = get_files_names_in_directory("./data/Rawf_200Hz/")

# for subname in sub_list:
#     npydata = np.load('./data/Rawf_200Hz/' + subname)

#     save_data = np.empty((0, 40, 5, 62, 2*fre))

#     for block_id in range(7):
#         print("block: ", block_id)
#         now_data = npydata[block_id]
#         l = 0
#         block_data = np.empty((0, 5, 62, 2*fre))
#         for class_id in tqdm(range(40)):
#             l += (3 * fre)
#             class_data = np.empty((0, 62, 2*fre))
#             for i in range(5):
#                 class_data = np.concatenate((class_data, now_data[:, l : l + 2*fre].reshape(1, 62, 2*fre)))
#                 l += (2 * fre)
#             block_data = np.concatenate((block_data, class_data.reshape(1, 5, 62, 2*fre)))
#         save_data = np.concatenate((save_data, block_data.reshape(1, 40, 5, 62, 2*fre)))

#     np.save('./data/Segmented_Rawf_200Hz_2s/' + subname, save_data)

# NEW VERSION
import numpy as np
import os
from tqdm import tqdm

fre = 200

# CONFIG
RAW_DIR = "/content/drive/MyDrive/EEG2Video_data/raw/EEG/"
SAVE_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments/"
LOG_FILE = "/content/drive/MyDrive/EEG2Video_data/processed/processed_log.txt"
PROCESS_TAG = "[SEGMENT]"   # change to [DE_PSD], [PSD], etc. in other scripts
CUTOFF_INDEX = 3            # only process files up to this index

os.makedirs(SAVE_DIR, exist_ok=True)

def get_files_names_in_directory(directory):
    return sorted([f for f in os.listdir(directory) if f.endswith(".npy")])

def load_processed_log():
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    return set(line for line in lines if line.startswith(PROCESS_TAG))

def update_processed_log(filename):
    with open(LOG_FILE, "a") as f:
        f.write(f"{PROCESS_TAG} {filename}\n")

# Load all subject files
sub_list = get_files_names_in_directory(RAW_DIR)
sub_list = sub_list[:CUTOFF_INDEX]

# Load already processed list for this PROCESS_TAG
processed = load_processed_log()

for subname in sub_list:
    if f"{PROCESS_TAG} {subname}" in processed:
        print(f"Skipping {subname} (already processed for {PROCESS_TAG})")
        continue

    print(f"Processing {subname}")
    npydata = np.load(os.path.join(RAW_DIR, subname))

    save_data = np.empty((0, 40, 5, 62, 2*fre))

    for block_id in range(7):
        print(" block:", block_id)
        now_data = npydata[block_id]
        l = 0
        block_data = np.empty((0, 5, 62, 2*fre))
        for class_id in tqdm(range(40), desc="Classes"):
            l += (3 * fre)
            class_data = np.empty((0, 62, 2*fre))
            for i in range(5):
                class_data = np.concatenate(
                    (class_data, now_data[:, l:l+2*fre].reshape(1, 62, 2*fre))
                )
                l += (2 * fre)
            block_data = np.concatenate((block_data, class_data.reshape(1, 5, 62, 2*fre)))
        save_data = np.concatenate((save_data, block_data.reshape(1, 40, 5, 62, 2*fre)))

    np.save(os.path.join(SAVE_DIR, subname), save_data)
    update_processed_log(subname)
    print(f"Finished {subname}, saved and logged under {PROCESS_TAG}")