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

# ---------------------------------------------------------------------------------------------------------------
# NEW VERSION
# ---------------------------------------------------------------------------------------------------------------
import numpy as np
import os
import re
from tqdm import tqdm

fre = 200  # sampling frequency

# CONFIG
RAW_DIR = "/content/drive/MyDrive/EEG2Video_data/raw/EEG/"
SAVE_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/EEG_segments/"

os.makedirs(SAVE_DIR, exist_ok=True)

def get_files_names_in_directory(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".npy")]
    return sorted(files, key=lambda x: int(re.findall(r'\d+', x)[0]))

# -----------------------------
# Ask user what to process
# -----------------------------
all_subjects = get_files_names_in_directory(RAW_DIR)
print("Available subject files:", all_subjects)

user_input = input("Enter subject files to process (comma separated, e.g. sub1.npy,sub2.npy): ")
sub_list = [f.strip() for f in user_input.split(",") if f.strip() in all_subjects]

if not sub_list:
    raise ValueError("No valid subjects selected!")

processed_count = 0

for subname in sub_list:
    print(f"\nProcessing {subname} ...")
    npydata = np.load(os.path.join(RAW_DIR, subname))  # shape: (7, 62, 104000)

    # Create subject folder
    subject_save_dir = os.path.join(SAVE_DIR, subname.replace(".npy", ""))
    os.makedirs(subject_save_dir, exist_ok=True)

    for block_id in range(7):
        block_name = f"Block{block_id+1}"
        block_save_dir = os.path.join(subject_save_dir, block_name)
        os.makedirs(block_save_dir, exist_ok=True)

        now_data = npydata[block_id]  # (62, 104000)
        l = 0

        for class_id in tqdm(range(40), desc=f"{subname} {block_name}"):
            l += (3 * fre)
            for clip_id in range(5):
                clip_data = now_data[:, l:l+2*fre]  # (62, 400)
                l += (2 * fre)

                save_name = f"class{class_id+1:02d}_clip{clip_id+1:02d}.npy"
                save_path = os.path.join(block_save_dir, save_name)
                np.save(save_path, clip_data)

    processed_count += 1
    print(f"Finished {subname}, saved into {subject_save_dir}")

print(f"\nSummary: {processed_count} subjects processed")