# Dataset File Structure

# BLIP-caption/
#     1st_10min.txt
#     2nd_10min.txt
#     3rd_10min.txt
#     4th_10min.txt
#     5th_10min.txt
#     6th_10min.txt
#     7th_10min.txt
#
# EEG/
#     sub1.npy
#     sub1_session2.npy
#     sub2.npy
#     sub3.npy
#     sub4.npy
#     sub5.npy
#     ... (21 files total)
#
# SEED-DV/
#     channel-order.xlsx
#     channel_62_pos.locs
#
# Video/
#     1st_10min.mp4
#     2nd_10min.mp4
#     3rd_10min.mp4
#     4th_10min.mp4
#     5th_10min.mp4
#     6th_10min.mp4
#     7th_10min.mp4
#     readme.txt
#
# meta-info/
#     All_video_color.npy
#     All_video_face_apperance.npy
#     All_video_human_apperance.npy
#     All_video_label.npy
#     All_video_obj_number.npy
#     All_video_optical_flow_score.npy

import os

# Path to your dataset (adapt if needed)
base_path = "/content/drive/MyDrive/EEG2Video_data/raw"

# Walk through the directory and preview structure
for root, dirs, files in os.walk(base_path):
    level = root.replace(base_path, '').count(os.sep)
    indent = ' ' * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 4 * (level + 1)
    for f in files[:10]:  # preview max 10 files per folder
        print(f"{subindent}{f}")
    if len(files) > 10:
        print(f"{subindent}... ({len(files)} files total)")