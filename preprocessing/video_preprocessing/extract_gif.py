# # 导入所需要的库
# import cv2
# import imageio
# import numpy as np

# def get_source_info_opencv(source_name):
#     return_value = 0  
#     try:
#         cap = cv2.VideoCapture(source_name)
#         width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
#         height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#         print("width:{} \nheight:{} \nfps:{} \nnum_frames:{}".format(width, height, fps, num_frames))
#     except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
#         print("init_source:{} error. {}\n".format(source_name, str(e)))
#         return_value = -1
#     return return_value

# for video_id in range(4, 5):  

#     video_path = "data/Video/" + str(video_id) + "th_10min.mp4"
      
#     get_source_info_opencv(video_path)
#     # 读取视频文件
#     videoCapture = cv2.VideoCapture(video_path) 

#     is_video = np.zeros(24*(8*60+40))
#     print(is_video.shape)

#     for i in range(40):
#         is_video[i*(24*(13)):i*(24*(13))+3*24] = 0
#         for j in range(5):
#             is_video[i*(24*(13))+3*24+j*24*2:i*(24*(13))+3*24+j*24*2+24*2] = j+1
      
#     #读帧
#     k = 0
#     i = -1
#     while i < 12480:
#         i += 1
#         success, frame = videoCapture.read()
#         frame = frame[..., ::-1] 
#         if(is_video[i] == 0):
#             continue
#         all_frame = [cv2.resize(frame, (512, 288), interpolation=cv2.INTER_LINEAR)]
#         while(i+1<12480 and is_video[i+1] == is_video[i]):
#             i += 1
#             success, frame = videoCapture.read()
#             frame = frame[..., ::-1] 
#             all_frame.append(cv2.resize(frame, (512, 288), interpolation=cv2.INTER_LINEAR))
#         gif_frame = []
#         for j in range(0, 48, 8):
#             gif_frame.append(all_frame[j])    
#         k += 1
#         print("k = ", k, len(gif_frame))
#         imageio.mimsave('data/Video_Gif/Block' + str(video_id) + '/'+str(k)+'.gif', gif_frame, 'GIF', duration=0.33333)

# ---------------------------------------------------------------------------------------------------------------
# NEW VERSION
# ---------------------------------------------------------------------------------------------------------------
import cv2
import imageio
import numpy as np
import os
import re
from gt_label import GT_LABEL   # <-- import the true class order

# CONFIG
RAW_VIDEO_DIR = "/content/drive/MyDrive/EEG2Video_data/raw/Video/"
SAVE_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/Video_Gif/"

os.makedirs(SAVE_DIR, exist_ok=True)

def get_video_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".mp4")]
    return sorted(files, key=lambda x: int(re.findall(r'\d+', x)[0]))

def get_source_info(video_path):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"{video_path} → width:{width}, height:{height}, fps:{fps}, frames:{num_frames}")
    cap.release()

# -----------------------------
# Ask user which videos to process
# -----------------------------
all_videos = get_video_files(RAW_VIDEO_DIR)
print("Available videos:", all_videos)

user_input = input("Enter video files to process (comma separated, e.g. 1st_10min.mp4,2nd_10min.mp4): ")
video_list = [f.strip() for f in user_input.split(",") if f.strip() in all_videos]

if not video_list:
    raise ValueError("No valid videos selected!")

processed_count = 0

BLOCK_MAP = {
    "1st_10min": "Block1",
    "2nd_10min": "Block2",
    "3rd_10min": "Block3",
    "4th_10min": "Block4",
    "5th_10min": "Block5",
    "6th_10min": "Block6",
    "7th_10min": "Block7",
}

for video_file in video_list:
    video_path = os.path.join(RAW_VIDEO_DIR, video_file)

    block_key = os.path.splitext(video_file)[0]  # e.g. "1st_10min"
    block_name = BLOCK_MAP.get(block_key, block_key)
    block_idx = int(block_name.replace("Block", "")) - 1
    block_save_dir = os.path.join(SAVE_DIR, block_name)
    os.makedirs(block_save_dir, exist_ok=True)

    print(f"\nProcessing {video_file}")
    get_source_info(video_path)

    cap = cv2.VideoCapture(video_path)
    is_video = np.zeros(24 * (8 * 60 + 40))  # mark which frames belong to clips

    # clip scheduling: 40 classes × (3s rest + 5 clips × 2s)
    for i in range(40):
        is_video[i * (24 * 13): i * (24 * 13) + 3 * 24] = 0
        for j in range(5):
            is_video[i * (24 * 13) + 3 * 24 + j * 24 * 2:
                     i * (24 * 13) + 3 * 24 + j * 24 * 2 + 24 * 2] = j + 1

    i = -1
    class_pos, clip_idx = 0, 1

    while i < 12480 and class_pos < 40:
        i += 1
        success, frame = cap.read()
        if not success:
            break
        frame = frame[..., ::-1]  # BGR → RGB
        if is_video[i] == 0:
            continue

        # collect consecutive frames for this clip
        all_frames = [cv2.resize(frame, (512, 288), interpolation=cv2.INTER_LINEAR)]
        while i + 1 < 12480 and is_video[i + 1] == is_video[i]:
            i += 1
            success, frame = cap.read()
            if not success:
                break
            all_frames.append(cv2.resize(frame[..., ::-1], (512, 288), interpolation=cv2.INTER_LINEAR))

        # sample frames (every 8th, up to 48 total)
        gif_frames = [all_frames[j] for j in range(0, min(48, len(all_frames)), 8)]

        # use GT_LABEL for correct class ID
        true_class = GT_LABEL[block_idx, class_pos]
        save_name = f"class{true_class:02d}_clip{clip_idx:02d}.gif"
        gif_path = os.path.join(block_save_dir, save_name)
        imageio.mimsave(gif_path, gif_frames, "GIF", duration=0.3333)
        print("Saved", gif_path)

        # update indices
        clip_idx += 1
        if clip_idx > 5:
            clip_idx = 1
            class_pos += 1

    cap.release()
    processed_count += 1
    print(f"Finished {video_file}, saved into {block_save_dir}")

print(f"\nSummary: {processed_count} videos processed")