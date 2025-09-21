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

# CONFIG
RAW_VIDEO_DIR = "/content/drive/MyDrive/EEG2Video_data/raw/Video/"
SAVE_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/Video_Gif/"
LOG_FILE = "/content/drive/MyDrive/EEG2Video_data/processed/processed_log.txt"
PROCESS_TAG = "[VIDEO_CLIP]"
CUTOFF_INDEX = 2   # process only the first N block videos

os.makedirs(SAVE_DIR, exist_ok=True)

def get_files_names_in_directory(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".mp4")]
    return sorted(files, key=lambda x: int(re.findall(r'\d+', x)[0]))

def load_processed_log():
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, "r") as f:
        return set(line.strip() for line in f.readlines())

def update_processed_log(filename):
    entry = f"{PROCESS_TAG} {filename}"
    if entry not in load_processed_log():
        with open(LOG_FILE, "a") as f:
            f.write(entry + "\n")

def get_source_info_opencv(source_name):
    cap = cv2.VideoCapture(source_name)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"width:{width} \nheight:{height} \nfps:{fps} \nnum_frames:{num_frames}")

# Get all videos in numeric order
video_files = get_files_names_in_directory(RAW_VIDEO_DIR)
video_files = video_files[:CUTOFF_INDEX]

processed = load_processed_log()
processed_count = 0
skipped_count = 0

for video_file in video_files:
    entry = f"{PROCESS_TAG} {video_file}"
    if entry in processed:
        print(f"Skipping {video_file} (already processed for {PROCESS_TAG})")
        skipped_count += 1
        continue

    video_path = os.path.join(RAW_VIDEO_DIR, video_file)
    print(f"Processing {video_file}")
    get_source_info_opencv(video_path)

    # Create save folder for this block
    block_name = os.path.splitext(video_file)[0]  # e.g. "1st_10min"
    block_folder = os.path.join(SAVE_DIR, block_name)
    os.makedirs(block_folder, exist_ok=True)

    videoCapture = cv2.VideoCapture(video_path)
    is_video = np.zeros(24*(8*60+40))

    for i in range(40):
        is_video[i*(24*13):i*(24*13)+3*24] = 0
        for j in range(5):
            is_video[i*(24*13)+3*24+j*24*2:i*(24*13)+3*24+j*24*2+24*2] = j+1

    k = 0
    i = -1
    while i < 12480:
        i += 1
        success, frame = videoCapture.read()
        if not success:
            break
        frame = frame[..., ::-1]
        if is_video[i] == 0:
            continue
        all_frame = [cv2.resize(frame, (512, 288), interpolation=cv2.INTER_LINEAR)]
        while i+1 < 12480 and is_video[i+1] == is_video[i]:
            i += 1
            success, frame = videoCapture.read()
            if not success:
                break
            frame = frame[..., ::-1]
            all_frame.append(cv2.resize(frame, (512, 288), interpolation=cv2.INTER_LINEAR))
        gif_frame = []
        for j in range(0, min(48, len(all_frame)), 8):
            gif_frame.append(all_frame[j])
        k += 1
        gif_path = os.path.join(block_folder, f"{k}.gif")
        imageio.mimsave(gif_path, gif_frame, "GIF", duration=0.33333)
        print("Saved", gif_path)

    update_processed_log(video_file)
    processed_count += 1
    print(f"Finished {video_file}, logged under {PROCESS_TAG}")

print(f"\nSummary: {processed_count} processed, {skipped_count} skipped, cutoff={CUTOFF_INDEX}")
