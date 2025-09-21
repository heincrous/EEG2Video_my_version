import os
import re
import torch
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
from gt_label import GT_LABEL

# CONFIG
RAW_BLIP_DIR = "/content/drive/MyDrive/EEG2Video_data/raw/BLIP-caption/"
EMB_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_embeddings/"
CAP_DIR = "/content/drive/MyDrive/EEG2Video_data/processed/BLIP_captions/"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP text encoder
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
model.eval()

os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(CAP_DIR, exist_ok=True)

def get_blip_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".txt")]
    return sorted(files, key=lambda x: int(re.findall(r'\d+', x)[0]))

BLOCK_MAP = {
    "1st_10min": "Block1",
    "2nd_10min": "Block2",
    "3rd_10min": "Block3",
    "4th_10min": "Block4",
    "5th_10min": "Block5",
    "6th_10min": "Block6",
    "7th_10min": "Block7",
}

all_blip_files = get_blip_files(RAW_BLIP_DIR)
print("Available BLIP caption files:", all_blip_files)

user_input = input("Enter BLIP caption files to process (comma separated, e.g. 1st_10min.txt,2nd_10min.txt): ")
blip_list = [f.strip() for f in user_input.split(",") if f.strip() in all_blip_files]

if not blip_list:
    raise ValueError("No valid BLIP caption files selected!")

processed_count = 0

for blip_file in blip_list:
    block_key = os.path.splitext(blip_file)[0]
    block_name = BLOCK_MAP.get(block_key, block_key)
    block_idx = int(block_name.replace("Block", "")) - 1

    block_emb_dir = os.path.join(EMB_DIR, block_name)
    block_cap_dir = os.path.join(CAP_DIR, block_name)
    os.makedirs(block_emb_dir, exist_ok=True)
    os.makedirs(block_cap_dir, exist_ok=True)

    print(f"\nProcessing {blip_file} -> {block_name}")

    with open(os.path.join(RAW_BLIP_DIR, blip_file), "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) != 200:
        print(f"Warning: {blip_file} has {len(lines)} captions (expected 200)")

    for class_pos in range(40):
        true_class = GT_LABEL[block_idx, class_pos]
        for clip_id in range(5):
            idx = class_pos * 5 + clip_id
            if idx >= len(lines):
                continue
            caption = lines[idx]

            with torch.no_grad():
                inputs = processor(text=[caption], return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(device)
                # Hidden states from the text encoder, shape (1,77,768)
                text_outputs = model.text_encoder(**inputs, output_hidden_states=True)
                embedding = text_outputs.last_hidden_state.squeeze(0).cpu().numpy()  # (77,768)

            base_name = f"class{true_class:02d}_clip{clip_id+1:02d}"

            np.save(os.path.join(block_emb_dir, base_name + ".npy"), embedding)

            with open(os.path.join(block_cap_dir, base_name + ".txt"), "w") as ftxt:
                ftxt.write(caption)

            processed_count += 1

    print(f"Finished {blip_file}, saved {processed_count} embeddings so far")

print(f"\nSummary: {processed_count} captions embedded")
