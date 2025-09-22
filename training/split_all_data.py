import os
import shutil
from tqdm import tqdm

base_dir = "/content/drive/MyDrive/EEG2Video_data/processed/"

# ---------------- Helpers ----------------
def get_split(clip_id):
    return "train" if clip_id < 4 else "test"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def safe_link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)

def collect_files(in_dir, exts=(".npy", ".mp4", ".gif", ".txt")):
    files = []
    for root, _, fnames in os.walk(in_dir):
        for f in fnames:
            if f.endswith(exts):
                rel = os.path.relpath(os.path.join(root, f), in_dir)
                files.append(rel)
    return sorted(files)

def parse_clip_id(fname):
    return int(fname.split("_")[-1].replace("clip", "").split(".")[0]) - 1

def write_list(path, items):
    with open(path, "w") as f:
        f.write("\n".join(items))

# =========================================================
# 1. Video_latents (anchor, no duplication)
# =========================================================
video_latents_dir = os.path.join(base_dir, "Video_latents")
video_latents_files = collect_files(video_latents_dir, exts=(".npy",))

latents_train, latents_test = [], []
for rel in video_latents_files:
    clip_id = parse_clip_id(rel)
    (latents_train if get_split(clip_id)=="train" else latents_test).append(rel)

write_list(os.path.join(video_latents_dir, "train_list.txt"), latents_train)
write_list(os.path.join(video_latents_dir, "test_list.txt"), latents_test)
print(f"[Video_latents] train={len(latents_train)} test={len(latents_test)} (no dup)")

# =========================================================
# 2. EEG_features (anchor for embeddings, subject-dependent)
# =========================================================
eeg_features_dir = os.path.join(base_dir, "EEG_features")
eeg_features_files = collect_files(eeg_features_dir, exts=(".npy",))

features_train, features_test = [], []
for rel in eeg_features_files:
    clip_id = parse_clip_id(rel)
    (features_train if get_split(clip_id)=="train" else features_test).append(rel)

write_list(os.path.join(eeg_features_dir, "train_list.txt"), features_train)
write_list(os.path.join(eeg_features_dir, "test_list.txt"), features_test)
print(f"[EEG_features] train={len(features_train)} test={len(features_test)}")

# =========================================================
# 3. EEG_windows (aligned with Video_latents, DUPLICATED)
# =========================================================
eeg_windows_dir = os.path.join(base_dir, "EEG_windows")
eeg_windows_files = collect_files(eeg_windows_dir, exts=(".npy",))

windows_train, windows_test = [], []
video_latents_train_dup, video_latents_test_dup = [], []

for rel in eeg_windows_files:
    parts = rel.split(os.sep)
    no_sub = os.sep.join(parts[1:])  # drop subject prefix
    if no_sub in latents_train:
        windows_train.append(rel)
        video_latents_train_dup.append(no_sub)
    elif no_sub in latents_test:
        windows_test.append(rel)
        video_latents_test_dup.append(no_sub)

write_list(os.path.join(eeg_windows_dir, "train_list.txt"), windows_train)
write_list(os.path.join(eeg_windows_dir, "test_list.txt"), windows_test)

# save duplicated video lists separately
write_list(os.path.join(video_latents_dir, "train_list_dup.txt"), video_latents_train_dup)
write_list(os.path.join(video_latents_dir, "test_list_dup.txt"), video_latents_test_dup)

print(f"[EEG_windows] train={len(windows_train)} test={len(windows_test)}")
print(f"[Video_latents DUP] train={len(video_latents_train_dup)} test={len(video_latents_test_dup)}")

# =========================================================
# 4. BLIP_text (aligned with Video_latents, no duplication)
# =========================================================
blip_text_dir = os.path.join(base_dir, "BLIP_text")
blip_files = collect_files(blip_text_dir, exts=(".txt",))

text_train, text_test = [], []
for v in latents_train:
    txt_path = v.replace(".npy", ".txt")
    if txt_path in blip_files:
        text_train.append(txt_path)
for v in latents_test:
    txt_path = v.replace(".npy", ".txt")
    if txt_path in blip_files:
        text_test.append(txt_path)

write_list(os.path.join(blip_text_dir, "train_list.txt"), text_train)
write_list(os.path.join(blip_text_dir, "test_list.txt"), text_test)
print(f"[BLIP_text] train={len(text_train)} test={len(text_test)}")

# =========================================================
# 5. BLIP_embeddings (aligned with EEG_features, DUPLICATED)
# =========================================================
blip_embed_dir = os.path.join(base_dir, "BLIP_embeddings")
embed_files = collect_files(blip_embed_dir, exts=(".npy",))

embed_train, embed_test = [], []
for f in features_train:
    no_sub = "/".join(f.split(os.sep)[1:])
    if no_sub in embed_files:
        embed_train.append(no_sub)
for f in features_test:
    no_sub = "/".join(f.split(os.sep)[1:])
    if no_sub in embed_files:
        embed_test.append(no_sub)

write_list(os.path.join(blip_embed_dir, "train_list_dup.txt"), embed_train)
write_list(os.path.join(blip_embed_dir, "test_list_dup.txt"), embed_test)
print(f"[BLIP_embeddings DUP] train={len(embed_train)} test={len(embed_test)}")

# =========================================================
# 6. EEG_segments (independent)
# =========================================================
eeg_segments_dir = os.path.join(base_dir, "EEG_segments")
seg_files = collect_files(eeg_segments_dir, exts=(".npy",))

seg_train, seg_test = [], []
for rel in seg_files:
    clip_id = parse_clip_id(rel)
    (seg_train if get_split(clip_id)=="train" else seg_test).append(rel)

write_list(os.path.join(eeg_segments_dir, "train_list.txt"), seg_train)
write_list(os.path.join(eeg_segments_dir, "test_list.txt"), seg_test)
print(f"[EEG_segments] train={len(seg_train)} test={len(seg_test)}")

# =========================================================
# 7. Video_mp4 & Video_gif (independent)
# =========================================================
for mod, ext in [("Video_mp4", ".mp4"), ("Video_gif", ".gif")]:
    in_dir = os.path.join(base_dir, mod)
    files = collect_files(in_dir, exts=(ext,))
    train, test = [], []
    for rel in files:
        clip_id = parse_clip_id(rel)
        (train if get_split(clip_id)=="train" else test).append(rel)
    write_list(os.path.join(in_dir, "train_list.txt"), train)
    write_list(os.path.join(in_dir, "test_list.txt"), test)
    print(f"[{mod}] train={len(train)} test={len(test)}")

# =========================================================
# VALIDATION
# =========================================================
def normalize(p, modality):
    parts = p.split("/")
    if modality.startswith("EEG"):
        return "/".join(parts[1:])  # drop subX
    else:
        return "/".join(parts)

def validate(master_lists, split="train"):
    lists = {}
    for mod, path in master_lists.items():
        with open(path) as f:
            lists[mod] = [normalize(l.strip(), mod) for l in f]

    lengths = {k: len(v) for k,v in lists.items()}
    print(f"\n[{split.upper()}] lengths:", lengths)

    if len(set(lengths.values())) != 1:
        print("  WARNING: lengths not equal")

    ref_name, ref_list = next(iter(lists.items()))
    for name, cur_list in lists.items():
        if name == ref_name: continue
        mism = [(i,a,b) for i,(a,b) in enumerate(zip(ref_list, cur_list)) if a!=b]
        if mism:
            print(f"  {name} mismatch vs {ref_name}: {len(mism)}")
            for i,a,b in mism[:5]:
                print(f"    idx {i}: REF={a} OTHER={b}")
        else:
            print(f"  {name} aligned with {ref_name}")

# sanity checks
validate({
    "EEG_windows": os.path.join(base_dir,"EEG_windows/train_list.txt"),
    "Video_latents DUP": os.path.join(base_dir,"Video_latents/train_list_dup.txt"),
}, split="train")

validate({
    "EEG_windows": os.path.join(base_dir,"EEG_windows/test_list.txt"),
    "Video_latents DUP": os.path.join(base_dir,"Video_latents/test_list_dup.txt"),
}, split="test")

validate({
    "EEG_features": os.path.join(base_dir,"EEG_features/train_list.txt"),
    "BLIP_embeddings DUP": os.path.join(base_dir,"BLIP_embeddings/train_list_dup.txt"),
}, split="train")

validate({
    "EEG_features": os.path.join(base_dir,"EEG_features/test_list.txt"),
    "BLIP_embeddings DUP": os.path.join(base_dir,"BLIP_embeddings/test_list_dup.txt"),
}, split="test")
