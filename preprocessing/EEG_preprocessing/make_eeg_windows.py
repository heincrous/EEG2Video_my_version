# ==========================================
# CREATE EEG WINDOWS FOR SEQ2SEQ INPUT
# ==========================================
# Input:
#   EEG2Video_data/processed/EEG_segments/subX.npy
#   Shape = [7, 40, 5, 62, 400]
#
# Process:
#   - Apply sliding windows:
#       Type A: window=200, overlap=100 → [7, 62, 200]
#       Type B: window=100, overlap=50  → [7, 62, 100]
#   - Each 2 s clip [62, 400] → windows [num, 62, win_len]
#   - Preserve subject-level structure
#
# Output:
#   EEG2Video_data/processed/EEG_windows_200/subX.npy
#       Shape = [7, 40, 5, 7, 62, 200]
#   EEG2Video_data/processed/EEG_windows_100/subX.npy
#       Shape = [7, 40, 5, 7, 62, 100]
# ==========================================

# === Standard libraries ===
import os

# === Third-party libraries ===
import numpy as np
from tqdm import tqdm


# ==========================================
# CONFIGURATION (EDITABLE PARAMETERS)
# ==========================================
config = {
    "drive_root": "/content/drive/MyDrive/EEG2Video_data",
    "segment_len": 400,    # samples (2 s @ 200 Hz)

    # Window type A (1 s @ 200 Hz)
    "winA_size": 200,
    "winA_overlap": 100,

    # Window type B (0.5 s @ 200 Hz)
    "winB_size": 100,
    "winB_overlap": 50,
}

# Derived values
config["winA_step"] = config["winA_size"] - config["winA_overlap"]
config["winB_step"] = config["winB_size"] - config["winB_overlap"]

config["winA_num"] = (config["segment_len"] - config["winA_size"]) // config["winA_step"] + 1
config["winB_num"] = (config["segment_len"] - config["winB_size"]) // config["winB_step"] + 1

# Paths
in_dir   = os.path.join(config["drive_root"], "processed", "EEG_segments")
outA_dir = os.path.join(config["drive_root"], "processed", "EEG_windows_200")
outB_dir = os.path.join(config["drive_root"], "processed", "EEG_windows_100")
os.makedirs(outA_dir, exist_ok=True)
os.makedirs(outB_dir, exist_ok=True)


# ==========================================
# Helper: windowing function
# ==========================================
def make_windows(seg, win_size, step):
    """
    seg: [62, 400]
    return: [num, 62, win_size]
    """
    windows = []
    for start in range(0, config["segment_len"] - win_size + 1, step):
        win = seg[:, start:start+win_size]  # [62, win_size]
        windows.append(win)
    return np.stack(windows, axis=0)


# ==========================================
# Main processing loop
# ==========================================
subjects = [f for f in sorted(os.listdir(in_dir)) if f.endswith(".npy")]

print("Available subject files:")
for i, subj in enumerate(subjects):
    print(f"[{i}] {subj}")

chosen = input("Enter indices of subjects to process (comma separated, 'all' for all): ").strip()
if chosen.lower() == "all":
    selected_idxs = list(range(len(subjects)))
else:
    selected_idxs = [int(c.strip()) for c in chosen.split(",") if c.strip().isdigit()]

for idx in selected_idxs:
    subj_file = subjects[idx]
    subj_name = subj_file.replace(".npy", "")
    subj_path = os.path.join(in_dir, subj_file)

    print(f"\nProcessing {subj_name}...")

    data = np.load(subj_path)  # [7, 40, 5, 62, 400]

    outA = np.zeros((7, 40, 5, config["winA_num"], 62, config["winA_size"]), dtype=np.float32)
    outB = np.zeros((7, 40, 5, config["winB_num"], 62, config["winB_size"]), dtype=np.float32)

    for b in range(7):
        for c in tqdm(range(40), desc=f"{subj_name} Block {b+1}"):
            for k in range(5):
                seg = data[b, c, k]  # [62, 400]

                windowsA = make_windows(seg, config["winA_size"], config["winA_step"])
                windowsB = make_windows(seg, config["winB_size"], config["winB_step"])

                outA[b, c, k] = windowsA
                outB[b, c, k] = windowsB

    np.save(os.path.join(outA_dir, f"{subj_name}.npy"), outA)
    np.save(os.path.join(outB_dir, f"{subj_name}.npy"), outB)

    print(f"Saved Type A (1 s windows) → {outA.shape}")
    print(f"Saved Type B (0.5 s windows) → {outB.shape}")

print("\nProcessing complete.")
