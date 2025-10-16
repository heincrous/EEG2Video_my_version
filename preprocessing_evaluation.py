# ==========================================
# EEG2Video Preprocessing Figures Generator (Shaded Overlap = Grey)
# ==========================================
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# === Paths (Drive already mounted) ===
base_path = "/content/drive/MyDrive/EEG2Video_data/processed"
save_path = "/content/drive/MyDrive/EEG2Video_results/figures"
os.makedirs(save_path, exist_ok=True)

de_path     = os.path.join(base_path, "EEG_DE_1per2s")
psd_path    = os.path.join(base_path, "EEG_PSD_1per2s")
win100_path = os.path.join(base_path, "EEG_windows_100")  # 0.5 s windows (100, overlap 50)
win200_path = os.path.join(base_path, "EEG_windows_200")  # 1.0 s windows (200, overlap 100)

# === Helper: load first subject file ===
def load_first_subject(folder):
    files = sorted(glob(os.path.join(folder, "*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {folder}")
    print(f"Loaded: {os.path.basename(files[0])}")
    return np.load(files[0])

# === Load arrays ===
DE      = load_first_subject(de_path)     
PSD     = load_first_subject(psd_path)    
win_100 = load_first_subject(win100_path) 
win_200 = load_first_subject(win200_path) 

# ==========================================
# 1. DE Feature Distribution – Mean ± SD
# ==========================================
de_mean = DE.mean(axis=(0,1,2,4))
de_std  = DE.std(axis=(0,1,2,4))

plt.figure(figsize=(8,4))
plt.bar(np.arange(1, 63), de_mean, yerr=de_std, capsize=2, color='steelblue')
plt.title("DE Feature Mean ± SD Across Channels")
plt.xlabel("EEG Channel Index")
plt.ylabel("Amplitude")
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(save_path, "feat_distribution_DE.png"), dpi=300)
plt.close()

# ==========================================
# 2. PSD Feature Distribution – Mean ± SD
# ==========================================
psd_mean = PSD.mean(axis=(0,1,2,4))
psd_std  = PSD.std(axis=(0,1,2,4))

plt.figure(figsize=(8,4))
plt.bar(np.arange(1, 63), psd_mean, yerr=psd_std, capsize=2, color='mediumseagreen')
plt.title("PSD Feature Mean ± SD Across Channels")
plt.xlabel("EEG Channel Index")
plt.ylabel("Power Spectral Density (a.u.)")
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(save_path, "feat_distribution_PSD.png"), dpi=300)
plt.close()

# ==========================================
# 3. Temporal Segmentation (Shaded Overlap = Grey) – 0.5 s Windows
# ==========================================
fs = 200
block, cls, clip, ch = 0, 0, 0, 0
windows = win_100[block, cls, clip, :, ch, :]  # (7,100)
win_len = windows.shape[-1]
step = win_len // 2  # 50 overlap → 50 samples

# reconstruct continuous signal (2 s total)
duration = (win_len + (windows.shape[0]-1)*step) / fs
t = np.linspace(0, duration, win_len + (windows.shape[0]-1)*step)
signal = np.zeros_like(t)
count = np.zeros_like(t)
for i, w in enumerate(windows):
    start = i * step
    signal[start:start+win_len] += w
    count[start:start+win_len] += 1
signal /= np.maximum(count, 1)

plt.figure(figsize=(8,3))
plt.plot(t, signal, color='black', lw=1)

# shaded areas: base windows (blue) + overlap regions (grey)
for i in range(windows.shape[0]):
    start = i * step / fs
    end = start + win_len / fs
    plt.axvspan(start, end, color='lightblue', alpha=0.3)
    if i > 0:
        overlap_start = start
        overlap_end = start + (win_len - step) / fs
        plt.axvspan(overlap_start, overlap_end, color='grey', alpha=0.25)

plt.title("0.5 s Window Segmentation – 50% Overlap Highlighted (Grey)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.tight_layout()
plt.savefig(os.path.join(save_path, "temp_window_shaded_overlap_0.5s.png"), dpi=300)
plt.close()

# ==========================================
# 4. Temporal Segmentation (Shaded Overlap = Grey) – 1 s Windows
# ==========================================
windows = win_200[block, cls, clip, :, ch, :]  # (3,200)
win_len = windows.shape[-1]
step = win_len // 2  # 100 samples

duration = (win_len + (windows.shape[0]-1)*step) / fs
t = np.linspace(0, duration, win_len + (windows.shape[0]-1)*step)
signal = np.zeros_like(t)
count = np.zeros_like(t)
for i, w in enumerate(windows):
    start = i * step
    signal[start:start+win_len] += w
    count[start:start+win_len] += 1
signal /= np.maximum(count, 1)

plt.figure(figsize=(8,3))
plt.plot(t, signal, color='black', lw=1)

for i in range(windows.shape[0]):
    start = i * step / fs
    end = start + win_len / fs
    plt.axvspan(start, end, color='lightcoral', alpha=0.3)
    if i > 0:
        overlap_start = start
        overlap_end = start + (win_len - step) / fs
        plt.axvspan(overlap_start, overlap_end, color='grey', alpha=0.25)

plt.title("1 s Window Segmentation – 50% Overlap Highlighted (Grey)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.tight_layout()
plt.savefig(os.path.join(save_path, "temp_window_shaded_overlap_1s.png"), dpi=300)
plt.close()

print("All figures saved to:", save_path)
