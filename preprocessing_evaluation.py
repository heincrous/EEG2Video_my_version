# ==========================================
# EEG2Video Preprocessing Figures Generator (Shaded Overlap Visualization)
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
win100_path = os.path.join(base_path, "EEG_windows_100")  # 0.5 s windows
win200_path = os.path.join(base_path, "EEG_windows_200")  # 1.0 s windows

# === Helper: load first subject file ===
def load_first_subject(folder):
    files = sorted(glob(os.path.join(folder, "*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {folder}")
    print(f"Loaded: {os.path.basename(files[0])}")
    return np.load(files[0])

# === Load arrays ===
DE      = load_first_subject(de_path)     # (7,40,5,62,200)
PSD     = load_first_subject(psd_path)    # (7,40,5,62,200)
win_100 = load_first_subject(win100_path) # (7,40,5,7,62,200) -> 0.5 s windows
win_200 = load_first_subject(win200_path) # (7,40,5,3,62,200) -> 1.0 s windows

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
# 3. Temporal Segmentation (Shaded) – 0.5 s Windows (EEG_windows_100)
# ==========================================
fs = 200
block, cls, clip, ch = 0, 0, 0, 0
windows = win_100[block, cls, clip, :, ch, :]  # (7,100)
win_len = windows.shape[-1]
step = win_len // 2  # 50% overlap -> 50 samples = 0.25 s
duration = (win_len + (windows.shape[0]-1)*step) / fs

# build continuous composite signal for visualization
t_signal = np.linspace(0, duration, win_len + (windows.shape[0]-1)*step)
continuous = np.zeros(len(t_signal))
count = np.zeros(len(t_signal))
for i, w in enumerate(windows):
    start = i * step
    continuous[start:start+win_len] += w
    count[start:start+win_len] += 1
continuous /= np.maximum(count, 1)

plt.figure(figsize=(8,3))
plt.plot(t_signal, continuous, color='black', lw=1)
for i in range(windows.shape[0]):
    start = i * step / fs
    end = start + win_len / fs
    plt.axvspan(start, end, color=plt.cm.tab10(i/float(windows.shape[0])), alpha=0.2)
plt.title("0.5 s Window Segmentation with 50% Overlap (Shaded View)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.tight_layout()
plt.savefig(os.path.join(save_path, "temp_window_shaded_0.5s.png"), dpi=300)
plt.close()

# ==========================================
# 4. Temporal Segmentation (Shaded) – 1 s Windows (EEG_windows_200)
# ==========================================
windows = win_200[block, cls, clip, :, ch, :]  # (3,200)
win_len = windows.shape[-1]
step = win_len // 2  # 50% overlap -> 100 samples = 0.5 s
duration = (win_len + (windows.shape[0]-1)*step) / fs

t_signal = np.linspace(0, duration, win_len + (windows.shape[0]-1)*step)
continuous = np.zeros(len(t_signal))
count = np.zeros(len(t_signal))
for i, w in enumerate(windows):
    start = i * step
    continuous[start:start+win_len] += w
    count[start:start+win_len] += 1
continuous /= np.maximum(count, 1)

plt.figure(figsize=(8,3))
plt.plot(t_signal, continuous, color='black', lw=1)
for i in range(windows.shape[0]):
    start = i * step / fs
    end = start + win_len / fs
    plt.axvspan(start, end, color=plt.cm.tab10(i/float(windows.shape[0])), alpha=0.2)
plt.title("1 s Window Segmentation with 50% Overlap (Shaded View)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.tight_layout()
plt.savefig(os.path.join(save_path, "temp_window_shaded_1s.png"), dpi=300)
plt.close()

print("All figures saved to:", save_path)
