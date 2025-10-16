# ==========================================
# EEG2Video Preprocessing Figures Generator (Final — Clear Overlap Version)
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
plt.bar(np.arange(1,63), de_mean, yerr=de_std, capsize=2, color='steelblue')
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
plt.bar(np.arange(1,63), psd_mean, yerr=psd_std, capsize=2, color='mediumseagreen')
plt.title("PSD Feature Mean ± SD Across Channels")
plt.xlabel("EEG Channel Index")
plt.ylabel("Power Spectral Density (a.u.)")
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(save_path, "feat_distribution_PSD.png"), dpi=300)
plt.close()

# ==========================================
# 3. Accurate Window Visualization – 0.5 s (EEG_windows_100)
# ==========================================
fs = 200
block, cls, clip, ch = 0, 0, 0, 0
win_len, step = 100, 50
windows = win_100[block, cls, clip, :, ch, :]
colors = plt.cm.tab10(np.linspace(0, 1, windows.shape[0]))

plt.figure(figsize=(10, 4))
for i, w in enumerate(windows):
    start = i * step / fs
    t = np.arange(start, start + win_len / fs, 1 / fs)
    plt.plot(t, w + i * 8e-5, color=colors[i], lw=0.9, label=f"Win {i+1}")

plt.title("EEG Windows (100 samples, 50 overlap)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude + offset (µV)")
plt.legend(fontsize=7, ncol=7)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_path, "accurate_window_overlap_0.5s.png"), dpi=300)
plt.close()

# ==========================================
# 4. Accurate Window Visualization – 1 s (EEG_windows_200)
# ==========================================
win_len, step = 200, 100
windows = win_200[block, cls, clip, :, ch, :]
colors = plt.cm.tab10(np.linspace(0, 1, windows.shape[0]))

plt.figure(figsize=(10, 4))
for i, w in enumerate(windows):
    start = i * step / fs
    t = np.arange(start, start + win_len / fs, 1 / fs)
    plt.plot(t, w + i * 8e-5, color=colors[i], lw=0.9, label=f"Win {i+1}")

plt.title("EEG Windows (200 samples, 100 overlap)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude + offset (µV)")
plt.legend(fontsize=8, ncol=3)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_path, "accurate_window_overlap_1s.png"), dpi=300)
plt.close()

print("All figures saved to:", save_path)
