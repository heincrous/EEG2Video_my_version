# ==========================================
# EEG2Video Preprocessing Figures Generator
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
win100_path = os.path.join(base_path, "EEG_windows_100")
win200_path = os.path.join(base_path, "EEG_windows_200")

# === Helper: load first subject file ===
def load_first_subject(folder):
    files = sorted(glob(os.path.join(folder, "*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {folder}")
    print(f"Loaded: {os.path.basename(files[0])}")
    return np.load(files[0])

# === Load arrays ===
DE      = load_first_subject(de_path)     # (40,5,62,200)
PSD     = load_first_subject(psd_path)    # (40,5,62,200)
win_100 = load_first_subject(win100_path) # (40,5,7,62,100)
win_200 = load_first_subject(win200_path) # (40,5,3,62,200)

# ==========================================
# 1. DE Feature Distribution – Mean ± SD
# ==========================================
de_mean = DE.mean(axis=(0,1,2,4))   # mean per channel
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
# 3. Temporal Window Verification – 1 s (EEG_windows_100)
# ==========================================
fs = 200  # sampling rate
s, c, t, ch = 0, 0, 0, 0  # subject, concept, trial, channel
windows = win_100[c, t, :, :, :]  # (7,62,100)
time = np.arange(windows.shape[-1]) / fs

plt.figure(figsize=(8,3))
for i, w in enumerate(windows):
    plt.plot(time + i * 0.5, w[ch], lw=0.8)  # select channel ch
plt.title("Temporal Segmentation – 1 s Overlapping Windows")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.tight_layout()
plt.savefig(os.path.join(save_path, "temp_window_verification_1s.png"), dpi=300)
plt.close()


# ==========================================
# 4. Temporal Window Verification – 0.5 s (EEG_windows_200)
# ==========================================
windows = win_200[c, t, :, :, :]  # (3,62,200)
time = np.arange(windows.shape[-1]) / fs

plt.figure(figsize=(8,3))
for i, w in enumerate(windows):
    plt.plot(time + i * 0.25, w[ch], lw=0.8)  # select channel ch
plt.title("Temporal Segmentation – 0.5 s Overlapping Windows")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.tight_layout()
plt.savefig(os.path.join(save_path, "temp_window_verification_0.5s.png"), dpi=300)
plt.close()

print("All figures saved to:", save_path)
