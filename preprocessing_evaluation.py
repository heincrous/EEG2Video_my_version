# ==========================================
# EEG2Video Preprocessing Figures Generator
# ==========================================
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# === Mount Google Drive (if not already mounted) ===
from google.colab import drive
drive.mount('/content/drive')

# === Define root paths ===
base_path   = "/content/drive/MyDrive/EEG2Video_data/processed"
save_path   = "/content/drive/MyDrive/EEG2Video_results/figures"

# create output directory if it does not exist
os.makedirs(save_path, exist_ok=True)

de_path     = os.path.join(base_path, "EEG_DE_1per2s")
psd_path    = os.path.join(base_path, "EEG_PSD_1per2s")
win100_path = os.path.join(base_path, "EEG_windows_100")
win200_path = os.path.join(base_path, "EEG_windows_200")

# === Helper: load first subject file in folder ===
def load_first_subject(folder):
    files = sorted(glob(os.path.join(folder, "*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {folder}")
    print(f"Loaded: {os.path.basename(files[0])}")
    return np.load(files[0])

# === Load data ===
DE      = load_first_subject(de_path)     # shape (40,5,62,200)
PSD     = load_first_subject(psd_path)    # shape (40,5,62,200)
win_100 = load_first_subject(win100_path) # shape (40,5,7,62,100)
win_200 = load_first_subject(win200_path) # shape (40,5,3,62,200)

# ==========================================
# 1. DE Feature Distribution
# ==========================================
plt.figure(figsize=(8,4))
plt.boxplot(DE.reshape(-1, DE.shape[-2]), patch_artist=True,
            boxprops=dict(facecolor='lightblue'))
plt.title("DE Feature Distribution Across Channels")
plt.xlabel("EEG Channel Index")
plt.ylabel("Amplitude")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_path, "feat_distribution_DE.png"), dpi=300)
plt.close()

# ==========================================
# 2. PSD Feature Distribution
# ==========================================
plt.figure(figsize=(8,4))
plt.boxplot(PSD.reshape(-1, PSD.shape[-2]), patch_artist=True,
            boxprops=dict(facecolor='lightgreen'))
plt.title("PSD Feature Distribution Across Channels")
plt.xlabel("EEG Channel Index")
plt.ylabel("Power Spectral Density (a.u.)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_path, "feat_distribution_PSD.png"), dpi=300)
plt.close()

# ==========================================
# 3. Temporal Window Verification – 1 s (EEG_windows_100)
# ==========================================
s, c, t, ch = 0, 0, 0, 0
segment = win_100[c, t, :, ch, :]  # (7,100)
fs = 200
full_signal = segment.flatten()

plt.figure(figsize=(8,3))
plt.plot(np.arange(len(full_signal))/fs, full_signal, 'k', lw=1)
window_len = segment.shape[-1] / fs  # 100 samples / 200 Hz = 0.5 s
starts = np.arange(0, len(full_signal)/fs, window_len)
for s_i in starts[:segment.shape[0]]:
    plt.axvspan(s_i, s_i + window_len, color='orange', alpha=0.25)
plt.title("Temporal Segmentation – 1 s Overlapping Windows")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.tight_layout()
plt.savefig(os.path.join(save_path, "temp_window_verification_1s.png"), dpi=300)
plt.close()

# ==========================================
# 4. Temporal Window Verification – 0.5 s (EEG_windows_200)
# ==========================================
segment = win_200[c, t, :, ch, :]  # (3,200)
full_signal = segment.flatten()

plt.figure(figsize=(8,3))
plt.plot(np.arange(len(full_signal))/fs, full_signal, 'k', lw=1)
window_len = segment.shape[-1] / fs  # 200 samples / 200 Hz = 1 s
starts = np.arange(0, len(full_signal)/fs, window_len/2)
for s_i in starts[:segment.shape[0]]:
    plt.axvspan(s_i, s_i + window_len, color='skyblue', alpha=0.25)
plt.title("Temporal Segmentation – 0.5 s Overlapping Windows")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.tight_layout()
plt.savefig(os.path.join(save_path, "temp_window_verification_0.5s.png"), dpi=300)
plt.close()

print("All figures saved to:", save_path)
