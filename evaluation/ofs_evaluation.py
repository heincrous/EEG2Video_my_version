# ==========================================
# EEG2Video Optical Flow Score Distribution (Final)
# ==========================================
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === Paths (Drive already mounted) ===
base_data = "/content/drive/MyDrive/EEG2Video_data/processed/meta-info"
save_path = "/content/drive/MyDrive/EEG2Video_results/figures"
os.makedirs(save_path, exist_ok=True)

# === Load processed scores ===
scores_path = os.path.join(base_data, "All_video_optical_flow_score_byclass.npy")
scores = np.load(scores_path)  # shape (7, 40, 5)
flat_scores = scores.flatten()
print("Loaded optical flow scores:", flat_scores.shape)

# === Compute median threshold ===
median_val = np.median(flat_scores)
print("Median optical flow score:", round(median_val, 4))

# ==========================================
# Plot: Optical Flow Score Distribution
# ==========================================
plt.figure(figsize=(8, 4))
sns.kdeplot(flat_scores, fill=True, color="royalblue", alpha=0.6, linewidth=1.5)
plt.axvline(median_val, color="red", linestyle="--", linewidth=1.2,
            label=f"Median = {median_val:.3f}")
plt.title("Distribution of Optical Flow Scores (OFS)")
plt.xlabel("Optical Flow Score")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()

# === Save ===
save_file = os.path.join(save_path, "optical_flow_distribution.png")
plt.savefig(save_file, dpi=300)
plt.close()

print("Figure saved to:", save_file)
