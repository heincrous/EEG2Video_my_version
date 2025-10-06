# ==========================================
# EEG2Video Semantic Embedding Visualizer (Unseeded)
# ==========================================
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ==========================================
# Config
# ==========================================
FEATURE_TYPE   = "EEG_windows_100"
SUBJECT_NAME   = "sub1.npy"
SUBSET_ID      = "1"

NUM_CLASSES        = 10     # same as len(CLASS_SUBSET)
SAMPLES_PER_CLASS  = 5

EMB_SAVE_PATH = "/content/drive/MyDrive/EEG2Video_outputs/semantic_embeddings"


# ==========================================
# Helper: locate embedding file
# ==========================================
def find_embedding_file():
    expected_name = f"pred_embeddings_{SUBJECT_NAME.replace('.npy','')}_subset{SUBSET_ID}.npy"
    full_path = os.path.join(EMB_SAVE_PATH, expected_name)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")
    print(f"→ Found embedding file: {full_path}")
    return full_path


# ==========================================
# Visualization
# ==========================================
def visualize_embeddings(pred_path):
    preds = np.load(pred_path)
    print(f"Loaded embeddings: {preds.shape}")

    if preds.ndim > 2:
        preds = preds.reshape(NUM_CLASSES * SAMPLES_PER_CLASS, -1)

    print("Running PCA → t-SNE (unseeded, layout may vary each run)...")
    pca = PCA(n_components=50).fit_transform(preds)
    tsne = TSNE(n_components=2, perplexity=15, n_iter=1000, learning_rate='auto').fit_transform(pca)

    plt.figure(figsize=(8,6))
    for i in range(NUM_CLASSES):
        idx = slice(i * SAMPLES_PER_CLASS, (i + 1) * SAMPLES_PER_CLASS)
        plt.scatter(tsne[idx, 0], tsne[idx, 1], label=f"Class {i}")
    plt.title(f"t-SNE of EEG→CLIP Predicted Embeddings\n({FEATURE_TYPE}, subset {SUBSET_ID})")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    pred_path = find_embedding_file()
    visualize_embeddings(pred_path)
