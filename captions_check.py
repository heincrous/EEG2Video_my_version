# ==========================================
# Print captions for test block (subset only)
# ==========================================
import numpy as np

# Paths
BLIP_TEXT_PATH = "BLIP_text.npy"

# Config
CLASS_SUBSET = [1, 10, 12, 16, 19, 23, 25, 31, 34, 39]  # adjust if needed

# Load captions
blip_text = np.load(BLIP_TEXT_PATH, allow_pickle=True)  # shape (7,40,5)

# Test block = block 6 (0-indexed)
test_block = 6
captions_test = blip_text[test_block]  # shape (40,5)

print(f"=== Captions from test block {test_block+1} (subset classes only) ===\n")

# Print all captions for each class in the subset
for cls in CLASS_SUBSET:
    cls_caps = captions_test[cls]  # shape (5,)
    print(f"Class {cls}:")
    for trial, cap in enumerate(cls_caps):
        print(f"  Trial {trial+1}: {cap}")
    print("-" * 60)
