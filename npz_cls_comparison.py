import os
import numpy as np

repo_root = os.path.expanduser("~/Desktop/4022_Code/EEG2Video_my_version")
train_path = os.path.join(repo_root, "sub12_train.npz")

# Load data
data = np.load(train_path, allow_pickle=True)
embs = data["BLIP_embeddings"]   # (N,77,768)
keys = data["keys"]

# Pick two samples from different classes/blocks
i1, i2 = 0, len(embs)//2   # first and middle sample for demo

cls1 = embs[i1][0]   # CLS vector of sample 1
cls2 = embs[i2][0]   # CLS vector of sample 2

# Compute difference
diff_norm = np.linalg.norm(cls1 - cls2)
cosine_sim = np.dot(cls1, cls2) / (np.linalg.norm(cls1) * np.linalg.norm(cls2))

print("Sample 1 key:", keys[i1])
print("Sample 2 key:", keys[i2])
print("CLS vector L2 norm difference:", diff_norm)
print("CLS vector cosine similarity:", cosine_sim)

# Check if all CLS vectors are identical across dataset
all_cls = embs[:,0,:]   # shape (N,768)
ref = all_cls[0]
identical = np.allclose(all_cls, ref, atol=1e-6)
print("Are all CLS vectors identical? ", identical)
