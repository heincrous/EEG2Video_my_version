import os
import numpy as np
import torch
import torch.nn.functional as F

repo_root = os.path.expanduser("~/Desktop/4022_Code/EEG2Video_my_version")
train_path = os.path.join(repo_root, "sub12_train.npz")
test_path  = os.path.join(repo_root, "sub12_test.npz")

# === Helper functions ===
def extract_class_id(key):
    fname = key.split("/")[-1]
    class_token = fname.split("_")[0]  # "class07"
    return int(class_token.replace("class", ""))

def pool_embedding(emb, method="exclude_cls"):
    if method == "cls":
        return emb[0]
    elif method == "all_mean":
        return emb.mean(0)
    elif method == "exclude_cls":
        return emb[1:].mean(0)
    else:
        raise ValueError("Unknown pooling method")

def build_prototypes(train_embs, train_keys, method):
    class_prototypes, class_counts = {}, {}
    for emb, key in zip(train_embs, train_keys):
        cls = extract_class_id(key)
        vec = pool_embedding(emb, method)
        if cls not in class_prototypes:
            class_prototypes[cls] = vec
            class_counts[cls] = 1
        else:
            class_prototypes[cls] += vec
            class_counts[cls] += 1
    for cls in class_prototypes:
        class_prototypes[cls] /= class_counts[cls]
        class_prototypes[cls] = torch.tensor(class_prototypes[cls], dtype=torch.float32)
    return class_prototypes

def evaluate(test_embs, test_keys, prototypes, method, show_samples=False):
    correct_top1, correct_top5 = 0, 0
    total = len(test_embs)

    for i, (emb, key) in enumerate(zip(test_embs, test_keys)):
        cls_true = extract_class_id(key)
        vec = torch.tensor(pool_embedding(emb, method), dtype=torch.float32)

        sims = {cls: F.cosine_similarity(vec, proto, dim=0).item()
                for cls, proto in prototypes.items()}
        ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        top_classes = [c for c, _ in ranked[:5]]

        if top_classes[0] == cls_true:
            correct_top1 += 1
        if cls_true in top_classes:
            correct_top5 += 1

        if show_samples and i < 5:
            print(f"{key} | True: {cls_true} | Pred: {top_classes[0]} | Top-5: {top_classes}")

    return correct_top1/total, correct_top5/total, total

# === Main ===
train_data = np.load(train_path, allow_pickle=True)
test_data  = np.load(test_path, allow_pickle=True)

for method in ["cls", "all_mean", "exclude_cls"]:
    print(f"\n--- Method: {method} ---")
    prototypes = build_prototypes(train_data["BLIP_embeddings"], train_data["keys"], method)
    top1, top5, total = evaluate(test_data["BLIP_embeddings"], test_data["keys"], prototypes, method, show_samples=True)
    print(f"Top-1 accuracy: {top1:.4f} ({int(top1*total)}/{total})")
    print(f"Top-5 accuracy: {top5:.4f} ({int(top5*total)}/{total})")
