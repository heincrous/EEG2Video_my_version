import os
import numpy as np
import random
import itertools

repo_root = os.path.expanduser("~/Desktop/4022_Code/EEG2Video_my_version")
train_path = os.path.join(repo_root, "sub12_train.npz")
test_path  = os.path.join(repo_root, "sub12_test.npz")

def extract_class_id(key):
    fname = key.split("/")[-1]
    class_token = fname.split("_")[0]  # "class07"
    return int(class_token.replace("class", ""))

def rowwise_cosine(a, b, include_cls=False):
    if not include_cls:
        a = a[1:]
        b = b[1:]
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return float((a * b).sum(-1).mean())

def compare_cosine(path):
    data = np.load(path, allow_pickle=True)
    embs = data["BLIP_embeddings"]
    keys = data["keys"]

    class_groups = {}
    for i, k in enumerate(keys):
        cid = extract_class_id(k)
        class_groups.setdefault(cid, []).append(embs[i])

    results = {}
    for include_cls in [False, True]:
        same_sims = []
        for cid, group in class_groups.items():
            for i, j in itertools.combinations(range(len(group)), 2):
                same_sims.append(rowwise_cosine(group[i], group[j], include_cls))

        diff_sims = []
        all_classes = list(class_groups.keys())
        for _ in range(100):
            c1, c2 = random.sample(all_classes, 2)
            e1 = random.choice(class_groups[c1])
            e2 = random.choice(class_groups[c2])
            diff_sims.append(rowwise_cosine(e1, e2, include_cls))

        mode = "INCLUDING CLS" if include_cls else "EXCLUDING CLS"
        results[mode] = {
            "same_mean": np.mean(same_sims),
            "same_std": np.std(same_sims),
            "diff_mean": np.mean(diff_sims),
            "diff_std": np.std(diff_sims),
            "same_n": len(same_sims),
            "diff_n": len(diff_sims),
        }
    print(f"\nComparison for {path}:")
    for mode, stats in results.items():
        print(f"{mode}:")
        print(f"  Same-class mean={stats['same_mean']:.4f}, std={stats['same_std']:.4f}, n={stats['same_n']}")
        print(f"  Diff-class mean={stats['diff_mean']:.4f}, std={stats['diff_std']:.4f}, n={stats['diff_n']}")

# === Main ===
compare_cosine(train_path)
compare_cosine(test_path)
