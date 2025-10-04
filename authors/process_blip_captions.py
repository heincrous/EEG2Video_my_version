# ==========================================
# BLIP CAPTIONS → CLIP EMBEDDINGS (FULL vs CLEANED)
# ==========================================
# Input:
#   EEG2Video_data/raw/BLIP-caption/Xth_10min.txt  (each has 200 captions)
#
# Process:
#   - Save full captions and cleaned (NOUN+VERB+ADJ only) versions
#   - Encode both sets using Stable Diffusion v1-4 CLIP text encoder
#   - Compare cosine separation between classes
#
# Output:
#   EEG2Video_data/processed/BLIP_text/
#       BLIP_text_full.npy
#       BLIP_text_cleaned.npy
#   EEG2Video_data/processed/CLIP_embeddings/
#       CLIP_embeddings_full.npy
#       CLIP_embeddings_cleaned.npy
# ==========================================

import os
import numpy as np
import torch
import spacy
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# CONFIG
# ==========================================
config = {
    "drive_root": "/content/drive/MyDrive/EEG2Video_data",
    "model_path": "/content/drive/MyDrive/EEG2Video_checkpoints/stable-diffusion-v1-4",
    "batch_size": 128,
}

caption_dir   = os.path.join(config["drive_root"], "raw", "BLIP-caption")
out_text_dir  = os.path.join(config["drive_root"], "processed", "BLIP_text_authors")
out_embed_dir = os.path.join(config["drive_root"], "processed", "CLIP_embeddings_authors")

os.makedirs(out_text_dir, exist_ok=True)
os.makedirs(out_embed_dir, exist_ok=True)


# ==========================================
# LOAD MODELS
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = CLIPTokenizer.from_pretrained(config["model_path"], subfolder="tokenizer")
encoder   = CLIPTextModel.from_pretrained(config["model_path"], subfolder="text_encoder").to(device)
encoder.eval()

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def clean_caption(text):
    doc = nlp(text.lower())
    return " ".join([t.lemma_ for t in doc if t.pos_ in {"NOUN", "VERB", "ADJ"}])

ordinals = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th"]


# ==========================================
# ALLOCATE ARRAYS
# ==========================================
all_texts_full    = np.empty((7, 40, 5), dtype=object)
all_texts_cleaned = np.empty((7, 40, 5), dtype=object)
all_embs_full     = np.zeros((7, 40, 5, 77, 768), dtype=np.float32)
all_embs_cleaned  = np.zeros((7, 40, 5, 77, 768), dtype=np.float32)


# ==========================================
# MAIN LOOP
# ==========================================
for block_id in range(7):
    fname = f"{ordinals[block_id]}_10min.txt"
    fpath = os.path.join(caption_dir, fname)
    if not os.path.exists(fpath):
        print(f"Missing caption file: {fname}")
        continue

    with open(fpath, "r") as f:
        captions = [c.strip() for c in f.readlines() if c.strip()]

    cleaned = [clean_caption(c) for c in captions]

    if len(captions) != 200:
        print(f"Warning: Block {block_id+1} has {len(captions)} captions (expected 200)")

    for start in tqdm(range(0, len(captions), config["batch_size"]), desc=f"Block {block_id+1}"):
        end = start + config["batch_size"]
        full_batch   = captions[start:end]
        clean_batch  = cleaned[start:end]

        # === Encode full captions ===
        tokens_full = tokenizer(full_batch, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
        with torch.inference_mode():
            emb_full = encoder(tokens_full["input_ids"])[0].cpu().numpy()

        # === Encode cleaned captions ===
        tokens_clean = tokenizer(clean_batch, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
        with torch.inference_mode():
            emb_clean = encoder(tokens_clean["input_ids"])[0].cpu().numpy()

        # === Store ===
        for j in range(len(full_batch)):
            idx = start + j
            cls_id  = idx // 5
            clip_id = idx % 5
            if cls_id < 40:
                all_texts_full[block_id, cls_id, clip_id]    = full_batch[j]
                all_texts_cleaned[block_id, cls_id, clip_id] = clean_batch[j]
                all_embs_full[block_id, cls_id, clip_id]     = emb_full[j]
                all_embs_cleaned[block_id, cls_id, clip_id]  = emb_clean[j]


# ==========================================
# SAVE OUTPUTS
# ==========================================
np.save(os.path.join(out_text_dir, "BLIP_text_full.npy"), all_texts_full)
np.save(os.path.join(out_text_dir, "BLIP_text_cleaned.npy"), all_texts_cleaned)
np.save(os.path.join(out_embed_dir, "CLIP_embeddings_full.npy"), all_embs_full)
np.save(os.path.join(out_embed_dir, "CLIP_embeddings_cleaned.npy"), all_embs_cleaned)

print("\nSaved:")
print("BLIP_text_full.npy          ", all_texts_full.shape)
print("BLIP_text_cleaned.npy       ", all_texts_cleaned.shape)
print("CLIP_embeddings_full.npy    ", all_embs_full.shape)
print("CLIP_embeddings_cleaned.npy ", all_embs_cleaned.shape)


# ==========================================
# CLASS SEPARABILITY TEST
# ==========================================
print("\n=== Computing cosine separation between classes ===")
mean_full   = all_embs_full.mean(axis=-2).reshape(-1, 768)
mean_clean  = all_embs_cleaned.mean(axis=-2).reshape(-1, 768)

labels = np.repeat(np.arange(40), 5*7)
cos_full  = cosine_similarity(mean_full)
cos_clean = cosine_similarity(mean_clean)

def compute_sep(cos_matrix, labels):
    within = np.mean([cos_matrix[i,j]
                      for i in range(len(labels))
                      for j in range(len(labels))
                      if labels[i]==labels[j] and i!=j])
    between = np.mean([cos_matrix[i,j]
                       for i in range(len(labels))
                       for j in range(len(labels))
                       if labels[i]!=labels[j]])
    return within, between, within - between

w_f, b_f, d_f = compute_sep(cos_full, labels)
w_c, b_c, d_c = compute_sep(cos_clean, labels)

print(f"Full captions   → within={w_f:.3f}, between={b_f:.3f}, Δ={d_f:.3f}")
print(f"Cleaned captions→ within={w_c:.3f}, between={b_c:.3f}, Δ={d_c:.3f}")
print("\nΔ = (within - between). Higher Δ = better class separation.")
