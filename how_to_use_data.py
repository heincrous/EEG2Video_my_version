"""
DATA SPLITTING AND USAGE SUMMARY
================================

1. Split strategy
-----------------
- Splitting is done at the **clip level**, not at the subject level.
- Each class has 5 clips → we use 4 clips for training and 1 clip for testing.
- If `class07_clip02` is in train, then *all subjects’ EEG* for that clip goes to train.
- Ensures all classes and blocks are represented in both train and test.

2. Resulting counts
-------------------
- Video-level data (subject-independent):
  • 1,400 clips total (7 blocks × 200 clips)
  • Train: 1,120 clips (80%)
  • Test: 280 clips (20%)

- EEG-level data (subject-dependent):
  • 21 subjects × 1,400 clips = 29,400 EEG windows
  • Train: 23,520 windows
  • Test: 5,880 windows

- All splits align by the key:  BlockY/classXX_clipZZ

3. Modality alignment
---------------------
EEG_windows       : [400, 62]       → 400 time points (2s × 200 Hz) × 62 EEG channels
EEG_DE            : [310]           → 62 channels × 5 bands (δ:1–4, θ:4–8, α:8–14, β:14–31, γ:31–50 Hz)
EEG_PSD           : [310]           → 62 channels × 5 bands (δ:1–4, θ:4–8, α:8–14, β:14–31, γ:31–50 Hz)
Video_latents     : [6, 4, 36, 64]  → 6 frames (2s @ 3 FPS) × 4 latent channels × 36 height × 64 width
BLIP_embeddings   : [77, 768]       → 77 tokens × 768-dim CLIP text embedding
BLIP_text         : plain text      → natural-language caption of the clip

4. Usage in training
--------------------
Seq2Seq model (EEG → video latents):
  Input  : EEG_windows
  Target : Video_latents
  Pair   : subX/BlockY/classZZ_clipQQ.npy ↔ BlockY/classZZ_clipQQ.npy

Semantic predictor (EEG → BLIP embeddings):
  Input  : EEG_DE or EEG_PSD
  Target : BLIP_embeddings
  Pair   : subX/BlockY/classZZ_clipQQ.npy ↔ BlockY/classZZ_clipQQ.npy

Diffusion model (video reconstruction):
  Input        : Video_latents
  Conditioning : BLIP_text (and optionally EEG semantic predictor output)
  Output       : Reconstructed video
  Evaluation   : Against Video_mp4 or Video_gif (reference only, not split)

-----------------------------------------
Bottom line:
- Train/test split is clip-based.
- EEG is subject-dependent but tied to clip IDs.
- All modalities align by BlockY/classXX_clipZZ.
- Seq2Seq uses EEG_windows ↔ Video_latents.
- Semantic predictor uses EEG_DE/PSD ↔ BLIP_embeddings.
- Diffusion uses Video_latents ↔ BLIP_text (+ EEG semantics).
"""