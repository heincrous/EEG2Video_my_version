"""
General Formatting Rules for EEG2Video Project Scripts
------------------------------------------------------

1. Imports
   - Group imports by type:
       * Standard library (os, sys, glob, gc, random, etc.)
       * Third-party libraries (numpy, torch, diffusers, transformers, einops, etc.)
       * Project-specific modules (pipelines, core_files, training, etc.)
   - Separate groups with a blank line.
   - Precede each group with a clear header comment, e.g.:
       # === Standard libraries ===
       # === Third-party libraries ===
       # === Repo imports ===

2. Path configuration
   - Collect all file and directory paths in a single block under # === Paths ===.
   - Align assignments for readability:
       data_root             = "/..."
       pretrained_model_path = "/..."
   - Immediately create any required directories.

3. Environment / memory setup
   - Place CUDA, system variables, garbage collection, and memory clearing 
     together under # === Memory config ===.
   - Keep setup minimal and consistent.

4. Section headers
   - Use large divider comments to mark each major stage:
       # ==========================================
       # Load pipeline
       # ==========================================
   - Every stage (loading, preprocessing, training, evaluation, saving, etc.) 
     should be clearly separated for easy navigation.

5. Helper functions
   - Define short, self-contained helper functions (e.g., select_ckpt, data reshaping) 
     near where they are used.
   - Surround with divider headers for clarity.

6. Model-related blocks
   - Each model load (pipeline, semantic predictor, seq2seq, scheduler, etc.) 
     in its own section with a divider.
   - Always move models to device with .to(device).
   - Set to .eval() where appropriate.
   - Log key shapes or parameters after forward passes to confirm correct operation.

7. Data handling
   - Open file lists, select samples, and load data in one section.
   - Perform reshaping or preprocessing inline.
   - Always log expected vs actual shapes for verification.
   - When using `.npz` files:
       * Document what arrays are stored inside.
       * Example for EEG2Video bundles:
           - `Video_latents` → numpy array (N,F,C,H,W)
           - `BLIP_text`     → list/array of N caption strings
           - `EEG_windows`   → optional numpy array (N,channels,timepoints)
           - `EEG_DE` or `EEG_PSD` → optional numpy array (N, features)
       * Access arrays explicitly:
           data = np.load(file, allow_pickle=True)
           latents = data["Video_latents"]
           texts   = data["BLIP_text"]
       * Comment expected shapes inline.

8. Task-specific blocks
   - If script is for training, include a # === Training loop === section.
   - If script is for inference, include a # === Run inference === section.
   - If script is for evaluation, include a # === Evaluation === section.
   - Each task should be encapsulated in its own section and function where possible.

9. User interaction (input prompts)
   - Present options clearly in a numbered list.
   - Wording should always follow this pattern:
       print("\nSelect <thing>:")
       print("  [0] option A (description)")
       print("  [1] option B (description)")
       ...
       choice = int(input("Enter choice index: "))
   - Use the word **“index”** in the input request to make it explicit that 
     the user should type the number.
   - After selection, confirm by printing:
       print("Using:", selected_option)

10. General style
   - Use two blank lines between major sections.
   - Keep variable assignments vertically aligned for readability.
   - Limit inline comments to non-obvious logic (especially tensor shapes).
   - Use print statements to log progress, shapes, and save locations.
   - Keep code blocks self-contained, with no hidden dependencies on global state.

11. Logging of metrics
   - Always log **average loss** per epoch in this format:
       print(f"[Epoch {epoch}] Avg loss: {avg_loss:.6f}")
   - Additional metrics (e.g., learning rate, accuracy) should follow the same style:
       print(f"[Epoch {epoch}] Avg loss: {avg_loss:.6f} | LR: {lr:.2e} | Acc: {acc:.3f}")
   - Only include metrics that are useful to monitor; keep logs concise and consistent.
"""