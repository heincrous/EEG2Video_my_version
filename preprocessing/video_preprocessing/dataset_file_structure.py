import os

# Path to your dataset (adapt if needed)
base_path = "/content/drive/MyDrive/EEG2Video_data/raw"

# Walk through the directory and preview structure
for root, dirs, files in os.walk(base_path):
    level = root.replace(base_path, '').count(os.sep)
    indent = ' ' * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 4 * (level + 1)
    for f in files[:10]:  # preview max 10 files per folder
        print(f"{subindent}{f}")
    if len(files) > 10:
        print(f"{subindent}... ({len(files)} files total)")