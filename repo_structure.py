import os

def save_repo_structure(root_dir, output_file="repo_structure.txt"):
    with open(output_file, "w") as f:
        for current_path, dirs, files in os.walk(root_dir):
            # Indent by depth
            depth = current_path.replace(root_dir, "").count(os.sep)
            indent = "    " * depth
            f.write(f"{indent}{os.path.basename(current_path)}/\n")
            for file in files:
                f.write(f"{indent}    {file}\n")

if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.abspath(__file__))
    save_repo_structure(repo_root)
    print("Repository structure saved to repo_structure.txt")