import os

# Root of your cleaned repo
repo_root = os.path.dirname(os.path.abspath(__file__))

# Walk the repo and create __init__.py where needed
for root, dirs, files in os.walk(repo_root):
    # Skip hidden directories like .git
    dirs[:] = [d for d in dirs if not d.startswith('.')]
    
    # Don’t put __init__.py in the repo root itself
    if root == repo_root:
        continue
    
    init_path = os.path.join(root, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("# Package marker\n")
        print(f"Created: {init_path}")
    else:
        print(f"Exists:  {init_path}")
