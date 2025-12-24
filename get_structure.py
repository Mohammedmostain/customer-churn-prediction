import os

def list_files(startpath):
    # Folders to skip
    ignore_dirs = {'.git', '.venv', 'venv', '__pycache__', '.ipynb_checkpoints', '.vscode', '.idea'}
    
    print(f"Project Structure for: {os.path.abspath(startpath)}")
    print("=" * 50)

    for root, dirs, files in os.walk(startpath):
        # Filter out ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')

if __name__ == '__main__':
    list_files('.')