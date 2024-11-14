import os
import argparse
import torch

import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

def is_valid_pytorch_file(file_path):
    try:
        loaded_object = torch.load(file_path, map_location=torch.device('cpu'))
        if isinstance(loaded_object, list) or isinstance(loaded_object, dict):
            pass
        else:
            print('type(loaded_object)', type(loaded_object), file_path)

        return True
    except Exception as e:
        print('e:', e)
        return False

import os
import time

def delete_invalid_pytorch_files(directory, dry_run=False):
    deleted_count = 0
    file_list = []

    # Step 1: Collect all files with their paths and modification times
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.pth', '.pt')):
                file_path = os.path.join(root, file)
                mod_time = os.path.getmtime(file_path)
                file_list.append((file_path, mod_time))

    # Step 2: Sort the file list based on modification time (latest first)
    file_list.sort(key=lambda x: x[1], reverse=True)

    # Step 3: Process the sorted files
    for file_path, _ in file_list:
        if not is_valid_pytorch_file(file_path):
            if dry_run:
                print(f"Would delete invalid file: {file_path}")
            else:
                try:
                    os.remove(file_path)
                    print(f"Deleted invalid file: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {file_path}: {str(e)}")

    return deleted_count

def main():
    parser = argparse.ArgumentParser(description="Delete invalid PyTorch model files (*.pth, *.pt) in a directory.")
    parser.add_argument("directory", help="The directory to search for invalid PyTorch files")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without actually deleting files")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return

    deleted_count = delete_invalid_pytorch_files(args.directory, args.dry_run)

    if args.dry_run:
        print(f"Dry run completed. {deleted_count} invalid files would be deleted.")
    elif deleted_count:
        print(f"Deletion completed. {deleted_count} invalid files were deleted.")

if __name__ == "__main__":
    main()
