import os
import shutil
from tqdm import tqdm

# Set these paths to your original and moved locations.
src_base = "/path/to/processed_dl3dv"  # original location
dst_base = "processed_dl3dv_ours"  # current (moved) location

# Set dry_run to True for testing (no changes made), and False to perform the actions.
dry_run = False

def merge_directories(source_dir, destination_dir, dry_run=False):
    """
    Merge all contents from source_dir into destination_dir.
    If an item already exists in destination_dir:
      - For files: remove the destination file and move the source file.
      - For directories: merge them recursively.
    After moving items, empty directories are removed.
    """
    for item in os.listdir(source_dir):
        source_item = os.path.join(source_dir, item)
        dest_item = os.path.join(destination_dir, item)
        if os.path.isdir(source_item):
            if os.path.exists(dest_item):
                # Recursively merge subdirectories.
                merge_directories(source_item, dest_item, dry_run=dry_run)
                # Remove the source subdirectory if empty.
                if not os.listdir(source_item):
                    if dry_run:
                        print(f"[Dry-run] Would remove empty directory: {source_item}")
                    else:
                        os.rmdir(source_item)
            else:
                if dry_run:
                    print(f"[Dry-run] Would move directory: {source_item} -> {dest_item}")
                else:
                    shutil.move(source_item, dest_item)
        else:
            # For files: if a file already exists at the destination, remove it.
            if os.path.exists(dest_item):
                if dry_run:
                    print(f"[Dry-run] Would remove existing file: {dest_item}")
                else:
                    os.remove(dest_item)
            if dry_run:
                print(f"[Dry-run] Would move file: {source_item} -> {dest_item}")
            else:
                shutil.move(source_item, dest_item)

# Build a list of relative folder paths in dst_base.
# This assumes the structure is: dst_base/f1/f2
all_folders = []
for f1 in os.listdir(dst_base):
    f1_path = os.path.join(dst_base, f1)
    if not os.path.isdir(f1_path):
        continue
    for f2 in os.listdir(f1_path):
        all_folders.append(os.path.join(f1, f2))

# Process each folder and move/merge it back to the original location.
for folder in tqdm(all_folders, desc="Moving folders back"):
    original_folder = os.path.join(src_base, folder)  # target location in the original path
    moved_folder = os.path.join(dst_base, folder)       # current location

    # Ensure the parent directory of the original folder exists.
    parent_dir = os.path.dirname(original_folder)
    if dry_run:
        if not os.path.exists(parent_dir):
            print(f"[Dry-run] Would create directory: {parent_dir}")
    else:
        os.makedirs(parent_dir, exist_ok=True)

    if not os.path.exists(original_folder):
        if dry_run:
            print(f"[Dry-run] Would move folder: {moved_folder} -> {original_folder}")
        else:
            shutil.move(moved_folder, original_folder)
    else:
        merge_directories(moved_folder, original_folder, dry_run=dry_run)
        # Remove the moved folder if it becomes empty.
        if not os.listdir(moved_folder):
            if dry_run:
                print(f"[Dry-run] Would remove empty directory: {moved_folder}")
            else:
                os.rmdir(moved_folder)
