#!/usr/bin/env python3
"""
Preprocess the Eden dataset.

This script processes the Eden dataset by copying RGB images, converting depth
data from .mat files to .npy format, and saving camera intrinsics from .mat files
into a structured output directory. Files are processed in parallel using
a ProcessPoolExecutor.

Usage:
    python preprocess_eden.py --root /path/to/data_raw_videos/data_eden \
                              --out_dir /path/to/data_raw_videos/processed_eden \
                              [--num_workers N]
"""

import os
import shutil
import scipy.io
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse


def process_basename(args):
    """
    Process a single basename: load the corresponding image, depth, and camera
    intrinsics files, then copy/save them into the output directories.

    Parameters:
        args (tuple): Contains (seq, basename, rgb_dir, depth_dir, cam_dir,
                      out_rgb_dir, out_depth_dir, out_cam_dir)
    Returns:
        None on success or an error message string on failure.
    """
    (
        seq,
        basename,
        rgb_dir,
        depth_dir,
        cam_dir,
        out_rgb_dir,
        out_depth_dir,
        out_cam_dir,
    ) = args
    # Define output paths.
    out_img_path = os.path.join(out_rgb_dir, f"{basename}.png")
    out_depth_path = os.path.join(out_depth_dir, f"{basename}.npy")
    out_cam_path = os.path.join(out_cam_dir, f"{basename}.npz")

    # Skip processing if the camera file has already been saved.
    if os.path.exists(out_cam_path):
        return None

    try:
        cam_type = "L"
        img_file = os.path.join(rgb_dir, f"{basename}_{cam_type}.png")
        depth_file = os.path.join(depth_dir, f"{basename}_{cam_type}.mat")
        cam_file = os.path.join(cam_dir, f"{basename}.mat")

        # Check if the required files exist.
        if not (
            os.path.exists(img_file)
            and os.path.exists(depth_file)
            and os.path.exists(cam_file)
        ):
            return f"Missing files for {basename} in {seq}"

        # Load depth data.
        depth_mat = scipy.io.loadmat(depth_file)
        depth = depth_mat.get("Depth")
        if depth is None:
            return f"Depth data missing in {depth_file}"
        depth = depth[..., 0]

        # Load camera intrinsics.
        cam_mat = scipy.io.loadmat(cam_file)
        intrinsics = cam_mat.get(f"K_{cam_type}")
        if intrinsics is None:
            return f"Intrinsics data missing in {cam_file}"

        # Copy the RGB image.
        shutil.copyfile(img_file, out_img_path)
        # Save the depth data.
        np.save(out_depth_path, depth)
        # Save the camera intrinsics.
        np.savez(out_cam_path, intrinsics=intrinsics)

    except Exception as e:
        return f"Error processing {basename} in {seq}: {e}"

    return None  # Indicate success.


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Eden dataset: copy RGB images, process depth maps, and save camera intrinsics."
    )
    parser.add_argument(
        "--root", type=str, default="", help="Root directory of the raw Eden data."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory for processed Eden data.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes to use.",
    )
    args = parser.parse_args()

    root = args.root
    out_dir = args.out_dir
    # Modes typically found in the Eden dataset.
    modes = ["clear", "cloudy", "overcast", "sunset", "twilight"]

    rgb_root = os.path.join(root, "RGB")
    depth_root = os.path.join(root, "Depth")
    cam_root = os.path.join(root, "cam_matrix")

    # Collect sequence directories by traversing the RGB root.
    seq_dirs = []
    for d in os.listdir(rgb_root):
        for m in modes:
            seq_path = os.path.join(rgb_root, d, m)
            if os.path.isdir(seq_path):
                # Save the relative path (e.g., "d/m").
                seq_dirs.append(os.path.join(d, m))

    all_tasks = []
    for seq in seq_dirs:
        rgb_dir = os.path.join(rgb_root, seq)
        depth_dir = os.path.join(depth_root, seq)
        cam_dir = os.path.join(cam_root, seq)

        # Create output directories for this sequence.
        # Replace any os.sep in the sequence name with an underscore.
        seq_name = "_".join(seq.split(os.sep))
        out_rgb_dir = os.path.join(out_dir, seq_name, "rgb")
        out_depth_dir = os.path.join(out_dir, seq_name, "depth")
        out_cam_dir = os.path.join(out_dir, seq_name, "cam")
        os.makedirs(out_rgb_dir, exist_ok=True)
        os.makedirs(out_depth_dir, exist_ok=True)
        os.makedirs(out_cam_dir, exist_ok=True)

        # Get basenames from the camera directory (assuming file extension .mat).
        basenames = sorted([d[:-4] for d in os.listdir(cam_dir) if d.endswith(".mat")])

        for basename in basenames:
            task = (
                seq,
                basename,
                rgb_dir,
                depth_dir,
                cam_dir,
                out_rgb_dir,
                out_depth_dir,
                out_cam_dir,
            )
            all_tasks.append(task)

    num_workers = args.num_workers
    print(f"Processing {len(all_tasks)} tasks using {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_basename, task): task[1] for task in all_tasks
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing tasks"
        ):
            error = future.result()
            if error:
                print(error)


if __name__ == "__main__":
    main()
