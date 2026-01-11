#!/usr/bin/env python3
"""
HOI4D Preprocessing Script

This script processes HOI4D data by:
  1. Searching specific subdirectories for RGB and depth images.
  2. Reading camera intrinsics from a .npy file (one per high-level scene).
  3. Rescaling the RGB images and depth maps to a fixed output resolution
     (e.g., 640x480) using the 'cropping' module.
  4. Saving results (RGB, .npy depth, .npz camera intrinsics) in a new directory structure.

Usage:
    python preprocess_hoi4d.py \
        --root_dir /path/to/HOI4D_release \
        --cam_root /path/to/camera_params \
        --out_dir /path/to/processed_hoi4d
"""

import os
import glob
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse

import src.dust3r.datasets.utils.cropping as cropping

def parse_arguments():
    """
    Parse command-line arguments for HOI4D preprocessing.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess HOI4D dataset by rescaling RGB and depth images."
    )
    parser.add_argument("--root_dir", required=True,
                        help="Path to the HOI4D_release directory.")
    parser.add_argument("--cam_root", required=True,
                        help="Path to the directory containing camera intrinsics.")
    parser.add_argument("--out_dir", required=True,
                        help="Path to the directory where processed files will be saved.")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Number of parallel workers. Default uses half of available CPU cores.")
    args = parser.parse_args()
    return args

def process_image(args):
    """
    Process a single image and depth map:
      - Loads the image (using PIL) and depth (using OpenCV).
      - Converts depth from mm to meters (divided by 1000).
      - Rescales both using 'cropping.rescale_image_depthmap'.
      - Saves the rescaled image (.png), depth (.npy), and camera intrinsics (.npz).

    Args:
        args (tuple): A tuple of:
          (img_path, depth_path, out_img_path, out_depth_path, out_cam_path, intrinsics)

    Returns:
        None. Errors are printed to the console but do not stop the workflow.
    """
    img_path, depth_path, out_img_path, out_depth_path, out_cam_path, intrinsics = args

    try:
        # Load image
        img = Image.open(img_path)

        # Load depth (in mm) and convert to meters
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise ValueError(f"Could not read depth image: {depth_path}")
        depth = depth.astype(np.float32) / 1000.0

        # Rescale image and depth map
        img_rescaled, depth_rescaled, intrinsics_rescaled = cropping.rescale_image_depthmap(
            img, depth, intrinsics.copy(), (640, 480)
        )

        # Save processed data
        img_rescaled.save(out_img_path)      # PNG image
        np.save(out_depth_path, depth_rescaled)  # Depth .npy
        np.savez(out_cam_path, intrinsics=intrinsics_rescaled)

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def main():
    args = parse_arguments()

    root = args.root_dir
    cam_root = args.cam_root
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Collect a list of subdirectories using a glob pattern
    # e.g.: root/ZY2021*/H*/C*/N*/S*/s*/T*
    scene_dirs = glob.glob(os.path.join(root, "ZY2021*", "H*", "C*", "N*", "S*", "s*", "T*"))

    # Build tasks
    tasks = []
    for scene_dir in tqdm(scene_dirs, desc="Collecting scenes"):
        # Build an output sub-directory name
        # Example: "ZY202101/H1/C1/N1/S1/s1/T1" -> "ZY202101_H1_C1_N1_S1_s1_T1"
        scene_relpath = os.path.relpath(scene_dir, root)
        scene_name = "_".join(scene_relpath.split(os.sep))

        # Load camera intrinsics from a .npy file in cam_root
        # e.g., first token of scene_relpath might point to the relevant .npy
        # "ZY202101" -> "cam_root/ZY202101/intrin.npy" (adjust logic as needed)
        top_level = scene_relpath.split(os.sep)[0]
        cam_file = os.path.join(cam_root, top_level, "intrin.npy")
        if not os.path.isfile(cam_file):
            print(f"Warning: Camera file not found: {cam_file}. Skipping {scene_dir}")
            continue
        intrinsics = np.load(cam_file)

        # Directories for this sequence
        rgb_dir = os.path.join(scene_dir, "align_rgb")
        depth_dir = os.path.join(scene_dir, "align_depth")

        # Output directories
        out_rgb_dir = os.path.join(out_dir, scene_name, "rgb")
        out_depth_dir = os.path.join(out_dir, scene_name, "depth")
        out_cam_dir = os.path.join(out_dir, scene_name, "cam")
        os.makedirs(out_rgb_dir, exist_ok=True)
        os.makedirs(out_depth_dir, exist_ok=True)
        os.makedirs(out_cam_dir, exist_ok=True)

        # Find all image paths
        img_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))

        # Build tasks for each image
        for img_path in img_paths:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            depth_path = os.path.join(depth_dir, f"{basename}.png")

            out_img_path = os.path.join(out_rgb_dir, f"{basename}.png")
            out_depth_path = os.path.join(out_depth_dir, f"{basename}.npy")
            out_cam_path = os.path.join(out_cam_dir, f"{basename}.npz")

            # Skip if already processed
            if (os.path.exists(out_img_path) and os.path.exists(out_depth_path) and
                    os.path.exists(out_cam_path)):
                continue

            task = (
                img_path,
                depth_path,
                out_img_path,
                out_depth_path,
                out_cam_path,
                intrinsics
            )
            tasks.append(task)

    # Process tasks in parallel
    max_workers = args.max_workers
    if max_workers is None:
        max_workers = max(1, os.cpu_count() // 2)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(
            executor.map(process_image, tasks),
            total=len(tasks),
            desc="Processing images"
        ))


if __name__ == "__main__":
    main()
