#!/usr/bin/env python3
"""
Preprocess the Matterport3D (MP3D) dataset.

This script reads camera parameters and overlap data from a configuration file,
processes RGB images and corresponding depth images, adjusts camera poses using a
conversion matrix, and then saves the processed images, depth maps, and camera
metadata into separate output directories.

Usage:
    python preprocess_mp3d.py --root_dir /path/to/data_mp3d/v1/scans \
                              --out_dir /path/to/processed_mp3d
"""

import os
import numpy as np
import cv2
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse


def process_image(args):
    """
    Process a single image: reads the RGB image and depth image, normalizes the depth,
    adjusts the camera pose using a conversion matrix, and saves the processed outputs.

    Parameters:
      args: tuple containing
         (i, paths, K, pose, img_dir, depth_dir, out_rgb_dir, out_depth_dir, out_cam_dir, R_conv)
         where:
           i             - the frame index
           paths         - tuple of (depth filename, RGB filename)
           K             - camera intrinsics matrix (3x3 NumPy array)
           pose          - camera pose (4x4 NumPy array)
           img_dir       - directory containing RGB images
           depth_dir     - directory containing depth images
           out_rgb_dir   - output directory for processed RGB images
           out_depth_dir - output directory for processed depth maps
           out_cam_dir   - output directory for processed camera metadata
           R_conv        - a 4x4 conversion matrix (NumPy array)
    Returns:
      None if successful, or an error string if processing fails.
    """
    (
        i,
        paths,
        K,
        pose,
        img_dir,
        depth_dir,
        out_rgb_dir,
        out_depth_dir,
        out_cam_dir,
        R_conv,
    ) = args

    depth_path, img_path = paths
    img_path_full = os.path.join(img_dir, img_path)
    depth_path_full = os.path.join(depth_dir, depth_path)

    try:
        # Read depth image using OpenCV (assumed to be stored with 16-bit depth)
        depth = cv2.imread(depth_path_full, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth = depth / 4000.0  # Normalize depth (adjust this factor as needed)

        # Adjust the camera pose with the conversion matrix
        pose_adjusted = pose @ R_conv

        # Generate output filenames using a zero-padded frame index.
        basename = f"{i:06d}"
        out_img_path = os.path.join(out_rgb_dir, basename + ".png")
        out_depth_path = os.path.join(out_depth_dir, basename + ".npy")
        out_cam_path = os.path.join(out_cam_dir, basename + ".npz")

        # Copy the RGB image.
        shutil.copyfile(img_path_full, out_img_path)

        # Save the depth map.
        np.save(out_depth_path, depth)

        # Save the camera intrinsics and adjusted pose.
        np.savez(out_cam_path, intrinsics=K, pose=pose_adjusted)

    except Exception as e:
        return f"Error processing image {img_path}: {e}"

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MP3D scans: convert and save RGB images, depth maps, and camera metadata."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/path/to/data_mp3d/v1/scans",
        help="Root directory of the raw MP3D data.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/path/to/processed_mp3d",
        help="Output directory for processed MP3D data.",
    )
    args = parser.parse_args()

    root = args.root_dir
    out_dir = args.out_dir

    # List sequence directories (each scan is stored as a separate directory).
    seqs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

    # Define a conversion matrix from MP3D to the desired coordinate system.
    R_conv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32
    )

    for seq in tqdm(seqs, desc="Sequences"):
        # The sequence directory structure assumes that images and depth files are stored
        # under a subdirectory with the same name as the sequence.
        seq_dir = os.path.join(root, seq, seq)

        img_dir = os.path.join(seq_dir, "undistorted_color_images")
        depth_dir = os.path.join(seq_dir, "undistorted_depth_images")
        cam_file = os.path.join(seq_dir, "undistorted_camera_parameters", f"{seq}.conf")
        overlap_file = os.path.join(seq_dir, "image_overlap_data", f"{seq}_iis.txt")

        # Read overlap data and save it (optional).
        overlap = []
        with open(overlap_file, "r") as f:
            for line in f:
                parts = line.split()
                overlap.append([int(parts[1]), int(parts[2]), float(parts[3])])
        overlap = np.array(overlap)
        os.makedirs(os.path.join(out_dir, seq), exist_ok=True)
        np.save(os.path.join(out_dir, seq, "overlap.npy"), overlap)

        # Read camera parameters from a configuration file.
        intrinsics = []
        camera_poses = []
        image_files = []

        with open(cam_file, "r") as file:
            lines = file.readlines()
        current_intrinsics = None
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "intrinsics_matrix":
                # Extract intrinsic parameters.
                fx, cx, fy, cy = (
                    float(parts[1]),
                    float(parts[3]),
                    float(parts[5]),
                    float(parts[6]),
                )
                current_intrinsics = np.array(
                    [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
                )
            elif parts[0] == "scan":
                # Read the image filenames and camera pose.
                depth_image = parts[1]
                color_image = parts[2]
                image_files.append((depth_image, color_image))
                matrix_values = list(map(float, parts[3:]))
                camera_pose = np.array(matrix_values).reshape(4, 4)
                camera_poses.append(camera_pose)
                if current_intrinsics is not None:
                    intrinsics.append(current_intrinsics.copy())

        if not (len(image_files) == len(intrinsics) == len(camera_poses)):
            print(f"Inconsistent data in sequence {seq}")
            continue

        # Prepare output directories.
        out_rgb_dir = os.path.join(out_dir, seq, "rgb")
        out_depth_dir = os.path.join(out_dir, seq, "depth")
        out_cam_dir = os.path.join(out_dir, seq, "cam")
        os.makedirs(out_rgb_dir, exist_ok=True)
        os.makedirs(out_depth_dir, exist_ok=True)
        os.makedirs(out_cam_dir, exist_ok=True)

        tasks = []
        for i, (paths, K, pose) in enumerate(
            zip(image_files, intrinsics, camera_poses)
        ):
            args_task = (
                i,
                paths,
                K,
                pose,
                img_dir,
                depth_dir,
                out_rgb_dir,
                out_depth_dir,
                out_cam_dir,
                R_conv,
            )
            tasks.append(args_task)

        num_workers = os.cpu_count() // 2
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_image, task): task[0] for task in tasks}
            for future in tqdm(
                as_completed(futures), total=len(futures), desc=f"Processing {seq}"
            ):
                error = future.result()
                if error:
                    print(error)


if __name__ == "__main__":
    main()
