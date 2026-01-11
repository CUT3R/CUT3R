#!/usr/bin/env python3
"""
Preprocess Script for Point Odyssey Dataset

This script processes the Point Odyssey dataset by:
  - Copying RGB images.
  - Converting 16-bit depth images to a normalized float32 depth map.
  - Inverting camera extrinsic matrices to obtain poses.
  - Saving intrinsics and computed poses in a structured output directory.

The dataset is expected to have subdirectories for each split (e.g., train, test, val),
with each split containing multiple sequence directories. Each sequence directory must
contain the following:
  - An 'rgbs' folder with .jpg images.
  - A 'depths' folder with .png depth images.
  - An 'anno.npz' file with 'intrinsics' and 'extrinsics' arrays.

Usage:
    python preprocess_point_odyssey.py --input_dir /path/to/input_dataset --output_dir /path/to/output_dataset
"""

import os
import argparse
import shutil
import numpy as np
import cv2
from tqdm import tqdm


def process_sequence(seq_dir, out_seq_dir):
    """
    Process a single sequence:
      - Verifies that required folders/files exist.
      - Loads camera annotations.
      - Processes each frame: copies the RGB image, processes the depth map,
        computes the camera pose, and saves the results.

    Args:
        seq_dir (str): Directory of the sequence (should contain 'rgbs', 'depths', and 'anno.npz').
        out_seq_dir (str): Output directory where processed files will be saved.
    """
    # Define input subdirectories and annotation file
    img_dir = os.path.join(seq_dir, "rgbs")
    depth_dir = os.path.join(seq_dir, "depths")
    cam_file = os.path.join(seq_dir, "anno.npz")

    # Ensure all necessary files/folders exist
    if not (
        os.path.exists(img_dir)
        and os.path.exists(depth_dir)
        and os.path.exists(cam_file)
    ):
        raise FileNotFoundError(f"Missing required data in {seq_dir}")

    # Create output subdirectories for images, depth maps, and camera parameters
    out_img_dir = os.path.join(out_seq_dir, "rgb")
    out_depth_dir = os.path.join(out_seq_dir, "depth")
    out_cam_dir = os.path.join(out_seq_dir, "cam")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)
    os.makedirs(out_cam_dir, exist_ok=True)

    # Load camera annotations
    annotations = np.load(cam_file)
    cam_ints = annotations["intrinsics"].astype(np.float32)
    cam_exts = annotations["extrinsics"].astype(np.float32)

    # List and sort image and depth filenames
    rgbs = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    depths = sorted([f for f in os.listdir(depth_dir) if f.endswith(".png")])

    # Ensure that the number of intrinsics, extrinsics, RGB images, and depth images match
    if not (len(cam_ints) == len(cam_exts) == len(rgbs) == len(depths)):
        raise ValueError(
            f"Mismatch in sequence {seq_dir}: "
            f"{len(cam_ints)} intrinsics, {len(cam_exts)} extrinsics, {len(rgbs)} images, {len(depths)} depths."
        )

    # Skip sequence if it has already been processed
    if len(os.listdir(out_img_dir)) == len(rgbs):
        return

    # Process each frame in the sequence
    for i in tqdm(range(len(cam_exts)), desc="Processing frames", leave=False):
        # Extract frame index from filenames
        basename_img = rgbs[i].split(".")[0].split("_")[-1]
        basename_depth = depths[i].split(".")[0].split("_")[-1]
        if int(basename_img) != i or int(basename_depth) != i:
            raise ValueError(
                f"Frame index mismatch in sequence {seq_dir} for frame {i}"
            )

        img_path = os.path.join(img_dir, rgbs[i])
        depth_path = os.path.join(depth_dir, depths[i])

        # Retrieve intrinsics and compute camera pose by inverting the extrinsic matrix
        intrins = cam_ints[i]
        cam_extrinsic = cam_exts[i]
        pose = np.linalg.inv(cam_extrinsic)
        if np.any(np.isinf(pose)) or np.any(np.isnan(pose)):
            raise ValueError(
                f"Invalid pose computed from extrinsics for frame {i} in {seq_dir}"
            )

        # Read and process depth image
        depth_16bit = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        depth = depth_16bit.astype(np.float32) / 65535.0 * 1000.0

        # Save processed files: copy the RGB image and save depth and camera parameters
        basename = basename_img  # or str(i)
        out_img_path = os.path.join(out_img_dir, basename + ".jpg")
        shutil.copyfile(img_path, out_img_path)
        np.save(os.path.join(out_depth_dir, basename + ".npy"), depth)
        np.savez(
            os.path.join(out_cam_dir, basename + ".npz"), intrinsics=intrins, pose=pose
        )


def process_split(split_dir, out_split_dir):
    """
    Process all sequences within a data split (e.g., train, test, or val).

    Args:
        split_dir (str): Directory for the split.
        out_split_dir (str): Output directory for the processed split.
    """
    sequences = sorted(
        [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    )
    for seq in tqdm(
        sequences, desc=f"Processing sequences in {os.path.basename(split_dir)}"
    ):
        seq_dir = os.path.join(split_dir, seq)
        out_seq_dir = os.path.join(out_split_dir, seq)
        process_sequence(seq_dir, out_seq_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Point Odyssey dataset by processing images, depth maps, and camera parameters."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the root input dataset directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the root output directory where processed data will be stored.",
    )
    args = parser.parse_args()

    # Define the expected dataset splits
    splits = ["train", "test", "val"]
    for split in splits:
        split_dir = os.path.join(args.input_dir, split)
        out_split_dir = os.path.join(args.output_dir, split)
        if not os.path.exists(split_dir):
            print(
                f"Warning: Split directory {split_dir} does not exist. Skipping this split."
            )
            continue
        os.makedirs(out_split_dir, exist_ok=True)
        process_split(split_dir, out_split_dir)


if __name__ == "__main__":
    main()
