#!/usr/bin/env python3
"""
Preprocess Script for UASOL Dataset

This script processes sequences in the UASOL dataset by:
  - Parsing camera parameters from a 'log.txt' file.
  - Reading a 'complete.json' manifest that describes frames (RGB + depth).
  - Converting depth from millimeters to meters.
  - Rescaling images and depth maps to a fixed resolution (default 640x480).
  - Saving the camera intrinsics and pose in .npz files.

Usage:
    python preprocess_uasol.py \
        --input_dir /path/to/data_uasol \
        --output_dir /path/to/processed_uasol
"""

import os
import json
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

import src.dust3r.datasets.utils.cropping as cropping


def parse_log_file(log_file):
    """
    Parses the log.txt file and returns a dictionary of camera parameters.

    Args:
        log_file (str): Path to the log.txt file containing camera parameters.

    Returns:
        dict: A dictionary of camera parameters parsed from the file.
    """
    camera_dict = {}
    start_parse = False
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("LEFT CAMERA PARAMETERS"):
                start_parse = True
                continue
            if start_parse and ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().replace(" ", "_").lower()
                value = value.strip().strip(".")
                # Handle numeric/list values
                if "," in value or "[" in value:
                    # Convert to list of floats
                    value = [float(v.strip()) for v in value.strip("[]").split(",")]
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                camera_dict[key] = value
    return camera_dict


def process_data(task_args):
    """
    Process a single frame of the dataset:
      - Reads the RGB image and depth map.
      - Converts depth from mm to meters.
      - Rescales the image and depth to a fixed output resolution.
      - Saves results (RGB, depth, camera intrinsics, and pose).

    Args:
        task_args (tuple): A tuple containing:
            - data (dict): Frame info from 'complete.json'.
            - seq_dir (str): Path to the sequence directory.
            - out_rgb_dir (str): Output directory for RGB images.
            - out_depth_dir (str): Output directory for depth maps.
            - out_cam_dir (str): Output directory for camera intrinsics/pose.
            - K (np.ndarray): 3x3 camera intrinsics matrix.
            - H (int): Original image height.
            - W (int): Original image width.

    Returns:
        str or None:
            Returns an error message (str) if something goes wrong.
            Otherwise, returns None on success.
    """
    data, seq_dir, out_rgb_dir, out_depth_dir, out_cam_dir, K, H, W = task_args
    try:
        img_p = data["color_frame_left"]
        depth_p = data["depth_frame"]
        matrix = data["m"]

        # Input file paths
        img_path = os.path.join(seq_dir, "Images", img_p + ".png")
        depth_path = os.path.join(seq_dir, "Images", depth_p + ".png")

        if not (os.path.isfile(img_path) and os.path.isfile(depth_path)):
            return f"Missing files for {img_p}"

        # Read RGB
        img = Image.open(img_path).convert("RGB")

        # Read depth (16-bit or 32-bit), then convert mm to meters
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        if depth.shape[0] != H or depth.shape[1] != W:
            return f"Depth shape mismatch for {img_p}"
        depth = depth / 1000.0  # mm to meters

        # Build the pose matrix
        pose = np.array(matrix, dtype=np.float32)
        # Convert translation (last column) from mm to meters
        pose[:3, 3] /= 1000.0

        # Rescale image and depth to desired output size (e.g., 640x480)
        image, depthmap, camera_intrinsics = cropping.rescale_image_depthmap(
            img, depth, K, output_resolution=(640, 480)
        )

        # Save outputs
        out_img_path = os.path.join(out_rgb_dir, img_p + ".png")
        out_depth_path = os.path.join(out_depth_dir, img_p + ".npy")
        out_cam_path = os.path.join(out_cam_dir, img_p + ".npz")

        image.save(out_img_path)
        np.save(out_depth_path, depthmap)
        np.savez(out_cam_path, intrinsics=camera_intrinsics, pose=pose)

    except Exception as e:
        return f"Error processing {img_p}: {e}"
    return None


def main():
    parser = argparse.ArgumentParser(description="Preprocess UASOL dataset.")
    parser.add_argument(
        "--input_dir", required=True, help="Path to the root UASOL directory."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to the directory where processed data will be stored.",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.input_dir)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Find all sequences that have a 'Images' folder
    seqs = []
    for d in os.listdir(root):
        images_path = os.path.join(root, d, "Images")
        if os.path.isdir(images_path):
            seqs.append(d)

    for seq in seqs:
        seq_dir = os.path.join(root, seq)
        log_file = os.path.join(seq_dir, "log.txt")
        manifest_file = os.path.join(seq_dir, "complete.json")

        # Create output subdirectories
        out_rgb_dir = os.path.join(out_dir, seq, "rgb")
        out_depth_dir = os.path.join(out_dir, seq, "depth")
        out_cam_dir = os.path.join(out_dir, seq, "cam")
        os.makedirs(out_rgb_dir, exist_ok=True)
        os.makedirs(out_depth_dir, exist_ok=True)
        os.makedirs(out_cam_dir, exist_ok=True)

        # Parse camera parameters from log.txt
        camera_dict = parse_log_file(log_file)

        # Extract relevant camera info
        cx = camera_dict["optical_center_along_x_axis,_defined_in_pixels"]
        cy = camera_dict["optical_center_along_y_axis,_defined_in_pixels"]
        fx = camera_dict["focal_length_in_pixels_alog_x_axis"]
        fy = camera_dict["focal_length_in_pixels_alog_y_axis"]
        W, H = map(int, camera_dict["resolution"])
        # Optionally read any 'depth_min_and_max_range_values' if needed
        # depth_range = camera_dict['depth_min_and_max_range_values']

        # Construct intrinsic matrix
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy

        # Read the JSON manifest
        if not os.path.isfile(manifest_file):
            print(
                f"Warning: No manifest file found at {manifest_file}. Skipping {seq}."
            )
            continue

        with open(manifest_file, "r") as f:
            metadata = json.load(f)["Data"]

        # Build tasks for parallel processing
        tasks = []
        for data in metadata:
            tasks.append(
                (data, seq_dir, out_rgb_dir, out_depth_dir, out_cam_dir, K, H, W)
            )

        # Process frames in parallel
        with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            futures = {
                executor.submit(process_data, t): t[0]["color_frame_left"]
                for t in tasks
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc=f"Processing {seq}"
            ):
                error = future.result()
                if error:
                    print(error)


if __name__ == "__main__":
    main()
