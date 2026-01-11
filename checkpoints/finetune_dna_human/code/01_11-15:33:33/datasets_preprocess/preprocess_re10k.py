#!/usr/bin/env python3
"""
Usage:
    python preprocess_re10k.py --root_dir /path/to/train \
                             --info_dir /path/to/RealEstate10K/train \
                             --out_dir /path/to/processed_re10k
"""

import os
import shutil
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def build_intrinsics(intrinsics_array, image_size):
    """
    Build a 3x3 camera intrinsics matrix from the given intrinsics array and image size.

    Args:
        intrinsics_array (np.ndarray): An array containing [fx_rel, fy_rel, cx_rel, cy_rel, ...].
                                       We assume the first four components define focal and center
                                       in normalized device coordinates (0..1).
        image_size (tuple): The (width, height) of the image.

    Returns:
        np.ndarray: A 3x3 intrinsics matrix.
    """
    # focal_length = intrinsics[:2] * (width, height)
    # principal_point = intrinsics[2:4] * (width, height)
    width, height = image_size
    fx_rel, fy_rel, cx_rel, cy_rel = intrinsics_array[:4]
    fx = fx_rel * width
    fy = fy_rel * height
    cx = cx_rel * width
    cy = cy_rel * height

    K = np.eye(3, dtype=np.float64)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy

    return K


def compute_pose(extrinsics_array):
    """
    Compute the 4x4 pose matrix by inverting the 3x4 extrinsic matrix (plus a row [0, 0, 0, 1]).

    Args:
        extrinsics_array (np.ndarray): A 12-element array reshaped to (3,4) that
                                       represents a camera-to-world or world-to-camera transform.

    Returns:
        np.ndarray: A 4x4 pose matrix (world-to-camera, or vice versa depending on your convention).
    """
    extrinsics_3x4 = extrinsics_array.reshape(3, 4)
    extrinsics_4x4 = np.vstack([extrinsics_3x4, [0, 0, 0, 1]])
    # Invert the extrinsics to get the pose
    pose = np.linalg.inv(extrinsics_4x4)
    return pose


def process_frame(task):
    """
    Process a single frame:
      - Reads the timestamp, intrinsics, and extrinsics.
      - Copies the image to the output directory.
      - Creates a .npz file containing camera intrinsics and the computed pose.

    Args:
        task (tuple): A tuple that contains:
          (seq_dir, out_rgb_dir, out_cam_dir, raw_line).

    Returns:
        str or None:
            A string with an error message if something fails; otherwise None on success.
    """
    seq_dir, out_rgb_dir, out_cam_dir, raw_line = task

    try:
        # Unpack the raw metadata line
        # Format (assuming): [timestamp, fx_rel, fy_rel, cx_rel, cy_rel, <2 unused>, extrinsics...]
        # Adjust as needed based on the real format of 'raw_line'.
        timestamp = int(raw_line[0])
        intrinsics_array = raw_line[1:7]
        extrinsics_array = raw_line[7:]

        img_name = f"{timestamp}.png"
        src_img_path = os.path.join(seq_dir, img_name)
        if not os.path.isfile(src_img_path):
            return f"Image file not found: {src_img_path}"

        # Derive output paths
        out_img_path = os.path.join(out_rgb_dir, img_name)
        out_cam_path = os.path.join(out_cam_dir, f"{timestamp}.npz")

        # Skip if the camera file already exists
        if os.path.isfile(out_cam_path):
            return None

        # Determine image size without loading the entire image
        with Image.open(src_img_path) as img:
            width, height = img.size

        # Build the intrinsics matrix (K)
        K = build_intrinsics(intrinsics_array, (width, height))

        # Compute the pose matrix
        pose = compute_pose(extrinsics_array)

        # Copy the image to the output directory
        shutil.copyfile(src_img_path, out_img_path)

        # Save intrinsics and pose
        np.savez(out_cam_path, intrinsics=K, pose=pose)

    except Exception as e:
        return f"Error processing frame for {seq_dir} at timestamp {timestamp}: {e}"

    return None  # Success indicator


def process_sequence(seq, root_dir, info_dir, out_dir):
    """
    Process a single sequence:
      - Reads a metadata .txt file containing intrinsics and extrinsics for each frame.
      - Prepares a list of tasks for parallel processing.

    Args:
        seq (str): Name of the sequence.
        root_dir (str): Directory where the original sequence images (e.g., .png) are stored.
        info_dir (str): Directory containing the .txt file with camera metadata for this sequence.
        out_dir (str): Output directory where processed frames will be stored.
    """
    seq_dir = os.path.join(root_dir, seq)
    scene_info_path = os.path.join(info_dir, f"{seq}.txt")

    if not os.path.isfile(scene_info_path):
        tqdm.write(f"Metadata file not found for sequence {seq} - skipping.")
        return

    # Load scene information
    try:
        # skiprows=1 if there's a header line in the .txt, adjust as needed
        scene_info = np.loadtxt(
            scene_info_path, delimiter=" ", dtype=np.float64, skiprows=1
        )
    except Exception as e:
        tqdm.write(f"Error reading scene info for {seq}: {e}")
        return

    # Create output subdirectories
    out_seq_dir = os.path.join(out_dir, seq)
    out_rgb_dir = os.path.join(out_seq_dir, "rgb")
    out_cam_dir = os.path.join(out_seq_dir, "cam")
    os.makedirs(out_rgb_dir, exist_ok=True)
    os.makedirs(out_cam_dir, exist_ok=True)

    # Build tasks
    tasks = [(seq_dir, out_rgb_dir, out_cam_dir, line) for line in scene_info]

    # Process frames in parallel
    with ProcessPoolExecutor(max_workers=os.cpu_count() // 2 or 1) as executor:
        futures = {executor.submit(process_frame, t): t for t in tasks}
        for future in as_completed(futures):
            error_msg = future.result()
            if error_msg:
                tqdm.write(error_msg)


def main():
    parser = argparse.ArgumentParser(
        description="Process video frames and associated camera metadata."
    )
    parser.add_argument(
        "--root_dir",
        required=True,
        help="Directory containing sequence folders with .png images.",
    )
    parser.add_argument(
        "--info_dir", required=True, help="Directory containing metadata .txt files."
    )
    parser.add_argument(
        "--out_dir", required=True, help="Output directory for processed data."
    )
    args = parser.parse_args()

    # Gather a list of sequences (each sequence is a folder under root_dir)
    if not os.path.isdir(args.root_dir):
        raise FileNotFoundError(f"Root directory not found: {args.root_dir}")

    seqs = [
        d
        for d in os.listdir(args.root_dir)
        if os.path.isdir(os.path.join(args.root_dir, d))
    ]
    if not seqs:
        raise ValueError(f"No sequence folders found in {args.root_dir}.")

    # Process each sequence
    for seq in tqdm(seqs, desc="Sequences"):
        process_sequence(seq, args.root_dir, args.info_dir, args.out_dir)


if __name__ == "__main__":
    main()
