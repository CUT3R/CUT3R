#!/usr/bin/env python3
"""
Preprocess the MVS Synth dataset.

This script processes each sequence in a given dataset directory by:
  - Reading the RGB image, EXR depth image, and JSON camera parameters.
  - Computing the camera pose from the extrinsic matrix (with a conversion matrix applied).
  - Creating a simple camera intrinsics matrix from the provided focal lengths and principal point.
  - Copying the RGB image (as JPG), saving the depth (as a NumPy array), and saving the camera data (as a NPZ file).

Usage:
    python preprocess_mvs_synth.py --root_dir /path/to/data_mvs_synth/GTAV_720/ \
                                   --out_dir /path/to/processed_mvs_synth \
                                   --num_workers 32
"""

import os
import shutil
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import cv2
import argparse

# Ensure OpenEXR support if needed
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Conversion matrix (example conversion, adjust if needed)
R_conv = np.array(
    [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
)


def process_basename(seq, basename, root_dir, out_dir):
    """
    Process a single frame identified by 'basename' within a given sequence.

    Reads the RGB image, depth (EXR) file, and camera parameters (JSON file),
    computes the adjusted camera pose, builds the camera intrinsics matrix,
    and saves the processed outputs.

    Parameters:
      seq (str): The sequence (subdirectory) name.
      basename (str): The basename of the file (without extension).
      root_dir (str): Root directory containing the raw data.
      out_dir (str): Output directory where processed data will be saved.

    Returns:
      None on success, or an error string on failure.
    """
    try:
        # Define input directories.
        seq_dir = os.path.join(root_dir, seq)
        img_dir = os.path.join(seq_dir, "images")
        depth_dir = os.path.join(seq_dir, "depths")
        cam_dir = os.path.join(seq_dir, "poses")

        # Define input file paths.
        img_path = os.path.join(img_dir, basename + ".png")
        depth_path = os.path.join(depth_dir, basename + ".exr")
        cam_path = os.path.join(cam_dir, basename + ".json")

        # Define output directories.
        out_seq_dir = os.path.join(out_dir, seq)
        out_img_dir = os.path.join(out_seq_dir, "rgb")
        out_depth_dir = os.path.join(out_seq_dir, "depth")
        out_cam_dir = os.path.join(out_seq_dir, "cam")
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_depth_dir, exist_ok=True)
        os.makedirs(out_cam_dir, exist_ok=True)

        # Define output file paths.
        out_img_path = os.path.join(out_img_dir, basename + ".jpg")
        out_depth_path = os.path.join(out_depth_dir, basename + ".npy")
        out_cam_path = os.path.join(out_cam_dir, basename + ".npz")

        # Read and process camera parameters.
        with open(cam_path, "r") as f:
            cam_data = json.load(f)
        c_x = cam_data["c_x"]
        c_y = cam_data["c_y"]
        f_x = cam_data["f_x"]
        f_y = cam_data["f_y"]
        extrinsic = np.array(cam_data["extrinsic"])
        # Invert extrinsic matrix to obtain camera-to-world pose.
        pose = np.linalg.inv(extrinsic)
        # Apply conversion matrix.
        pose = R_conv @ pose

        # Build a simple intrinsics matrix.
        intrinsics = np.array(
            [[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]], dtype=np.float32
        )

        if np.any(np.isinf(pose)) or np.any(np.isnan(pose)):
            raise ValueError(f"Invalid pose for {basename}")

        # Read depth image.
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth[np.isinf(depth)] = 0.0  # Clean up any infinite values

        # Save the processed data.
        shutil.copyfile(img_path, out_img_path)
        np.save(out_depth_path, depth)
        np.savez(out_cam_path, intrinsics=intrinsics, pose=pose)

    except Exception as e:
        return f"Error processing {seq}/{basename}: {e}"

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MVS Synth dataset: convert images, depth, and camera data."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/path/to/data_mvs_synth/GTAV_720/",
        help="Root directory of the raw MVS Synth data.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/path/to/processed_mvs_synth",
        help="Output directory for processed data.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=32, help="Number of parallel workers."
    )
    args = parser.parse_args()

    root_dir = args.root_dir
    out_dir = args.out_dir

    # Get list of sequence directories.
    seqs = sorted(
        [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    )

    # Pre-create output directories for each sequence.
    for seq in seqs:
        out_seq_dir = os.path.join(out_dir, seq)
        os.makedirs(os.path.join(out_seq_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(out_seq_dir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(out_seq_dir, "cam"), exist_ok=True)

    # Build list of processing tasks.
    tasks = []
    for seq in seqs:
        seq_dir = os.path.join(root_dir, seq)
        img_dir = os.path.join(seq_dir, "images")
        basenames = sorted([d[:-4] for d in os.listdir(img_dir) if d.endswith(".png")])
        for basename in basenames:
            tasks.append((seq, basename, root_dir, out_dir))

    num_workers = args.num_workers
    print(f"Processing {len(tasks)} tasks using {num_workers} workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_basename, *task): task[1] for task in tasks}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing"
        ):
            error = future.result()
            if error:
                print(error)


if __name__ == "__main__":
    main()
