#!/usr/bin/env python3
"""
Process 3D Ken Burns data by selecting random view types, copying images and depth files,
and computing camera intrinsics from a field-of-view value. The output files are stored in an
organized folder structure.

Usage:
    python preprocess_3dkb.py --root /path/to/data_3d_ken_burns \
                           --out_dir /path/to/processed_3dkb \
                           [--num_workers 4] [--seed 42]
"""

import os
import json
import random
import shutil
from functools import partial
from pathlib import Path
import argparse

import cv2  # noqa: F401; cv2 is imported to ensure OpenEXR support.
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure OpenCV can read OpenEXR files.
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def fov_to_intrinsic_matrix(width, height, fov_deg, fov_type="horizontal"):
    """
    Converts field of view (FOV) in degrees to a camera intrinsic matrix.

    Args:
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        fov_deg (float): Field of view in degrees.
        fov_type (str): 'horizontal' or 'vertical'; determines which FOV is used.

    Returns:
        np.ndarray: A 3x3 camera intrinsic matrix.

    Raises:
        ValueError: If width or height is non-positive or if fov_deg is not in (0, 180).
    """
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive numbers.")
    if not (0 < fov_deg < 180):
        raise ValueError("FOV must be between 0 and 180 degrees (non-inclusive).")
    if fov_type not in ["horizontal", "vertical"]:
        raise ValueError("fov_type must be either 'horizontal' or 'vertical'.")

    fov_rad = np.deg2rad(fov_deg)

    if fov_type == "horizontal":
        f_x = width / (2 * np.tan(fov_rad / 2))
        aspect_ratio = height / width
        f_y = f_x * aspect_ratio
    else:
        f_y = height / (2 * np.tan(fov_rad / 2))
        aspect_ratio = width / height
        f_x = f_y * aspect_ratio

    c_x = width / 2
    c_y = height / 2
    K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])
    return K


def process_basename(root, seq, basename, view_types, out_dir):
    """
    Processes a single basename: selects a random view type, copies the corresponding
    image and depth file, and computes the camera intrinsics from the JSON metadata.

    Args:
        root (str): Root directory of the raw data.
        seq (str): Sequence directory name.
        basename (str): Basename (common identifier) for the files.
        view_types (list): List of view types to choose from (e.g. ['bl', 'br', 'tl', 'tr']).
        out_dir (str): Output directory where processed data will be saved.

    Returns:
        str or None: Returns an error message string on failure; otherwise, returns None.
    """
    # Select a random view type.
    view_type = random.choice(view_types)

    imgname = f"{basename}-{view_type}-image.png"
    depthname = f"{basename}-{view_type}-depth.exr"

    img_path = os.path.join(root, seq, imgname)
    cam_path = os.path.join(root, seq, f"{basename}-meta.json")
    depth_path = os.path.join(root, f"{seq}-depth", depthname)

    # Prepare output directories.
    out_seq_dir = os.path.join(out_dir, seq)
    out_rgb_dir = os.path.join(out_seq_dir, "rgb")
    out_depth_dir = os.path.join(out_seq_dir, "depth")
    out_cam_dir = os.path.join(out_seq_dir, "cam")

    # Output file paths.
    out_img_path = os.path.join(out_rgb_dir, f"{basename}.png")
    out_depth_path = os.path.join(out_depth_dir, f"{basename}.exr")
    out_cam_path = os.path.join(out_cam_dir, f"{basename}.npz")

    try:
        # Load image using PIL and save as PNG.
        with Image.open(img_path) as img:
            W, H = img.size
            img.save(out_img_path, format="PNG")

        # Load camera JSON metadata.
        with open(cam_path, "r") as f:
            cam = json.load(f)
        fov = cam["fltFov"]
        K = fov_to_intrinsic_matrix(W, H, fov)

        # Copy depth file.
        shutil.copy(depth_path, out_depth_path)

        # Save camera intrinsics.
        np.savez(out_cam_path, intrinsics=K)

    except Exception as e:
        return f"Error processing {seq}/{basename}: {e}"

    return None  # Success indicator


def main():
    parser = argparse.ArgumentParser(
        description="Process raw 3D Ken Burns video data and generate processed images, depth maps, and camera intrinsics."
    )
    parser.add_argument(
        "--root", type=str, required=True, help="Root directory of the raw data."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for processed data.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes to use (default: half of available CPUs).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--view_types",
        type=str,
        nargs="+",
        default=["bl", "br", "tl", "tr"],
        help="List of view types to choose from (default: bl br tl tr).",
    )
    args = parser.parse_args()

    # Set the random seed.
    random.seed(args.seed)

    root = args.root
    out_dir = args.out_dir
    view_types = args.view_types

    # Determine number of worker processes.
    num_workers = (
        args.num_workers if args.num_workers is not None else (os.cpu_count() or 4) // 2
    )

    # Collect all sequence directories from root.
    seq_dirs = [
        d
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.endswith("-depth")
    ]

    # Pre-create output directory structure.
    for seq in seq_dirs:
        for subfolder in ["rgb", "depth", "cam"]:
            (Path(out_dir) / seq / subfolder).mkdir(parents=True, exist_ok=True)

    # Prepare list of tasks.
    tasks = []
    for seq in seq_dirs:
        seq_path = os.path.join(root, seq)
        # Assume JSON files contain metadata and have a name ending with "-meta.json".
        json_files = [f for f in os.listdir(seq_path) if f.endswith(".json")]
        # Remove the trailing "-meta.json" (10 characters) to get the basename.
        basenames = sorted([f[:-10] for f in json_files])
        for basename in basenames:
            tasks.append((seq, basename))

    # Define a partial function with fixed root, view_types, and out_dir.
    process_func = partial(
        process_basename, root, view_types=view_types, out_dir=out_dir
    )

    # Process tasks in parallel using ProcessPoolExecutor.
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_func, seq, basename): (seq, basename)
            for seq, basename in tasks
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing"
        ):
            error = future.result()
            if error:
                print(error)


if __name__ == "__main__":
    main()
