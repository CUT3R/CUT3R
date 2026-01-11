#!/usr/bin/env python3
"""
Preprocess Synscapes Data

This script processes Synscapes data by:
  1. Copying the RGB images.
  2. Reading the EXR depth data and saving it as .npy.
  3. Generating a sky mask using the class labels.
  4. Extracting camera intrinsics from the meta file.

The directory structure is expected to be:
    synscapes_dir/
        img/
            rgb/
            depth/
            class/
        meta/
    Each file shares the same base name, e.g. 000000.png/exr in corresponding folders.

Usage:
    python preprocess_synscapes.py \
        --synscapes_dir /path/to/Synscapes/Synscapes \
        --output_dir /path/to/processed_synscapes
"""

import os
import json
import shutil
import argparse
import numpy as np
import cv2
import OpenEXR
from tqdm import tqdm

# Enable EXR support in OpenCV if desired:
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def process_basename(
    basename,
    rgb_dir,
    depth_dir,
    class_dir,
    meta_dir,
    out_rgb_dir,
    out_depth_dir,
    out_mask_dir,
    out_cam_dir,
    sky_id=23,
):
    """
    Process a single sample of the Synscapes dataset:
      1. Reads an RGB .png and depth .exr file.
      2. Reads a class label .png, generating a sky mask.
      3. Reads camera intrinsics from the meta .json file.
      4. Saves the resulting data to the specified output folders.

    Args:
        basename (str): The base filename (without extension).
        rgb_dir (str): Directory containing RGB .png files.
        depth_dir (str): Directory containing depth .exr files.
        class_dir (str): Directory containing class .png files.
        meta_dir (str): Directory containing camera metadata .json files.
        out_rgb_dir (str): Output directory for RGB files.
        out_depth_dir (str): Output directory for depth .npy files.
        out_mask_dir (str): Output directory for sky masks.
        out_cam_dir (str): Output directory for camera intrinsics (.npz).
        sky_id (int): Class ID for sky pixels in the class label images.

    Returns:
        None or str:
            If an error occurs, returns an error message (str). Otherwise, returns None.
    """
    try:
        # Input file paths
        rgb_file = os.path.join(rgb_dir, f"{basename}.png")
        depth_file = os.path.join(depth_dir, f"{basename}.exr")
        class_file = os.path.join(class_dir, f"{basename}.png")
        meta_file = os.path.join(meta_dir, f"{basename}.json")

        # Output file paths
        out_img_path = os.path.join(out_rgb_dir, f"{basename}.png")
        out_depth_path = os.path.join(out_depth_dir, f"{basename}.npy")
        out_mask_path = os.path.join(out_mask_dir, f"{basename}.png")
        out_cam_path = os.path.join(out_cam_dir, f"{basename}.npz")

        # --- Read Depth Data ---
        # If you want to use OpenEXR directly (matching your code), do so here:
        exr_file = OpenEXR.InputFile(depth_file)
        # e.g. reading "Z" channel. Adjust channel name as needed.
        # It's possible that the data is stored in multiple channels (R/G/B or separate "Z").
        # Check your file structure to match the correct channel name.
        # The snippet below is just an example approach using .parts and .channels.
        # If your EXR file is a single-part file with a standard channel, you'd do something like:
        #   depth = np.frombuffer(exr_file.channel('Z', Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32)
        # The way you've shown "parts[0].channels['Z'].pixels" may or may not be valid for your version of PyOpenEXR.

        # This example code is approximate and may need to be adapted:
        # If your version of OpenEXR has a different interface, change accordingly.
        # The snippet below won't work unless you install a specific PyOpenEXR wrapper that supports .parts, .channels, etc.
        #
        # For demonstration, let's assume a single-part EXR with channel 'Z':
        # depth_data = exr_file.channel('Z')  # returns raw bytes
        # depth = np.frombuffer(depth_data, dtype=np.float32).reshape((height, width))  # you need to know (height, width) or read header

        # As you mentioned "np.array(OpenEXR.File(depth_file).parts[0].channels['Z'].pixels)",
        # let's keep it consistent with your original snippet:
        depth = np.array(OpenEXR.InputFile(depth_file).parts[0].channels["Z"].pixels)

        # --- Read Class Image (for Sky Mask) ---
        class_img = cv2.imread(class_file, cv2.IMREAD_UNCHANGED)
        # Create sky mask
        sky_mask = (class_img == sky_id).astype(np.uint8) * 255

        # --- Read Meta Data (for Camera Intrinsics) ---
        with open(meta_file, "r") as f:
            cam_info = json.load(f)["camera"]
            intrinsic = cam_info["intrinsic"]
            fx, fy, cx, cy = (
                intrinsic["fx"],
                intrinsic["fy"],
                intrinsic["u0"],
                intrinsic["v0"],
            )

        K = np.eye(3, dtype=np.float32)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy

        # --- Copy RGB ---
        shutil.copy(rgb_file, out_img_path)

        # --- Save Depth, Mask, and Intrinsics ---
        np.save(out_depth_path, depth)
        cv2.imwrite(out_mask_path, sky_mask)
        np.savez(out_cam_path, intrinsics=K)

    except Exception as e:
        return f"Error processing {basename}: {e}"

    return None


def main():
    parser = argparse.ArgumentParser(description="Preprocess Synscapes data.")
    parser.add_argument(
        "--synscapes_dir",
        required=True,
        help="Path to the main Synscapes directory (contains 'img' and 'meta' folders).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to the output directory for processed data.",
    )
    parser.add_argument(
        "--sky_id",
        type=int,
        default=23,
        help="Class ID for sky pixels in class .png. Default is 23.",
    )
    args = parser.parse_args()

    synscapes_dir = os.path.abspath(args.synscapes_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Define input subdirectories
    rgb_dir = os.path.join(synscapes_dir, "img", "rgb")
    depth_dir = os.path.join(synscapes_dir, "img", "depth")
    class_dir = os.path.join(synscapes_dir, "img", "class")
    meta_dir = os.path.join(synscapes_dir, "meta")

    # Define output subdirectories
    out_rgb_dir = os.path.join(output_dir, "rgb")
    out_depth_dir = os.path.join(output_dir, "depth")
    out_mask_dir = os.path.join(output_dir, "sky_mask")
    out_cam_dir = os.path.join(output_dir, "cam")
    for d in [out_rgb_dir, out_depth_dir, out_mask_dir, out_cam_dir]:
        os.makedirs(d, exist_ok=True)

    # Collect all EXR depth filenames (excluding extension)
    basenames = sorted(
        [
            os.path.splitext(fname)[0]
            for fname in os.listdir(depth_dir)
            if fname.endswith(".exr")
        ]
    )

    # Parallel processing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    num_workers = max(1, os.cpu_count() // 2)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_basename = {
            executor.submit(
                process_basename,
                bname,
                rgb_dir,
                depth_dir,
                class_dir,
                meta_dir,
                out_rgb_dir,
                out_depth_dir,
                out_mask_dir,
                out_cam_dir,
                args.sky_id,
            ): bname
            for bname in basenames
        }

        for future in tqdm(
            as_completed(future_to_basename),
            total=len(future_to_basename),
            desc="Processing Synscapes",
        ):
            basename = future_to_basename[future]
            error = future.result()
            if error:
                print(error)


if __name__ == "__main__":
    main()
