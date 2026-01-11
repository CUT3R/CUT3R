#!/usr/bin/env python3
"""
Preprocess Script for UrbanSyn Dataset

This script:
  1. Reads RGB, depth (EXR), and semantic segmentation (class) files from an UrbanSyn dataset directory.
  2. Retrieves camera intrinsics from a JSON metadata file.
  3. Rescales images, depth maps, and masks to a fixed resolution (e.g., 640×480).
  4. Saves processed data (RGB, .npy depth, .png sky mask, and .npz intrinsics) in an organized structure.

Usage:
    python preprocess_urbansyn.py \
        --input_dir /path/to/data_urbansyn \
        --output_dir /path/to/processed_urbansyn
"""

import os
import json
import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

# Make sure OpenCV EXR support is enabled
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Custom "cropping" module (ensure cropping.py is available/installed)
import cropping


def process_basename(
    basename,
    rgb_dir,
    depth_dir,
    class_dir,
    cam_info,
    out_rgb_dir,
    out_depth_dir,
    out_mask_dir,
    out_cam_dir,
):
    """
    Process a single file triplet (RGB, depth, class) for a given basename.

    Args:
        basename (str): Base name without file extension (e.g., 'image_0001').
        rgb_dir (str): Directory containing RGB .png files.
        depth_dir (str): Directory containing .exr depth files.
        class_dir (str): Directory containing class .png files (semantic segmentation).
        cam_info (dict): Dictionary with camera metadata (focal length, sensor size).
        out_rgb_dir (str): Output directory for rescaled RGB images.
        out_depth_dir (str): Output directory for rescaled depth files.
        out_mask_dir (str): Output directory for sky masks.
        out_cam_dir (str): Output directory for camera intrinsics.

    Returns:
        str or None:
            - Returns None if successful.
            - Returns an error message if something fails.
    """

    # Construct output file paths
    out_img_path = os.path.join(out_rgb_dir, f"{basename}.png")
    out_depth_path = os.path.join(out_depth_dir, f"{basename}.npy")
    out_mask_path = os.path.join(out_mask_dir, f"{basename}.png")
    out_cam_path = os.path.join(out_cam_dir, f"{basename}.npz")

    # Skip if already processed
    if (
        os.path.exists(out_img_path)
        and os.path.exists(out_depth_path)
        and os.path.exists(out_mask_path)
        and os.path.exists(out_cam_path)
    ):
        return None

    try:
        # Build file paths
        img_file = os.path.join(rgb_dir, f"{basename}.png")
        depth_file = os.path.join(depth_dir, f'{basename.replace("rgb", "depth")}.exr')
        class_file = os.path.join(class_dir, basename.replace("rgb", "ss") + ".png")

        # 1. Read RGB image
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        if img is None:
            return f"Error: Could not read image file {img_file}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR -> RGB
        H, W = img.shape[:2]

        # 2. Read depth from EXR
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        if depth is None:
            # Attempt fallback if there's a '.exr.1' file
            alt_exr_1 = depth_file + ".1"
            if os.path.exists(alt_exr_1):
                temp_exr = depth_file.replace(".exr", "_tmp.exr")
                os.rename(alt_exr_1, temp_exr)
                depth = cv2.imread(temp_exr, cv2.IMREAD_UNCHANGED)
                if depth is None:
                    return f"Error reading depth file (fallback) {temp_exr}"
                depth *= 1e5
            else:
                return f"Error reading depth file {depth_file}"
        else:
            depth *= 1e5  # multiply by 1e5, consistent with your original code

        # 3. Read class image, build sky mask
        cl = cv2.imread(class_file, cv2.IMREAD_UNCHANGED)
        if cl is None:
            return f"Error: Could not read class file {class_file}"
        sky_mask = (cl[..., 0] == 10).astype(np.uint8)  # class ID 10 => sky

        # 4. Build camera intrinsics
        f_mm = cam_info["focalLength_mm"]
        w_mm = cam_info["sensorWidth_mm"]
        h_mm = cam_info["sensorHeight_mm"]
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = f_mm / w_mm * W
        K[1, 1] = f_mm / h_mm * H
        K[0, 2] = W / 2
        K[1, 2] = H / 2

        # 5. Combine depth + sky_mask in a single array for rescaling
        depth_with_mask = np.stack([depth, sky_mask], axis=-1)

        # 6. Rescale to desired size
        image_pil = Image.fromarray(img)
        image_rescaled, depth_with_mask_rescaled, K_rescaled = (
            cropping.rescale_image_depthmap(
                image_pil, depth_with_mask, K, output_resolution=(640, 480)
            )
        )

        # Write outputs
        image_rescaled.save(out_img_path)
        np.save(out_depth_path, depth_with_mask_rescaled[..., 0])
        cv2.imwrite(
            out_mask_path, (depth_with_mask_rescaled[..., 1] * 255).astype(np.uint8)
        )
        np.savez(out_cam_path, intrinsics=K_rescaled)

    except Exception as e:
        return f"Error processing {basename}: {e}"

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess UrbanSyn dataset by loading RGB/Depth/Seg "
        "and rescaling them with camera intrinsics."
    )
    parser.add_argument(
        "--input_dir", required=True, help="Path to the UrbanSyn dataset directory."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to the directory where processed data will be stored.",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Define input subdirectories
    rgb_dir = os.path.join(input_dir, "rgb")
    depth_dir = os.path.join(input_dir, "depth")
    class_dir = os.path.join(input_dir, "ss")
    meta_file = os.path.join(input_dir, "camera_metadata.json")

    # Define output subdirectories
    out_rgb_dir = os.path.join(output_dir, "rgb")
    out_depth_dir = os.path.join(output_dir, "depth")
    out_mask_dir = os.path.join(output_dir, "sky_mask")
    out_cam_dir = os.path.join(output_dir, "cam")
    for d in [out_rgb_dir, out_depth_dir, out_mask_dir, out_cam_dir]:
        os.makedirs(d, exist_ok=True)

    # Gather basenames from RGB files
    basenames = sorted(
        [
            os.path.splitext(fname)[0]
            for fname in os.listdir(rgb_dir)
            if fname.endswith(".png")
        ]
    )
    if not basenames:
        print(f"No RGB .png files found in {rgb_dir}. Exiting.")
        return

    # Load camera metadata
    if not os.path.isfile(meta_file):
        print(f"Error: metadata file not found at {meta_file}. Exiting.")
        return

    with open(meta_file, "r") as f:
        cam_info_full = json.load(f)
        cam_info = cam_info_full["parameters"][0]["Camera"]

    # Process in parallel
    num_workers = max(1, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_basename,
                basename,
                rgb_dir,
                depth_dir,
                class_dir,
                cam_info,
                out_rgb_dir,
                out_depth_dir,
                out_mask_dir,
                out_cam_dir,
            ): basename
            for basename in basenames
        }

        # Use tqdm for progress
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing UrbanSyn"
        ):
            error = future.result()
            if error:
                print(error)


if __name__ == "__main__":
    main()
