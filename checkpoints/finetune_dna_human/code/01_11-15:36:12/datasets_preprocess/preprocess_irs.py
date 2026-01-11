#!/usr/bin/env python3
"""
Preprocess the IRS dataset.

This script converts disparity EXR files into depth maps, copies corresponding RGB images,
and saves camera intrinsics computed from a given focal length and baseline. Processing is
done per sequence directory using parallel processing.

Usage:
    python preprocess_irs.py
       --root_dir /path/to/data_irs
       --out_dir /path/to/processed_irs
"""

import os
import shutil
import re
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import OpenEXR
import Imath
import imageio
from PIL import Image
from tqdm import tqdm
import argparse

# Ensure OpenEXR support in OpenCV if needed.
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def exr2hdr(exrpath):
    """
    Read an OpenEXR file and return an HDR image as a NumPy array.
    """
    file = OpenEXR.InputFile(exrpath)
    pixType = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = file.header()["dataWindow"]
    num_channels = len(file.header()["channels"].keys())
    if num_channels > 1:
        channels = ["R", "G", "B"]
        num_channels = 3
    else:
        channels = ["G"]

    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    pixels = [
        np.fromstring(file.channel(c, pixType), dtype=np.float32) for c in channels
    ]
    hdr = np.zeros((size[1], size[0], num_channels), dtype=np.float32)
    if num_channels == 1:
        hdr[:, :, 0] = np.reshape(pixels[0], (size[1], size[0]))
    else:
        hdr[:, :, 0] = np.reshape(pixels[0], (size[1], size[0]))
        hdr[:, :, 1] = np.reshape(pixels[1], (size[1], size[0]))
        hdr[:, :, 2] = np.reshape(pixels[2], (size[1], size[0]))
    return hdr


def writehdr(hdrpath, hdr):
    """
    Write an HDR image to a file using the HDR format.
    If the input has one channel, duplicate it across R, G, and B.
    """
    h, w, c = hdr.shape
    if c == 1:
        hdr = np.pad(hdr, ((0, 0), (0, 0), (0, 2)), "constant")
        hdr[:, :, 1] = hdr[:, :, 0]
        hdr[:, :, 2] = hdr[:, :, 0]
    imageio.imwrite(hdrpath, hdr, format="hdr")


def load_exr(filename):
    """
    Load an EXR file and return the HDR image as a NumPy array.
    """
    hdr = exr2hdr(filename)
    h, w, c = hdr.shape
    if c == 1:
        hdr = np.squeeze(hdr)
    return hdr


def process_basename(args):
    """
    Process a single basename:
      - Load an RGB image and disparity (EXR) file.
      - Compute a depth map from disparity using: depth = (baseline * f) / disparity.
      - Copy the RGB image and save the computed depth and camera intrinsics.

    Parameters:
      args: tuple containing
            (basename, seq_dir, out_rgb_dir, out_depth_dir, out_cam_dir, f, baseline)
    Returns:
      None on success or an error string on failure.
    """
    basename, seq_dir, out_rgb_dir, out_depth_dir, out_cam_dir, f, baseline = args
    out_img_path = os.path.join(out_rgb_dir, f"{basename}.png")
    out_depth_path = os.path.join(out_depth_dir, f"{basename}.npy")
    out_cam_path = os.path.join(out_cam_dir, f"{basename}.npz")
    if os.path.exists(out_cam_path):
        return

    try:
        img_file = os.path.join(seq_dir, f"l_{basename}.png")
        disp_file = os.path.join(seq_dir, f"d_{basename}.exr")

        # Load image using PIL.
        img = Image.open(img_file)

        # Load disparity using the custom load_exr function.
        disp = load_exr(disp_file).astype(np.float32)
        H, W = disp.shape

        # Verify that the image size matches the disparity map.
        if img.size != (W, H):
            return f"Size mismatch for {basename}: Image size {img.size}, Disparity size {(W, H)}"

        # Create a simple camera intrinsics matrix.
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = f
        K[1, 1] = f
        K[0, 2] = W // 2
        K[1, 2] = H // 2

        # Compute depth from disparity.
        depth = baseline * f / disp

        # Copy the RGB image.
        shutil.copyfile(img_file, out_img_path)
        # Save the depth map.
        np.save(out_depth_path, depth)
        # Save the camera intrinsics.
        np.savez(out_cam_path, intrinsics=K)

    except Exception as e:
        return f"Error processing {basename}: {e}"

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess IRS dataset: convert EXR disparity to depth, "
        "copy RGB images, and save camera intrinsics."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/path/to/data_raw_videos/data_irs",
        help="Root directory of the raw IRS data.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/path/to/data_raw_videos/processed_irs",
        help="Output directory for processed IRS data.",
    )
    args = parser.parse_args()

    # Example parameters (adjust as needed)
    baseline = 0.1
    f = 480

    root = args.root_dir
    out_dir = args.out_dir

    # Gather sequence directories.
    seq_dirs = []
    for d in os.listdir(root):
        if os.path.isdir(os.path.join(root, d)):
            if d == "Store":
                for sub in os.listdir(os.path.join(root, d)):
                    if os.path.isdir(os.path.join(root, d, sub)):
                        seq_dirs.append(os.path.join(d, sub))
            elif d == "IRS_small":
                for sub in os.listdir(os.path.join(root, d)):
                    if os.path.isdir(os.path.join(root, d, sub)):
                        for subsub in os.listdir(os.path.join(root, d, sub)):
                            if os.path.isdir(os.path.join(root, d, sub, subsub)):
                                seq_dirs.append(os.path.join(d, sub, subsub))
            else:
                seq_dirs.append(d)

    seq_dirs.sort()

    # Process each sequence.
    for seq in seq_dirs:
        seq_dir = os.path.join(root, seq)
        out_rgb_dir = os.path.join(out_dir, seq, "rgb")
        out_depth_dir = os.path.join(out_dir, seq, "depth")
        out_cam_dir = os.path.join(out_dir, seq, "cam")

        os.makedirs(out_rgb_dir, exist_ok=True)
        os.makedirs(out_depth_dir, exist_ok=True)
        os.makedirs(out_cam_dir, exist_ok=True)

        # Get basenames from disparity files.
        basenames = sorted([d[2:-4] for d in os.listdir(seq_dir) if d.endswith(".exr")])

        tasks = []
        for basename in basenames:
            task = (
                basename,
                seq_dir,
                out_rgb_dir,
                out_depth_dir,
                out_cam_dir,
                f,
                baseline,
            )
            tasks.append(task)

        num_workers = os.cpu_count() // 2
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_basename, task): task[0] for task in tasks
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc=f"Processing {seq}"
            ):
                error = future.result()
                if error:
                    print(error)


if __name__ == "__main__":
    main()
