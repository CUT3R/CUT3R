#!/usr/bin/env python3
"""
This script processes scene data by reading images, depth maps, and camera poses,
computing camera intrinsics, and saving the results in a structured format.

Usage:
    python preprocess_omniobject3d.py --input_dir /path/to/input_root --output_dir /path/to/output_root
"""

import os
import os.path as osp
import json
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cv2
import imageio.v2 as imageio
from tqdm import tqdm
import math

# Enable OpenEXR support in OpenCV
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def prepare_scene_args(scene, input_root, output_root):
    """
    Prepare processing arguments for a given scene.

    Args:
        scene (str): Scene directory name.
        input_root (str): Root directory for input data.
        output_root (str): Root directory for output data.

    Returns:
        list or None: A list of arguments for each frame in the scene or None if preparation fails.
    """
    seq_dir = osp.join(input_root, scene, "render")
    rgb_dir = osp.join(seq_dir, "images")
    depth_dir = osp.join(seq_dir, "depths")
    pose_file = osp.join(seq_dir, "transforms.json")
    out_seq_dir = osp.join(output_root, scene)

    # Check if the necessary file exists
    if not osp.exists(pose_file):
        print(f"Pose file not found: {pose_file}")
        return None

    # Load metadata from JSON
    with open(pose_file, "r") as fp:
        meta = json.load(fp)

    camera_angle_x = float(meta["camera_angle_x"])

    # Create output directories for this scene
    os.makedirs(osp.join(out_seq_dir, "rgb"), exist_ok=True)
    os.makedirs(osp.join(out_seq_dir, "depth"), exist_ok=True)
    os.makedirs(osp.join(out_seq_dir, "cam"), exist_ok=True)

    # Prepare a list of frame processing arguments
    frame_args = [
        (frame, camera_angle_x, rgb_dir, depth_dir, out_seq_dir)
        for frame in meta.get("frames", [])
    ]

    return frame_args


def process_frame(args):
    """
    Process a single frame:
      - Reads the image and depth data.
      - Handles alpha channels by compositing over a white background.
      - Computes the camera intrinsics.
      - Saves the processed RGB image, depth map, and camera parameters.

    Args:
        args (tuple): A tuple containing:
            - frame (dict): Frame metadata.
            - camera_angle_x (float): Camera field-of-view.
            - rgb_dir (str): Directory containing RGB images.
            - depth_dir (str): Directory containing depth maps.
            - out_seq_dir (str): Output directory for the processed scene.
    """
    frame, camera_angle_x, rgb_dir, depth_dir, out_seq_dir = args

    # Derive the base name from the frame's file path
    frame_name = osp.basename(frame["file_path"])

    # Define file paths for input and output
    image_path = osp.join(rgb_dir, frame_name + ".png")
    depth_path = osp.join(depth_dir, frame_name + "_depth.exr")
    out_img_path = osp.join(out_seq_dir, "rgb", frame_name + ".png")
    out_depth_path = osp.join(out_seq_dir, "depth", frame_name + ".npy")
    out_cam_path = osp.join(out_seq_dir, "cam", frame_name + ".npz")

    # Skip processing if outputs already exist
    if (
        osp.exists(out_img_path)
        and osp.exists(out_depth_path)
        and osp.exists(out_cam_path)
    ):
        return

    # Read image using imageio
    img = imageio.imread(image_path)

    # If image has an alpha channel, composite it over a white background
    if img.shape[-1] == 4:
        alpha_channel = img[..., 3]
        rgb_channels = img[..., :3]
        white_background = np.full_like(rgb_channels, 255)
        img = np.where(alpha_channel[..., None] == 0, white_background, rgb_channels)
    else:
        img = img[..., :3]

    H, W, _ = img.shape

    # Process the camera pose
    pose = np.array(frame["transform_matrix"], dtype=np.float32)
    pose[:, 1:3] *= -1  # Invert Y and Z axes if necessary

    # Compute camera intrinsics using the provided camera angle
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    intrinsics = np.array(
        [[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]], dtype=np.float32
    )

    # Read depth data using OpenCV (which supports OpenEXR)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        print(f"Warning: Depth file not found or failed to read: {depth_path}")
        return

    # Use the last channel of the depth data and convert to float32
    depth = depth[..., -1].astype(np.float32)
    depth[depth >= 65504.0] = 0.0  # Set invalid depth values to 0

    # Save the processed outputs
    imageio.imwrite(out_img_path, img.astype(np.uint8))
    np.save(out_depth_path, depth)
    np.savez_compressed(out_cam_path, intrinsics=intrinsics, pose=pose)


def process_scene(frame_args):
    """
    Process all frames within a single scene.

    Args:
        frame_args (list): List of frame arguments for the scene.
    """
    if frame_args is None:
        return

    for args in frame_args:
        process_frame(args)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess scene data by extracting RGB images, depth maps, and camera parameters."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the root input directory containing scene data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where processed data will be saved.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Maximum number of worker processes. Defaults to the number of CPU cores.",
    )
    args = parser.parse_args()

    input_root = args.input_dir
    output_root = args.output_dir

    # Ensure the output root directory exists
    os.makedirs(output_root, exist_ok=True)

    # List all scene directories in the input root
    scenes = sorted(
        [d for d in os.listdir(input_root) if osp.isdir(osp.join(input_root, d))]
    )

    # Determine the number of workers to use
    max_workers = (
        args.max_workers if args.max_workers is not None else os.cpu_count() or 1
    )

    # Prepare processing arguments for each scene in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        scene_args_list = list(
            tqdm(
                executor.map(
                    lambda s: prepare_scene_args(s, input_root, output_root), scenes
                ),
                total=len(scenes),
                desc="Preparing scenes",
            )
        )

    # Filter out scenes where preparation failed
    scene_frame_args = [fa for fa in scene_args_list if fa is not None]

    # Process each scene in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(process_scene, scene_frame_args),
                total=len(scene_frame_args),
                desc="Processing scenes",
            )
        )


if __name__ == "__main__":
    main()
