#!/usr/bin/env python3
"""
Preprocess ScanNet scenes to generate video collections.

This script processes each scene in specified splits by reading the image filenames
from the "color" folder, grouping images into video sequences based on a maximum
timestamp interval, and then saving the per-scene metadata as a NumPy .npz file.

Usage:
    python generate_set_scannet.py --root /path/to/processed_scannet \
        --splits scans_test scans_train --max_interval 150 --num_workers 8
"""

import os
import os.path as osp
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def get_timestamp(img_name):
    """
    Convert an image basename to an integer timestamp.

    For ScanNet data, it is assumed that the basename is an integer string.

    Args:
        img_name (str): Image basename (without extension).

    Returns:
        int: The timestamp as an integer.
    """
    return int(img_name)


def process_scene(root, split, scene, max_interval):
    """
    Process a single scene: group images into video sequences and save metadata.

    Args:
        root (str): Root directory for the processed data.
        split (str): Name of the split (e.g., 'scans_test', 'scans_train').
        scene (str): Name of the scene directory.
        max_interval (int): Maximum allowed difference in timestamps for grouping images.
    """
    scene_dir = osp.join(root, split, scene)
    color_dir = osp.join(scene_dir, "color")
    # depth_dir and camera_dir are defined in case you need them in future modifications.
    # depth_dir = osp.join(scene_dir, 'depth')
    # camera_dir = osp.join(scene_dir, 'cam')

    # Get all image basenames from the color folder (without file extension)
    basenames = sorted(
        [f.split(".")[0] for f in os.listdir(color_dir) if f.endswith(".jpg")],
        key=lambda x: get_timestamp(x),
    )

    video_collection = {}
    for i, image in enumerate(basenames):
        video_collection[i] = []
        for j in range(i + 1, len(basenames)):
            # Group images that fall within max_interval seconds of the reference image.
            if get_timestamp(basenames[j]) - get_timestamp(image) > max_interval:
                break
            video_collection[i].append(j)

    # Save the scene metadata (list of basenames and the video collection) to an NPZ file.
    out_path = osp.join(scene_dir, "new_scene_metadata.npz")
    np.savez(out_path, images=basenames, video_collection=video_collection)
    print(f"Processed scene: {scene} (split: {split})")


def main(args):
    root = args.root
    splits = args.splits
    max_interval = args.max_interval
    num_workers = args.num_workers

    futures = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for split in splits:
            split_dir = osp.join(root, split)
            if not osp.isdir(split_dir):
                print(
                    f"Warning: Split directory '{split_dir}' does not exist; skipping."
                )
                continue
            scenes = os.listdir(split_dir)
            for scene in scenes:
                futures.append(
                    executor.submit(process_scene, root, split, scene, max_interval)
                )
        # Use tqdm to display progress as futures complete.
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing scenes"
        ):
            # This will re-raise any exceptions from process_scene.
            future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess ScanNet scenes to create video collections based on image timestamps."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="",
        help="Root directory containing the processed ScanNet splits.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["scans_test", "scans_train"],
        help="List of split directories to process (e.g., scans_test scans_train).",
    )
    parser.add_argument(
        "--max_interval",
        type=int,
        default=150,
        help="Maximum allowed timestamp difference (in integer units) for grouping images.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker threads for parallel processing.",
    )
    args = parser.parse_args()
    main(args)
