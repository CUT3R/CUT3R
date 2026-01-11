#!/usr/bin/env python3
"""
Preprocess scenes by sorting images and generating image/video collections.

This script processes scenes in parallel using a thread pool, updating metadata
with sorted images, trajectories, intrinsics, and generating pair, image collection,
and video collection data. The processed metadata is saved to a new file in each scene directory.

Usage:
    python generate_set_arkitscenes.py --root /path/to/data --splits Training Test --max_interval 5.0 --num_workers 8
"""

import os
import os.path as osp
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def get_timestamp(img_name):
    """
    Extract the timestamp from an image filename.
    Assumes the timestamp is the last underscore-separated token in the name (before the file extension).

    Args:
        img_name (str): The image filename.

    Returns:
        float: The extracted timestamp.
    """
    return float(img_name[:-4].split("_")[-1])


def process_scene(root, split, scene, max_interval):
    """
    Process a single scene by sorting its images by timestamp, updating trajectories,
    intrinsics, and pairings, and generating image/video collections.

    Args:
        root (str): Root directory of the dataset.
        split (str): The dataset split (e.g., 'Training', 'Test').
        scene (str): The scene identifier.
        max_interval (float): Maximum allowed time interval (in seconds) between images to consider them in the same video collection.
    """
    scene_dir = osp.join(root, split, scene)
    metadata_path = osp.join(scene_dir, "scene_metadata.npz")

    # Load the scene metadata
    with np.load(metadata_path) as data:
        images = data["images"]
        trajectories = data["trajectories"]
        intrinsics = data["intrinsics"]
        pairs = data["pairs"]

    # Sort images by timestep
    imgs_with_indices = sorted(enumerate(images), key=lambda x: x[1])
    indices, images = zip(*imgs_with_indices)
    indices = np.array(indices)
    index2sorted = {index: i for i, index in enumerate(indices)}

    # Reorder trajectories and intrinsics based on the new image order
    trajectories = trajectories[indices]
    intrinsics = intrinsics[indices]

    # Update pair indices (each pair is (id1, id2, score))
    pairs = [(index2sorted[id1], index2sorted[id2], score) for id1, id2, score in pairs]

    # Form image_collection: mapping from an image id to a list of (other image id, score)
    image_collection = {}
    for id1, id2, score in pairs:
        image_collection.setdefault(id1, []).append((id2, score))

    # Form video_collection: for each image, collect subsequent images within the max_interval time window
    video_collection = {}
    for i, image in enumerate(images):
        j = i + 1
        for j in range(i + 1, len(images)):
            if get_timestamp(images[j]) - get_timestamp(image) > max_interval:
                break
        video_collection[i] = list(range(i + 1, j))

    # Save the new metadata
    output_path = osp.join(scene_dir, "new_scene_metadata.npz")
    np.savez(
        output_path,
        images=images,
        trajectories=trajectories,
        intrinsics=intrinsics,
        pairs=pairs,
        image_collection=image_collection,
        video_collection=video_collection,
    )
    print(f"Processed scene: {scene}")


def main(args):
    """
    Main function to process scenes across specified dataset splits in parallel.
    """
    root = args.root
    splits = args.splits
    max_interval = args.max_interval
    num_workers = args.num_workers

    futures = []

    # Create a ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for split in splits:
            all_meta_path = osp.join(root, split, "all_metadata.npz")
            with np.load(all_meta_path) as data:
                scenes = data["scenes"]

            # Submit processing tasks for each scene in the current split
            for scene in scenes:
                futures.append(
                    executor.submit(process_scene, root, split, scene, max_interval)
                )

        # Use tqdm to display a progress bar as futures complete
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing scenes"
        ):
            # This will raise any exceptions caught during scene processing.
            future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess scene data to update metadata with sorted images and collections."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="",
        help="Root directory containing the dataset splits.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["Training", "Test"],
        help="List of dataset splits to process (e.g., Training Test).",
    )
    parser.add_argument(
        "--max_interval",
        type=float,
        default=5.0,
        help="Maximum time interval (in seconds) between images to consider them in the same video sequence.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker threads for parallel processing.",
    )
    args = parser.parse_args()
    main(args)
