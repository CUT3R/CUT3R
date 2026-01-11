#!/usr/bin/env python3
"""
Preprocess processed_scannetpp scenes to update scene metadata.

This script reads each scene's "scene_metadata.npz", sorts images by timestamp,
updates trajectories, intrinsics, and pair indices, and builds two collections:
  - image_collection: For each image, stores pairs (other image index, score)
  - video_collection: For each image, groups subsequent images whose timestamps
                      differ by at most a given max_interval (and share the same
                      first character in the image name).

The new metadata is saved as "new_scene_metadata.npz" in each scene folder.

Usage:
    python generate_set_scannetpp.py --root /path/to/processed_scannetpp \
        --max_interval 150 --num_workers 8
"""

import os
import os.path as osp
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def get_timestamp(img_name):
    """
    Convert an image name to a timestamp (integer).

    If the image name starts with 'DSC', the timestamp is the integer part after 'DSC'.
    Otherwise, it is assumed the image name has an underscore, and the second element is used.

    Args:
        img_name (str): The image basename (without extension).

    Returns:
        int: The extracted timestamp.
    """
    if img_name.startswith("DSC"):
        return int(img_name[3:])
    else:
        return int(img_name.split("_")[1])


def process_scene(root, scene, max_interval):
    """
    Process a single scene: sort images, update trajectories/intrinsics/pairs, and
    form image and video collections. Save the updated metadata.

    Args:
        root (str): Root directory containing scene folders.
        scene (str): Scene folder name.
        max_interval (int): Maximum allowed difference (in timestamp units) for video grouping.
    """
    scene_dir = osp.join(root, scene)
    metadata_path = osp.join(scene_dir, "scene_metadata.npz")
    with np.load(metadata_path, allow_pickle=True) as data:
        images = data["images"]
        trajectories = data["trajectories"]
        intrinsics = data["intrinsics"]
        pairs = data["pairs"]

    # Sort images by timestamp.
    imgs_with_indices = sorted(enumerate(images), key=lambda x: x[1])
    indices, images = zip(*imgs_with_indices)
    indices = np.array(indices)
    index2sorted = {index: i for i, index in enumerate(indices)}

    # Update trajectories and intrinsics arrays according to the new order.
    trajectories = trajectories[indices]
    intrinsics = intrinsics[indices]

    # Update pairs (each pair is (id1, id2, score)) with new indices.
    pairs = [(index2sorted[id1], index2sorted[id2], score) for id1, id2, score in pairs]

    # Build image_collection: for each pair, verify that both image files exist.
    image_collection = {}
    for id1, id2, score in pairs:
        img1 = images[id1]
        img2 = images[id2]
        img1_path = osp.join(scene_dir, "images", img1 + ".jpg")
        img2_path = osp.join(scene_dir, "images", img2 + ".jpg")
        if not (osp.exists(img1_path) and osp.exists(img2_path)):
            continue
        if id1 not in image_collection:
            image_collection[id1] = []
        image_collection[id1].append((id2, score))

    # Build video_collection: for each image, group subsequent images if:
    #  1. Their timestamp difference is at most max_interval.
    #  2. Their name's first character is the same as the current image.
    video_collection = {}
    for i, image in enumerate(images):
        img_path = osp.join(scene_dir, "images", image + ".jpg")
        if not osp.exists(img_path):
            continue
        video_collection[i] = []
        for j in range(i + 1, len(images)):
            next_img_path = osp.join(scene_dir, "images", images[j] + ".jpg")
            if not osp.exists(next_img_path):
                continue
            if (
                get_timestamp(images[j]) - get_timestamp(image) > max_interval
                or images[j][0] != image[0]
            ):
                break
            video_collection[i].append(j)

    # Save the updated metadata to a new file.
    out_path = osp.join(scene_dir, "new_scene_metadata.npz")
    np.savez(
        out_path,
        images=images,
        trajectories=trajectories,
        intrinsics=intrinsics,
        pairs=pairs,
        image_collection=image_collection,
        video_collection=video_collection,
    )
    print(f"Processed scene: {scene}")


def main(args):
    root = args.root
    max_interval = args.max_interval
    num_workers = args.num_workers

    # Load the list of scenes from the 'all_metadata.npz' file.
    all_metadata_path = osp.join(root, "all_metadata.npz")
    with np.load(all_metadata_path, allow_pickle=True) as data:
        scenes = data["scenes"]

    # Process scenes in parallel.
    futures = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for scene in scenes:
            futures.append(executor.submit(process_scene, root, scene, max_interval))
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing scenes"
        ):
            # This will raise any exceptions from process_scene.
            future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess processed_scannetpp scenes to update scene metadata."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing processed_scannetpp scene folders.",
    )
    parser.add_argument(
        "--max_interval",
        type=int,
        default=150,
        help="Maximum timestamp interval for grouping images (default: 150).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker threads for parallel processing (default: 8).",
    )
    args = parser.parse_args()
    main(args)
