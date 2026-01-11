#!/usr/bin/env python3
"""
Preprocess Script for SmartPortraits Dataset

This script processes each sequence in a specified input directory. Each sequence must contain:
  - An "association.txt" file listing (timestamp_rgb, rgb_filename, timestamp_depth, depth_filename)
  - Pairs of .png files (one for RGB and one for depth)

The script copies each RGB .png file to an output "rgb" folder and converts each 16-bit depth
image to a float32 .npy file in an output "depth" folder. It runs in parallel using
ProcessPoolExecutor for faster performance on multi-core systems.

Usage:
    python preprocess_smartportraits.py \
        --input_dir /path/to/processed_smartportraits1 \
        --output_dir /path/to/processed_smartportraits
"""

import os
import shutil
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_pair(args):
    """
    Process a single (RGB, depth) pair by:
      - Reading the depth .png file and converting it to float32 (depth_in_meters = depth_val / 5000).
      - Copying the RGB file to the output directory.
      - Saving the converted depth to a .npy file.

    Args:
        args (tuple): A tuple containing:
            - seq_dir (str): Path to the sequence directory.
            - seq (str): The name of the current sequence.
            - pair_index (int): Index of the pair in the association file (for naming outputs).
            - pair (tuple): (rgb_filename, depth_filename).
            - out_rgb_dir (str): Output directory for RGB images.
            - out_depth_dir (str): Output directory for depth .npy files.

    Returns:
        None or str:
            - Returns None upon successful processing.
            - Returns an error message (str) if something fails.
    """
    seq_dir, seq, pair_index, pair, out_rgb_dir, out_depth_dir = args
    out_rgb_path = os.path.join(out_rgb_dir, f"{pair_index:06d}.png")
    out_depth_path = os.path.join(out_depth_dir, f"{pair_index:06d}.npy")

    # Skip if both output files already exist
    if os.path.exists(out_rgb_path) and os.path.exists(out_depth_path):
        return None

    try:
        rgb_path = os.path.join(seq_dir, pair[0])
        depth_path = os.path.join(seq_dir, pair[1])

        if not os.path.isfile(rgb_path):
            return f"RGB image not found: {rgb_path}"
        if not os.path.isfile(depth_path):
            return f"Depth image not found: {depth_path}"

        # Read the 16-bit depth file
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth is None:
            return f"Failed to read depth image: {depth_path}"

        # Convert depth values to float32, scale by 1/5000
        depth = depth.astype(np.float32) / 5000.0

        # Copy the RGB image
        shutil.copyfile(rgb_path, out_rgb_path)

        # Save depth as a .npy file
        np.save(out_depth_path, depth)

    except Exception as e:
        return f"Error processing pair {pair_index} in sequence '{seq}': {e}"

    return None


def process_sequence(seq, input_dir, output_dir):
    """
    Process all (RGB, depth) pairs within a single sequence directory.

    Args:
        seq (str): Name of the sequence (subdirectory).
        input_dir (str): Base input directory containing all sequences.
        output_dir (str): Base output directory where processed data will be stored.
    """
    seq_dir = os.path.join(input_dir, seq)
    assoc_file = os.path.join(seq_dir, "association.txt")

    # If the association file does not exist, skip this sequence
    if not os.path.isfile(assoc_file):
        tqdm.write(f"No association.txt found for sequence {seq}. Skipping.")
        return

    # Prepare output directories
    out_rgb_dir = os.path.join(output_dir, seq, "rgb")
    out_depth_dir = os.path.join(output_dir, seq, "depth")
    os.makedirs(out_rgb_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)

    # Read the association file
    pairs = []
    with open(assoc_file, "r") as f:
        for line in f:
            items = line.strip().split()
            # Format: <timestamp_rgb> <rgb_filename> <timestamp_depth> <depth_filename>
            if len(items) < 4:
                continue
            rgb_file = items[1]
            depth_file = items[3]
            pairs.append((rgb_file, depth_file))

    # Build a list of tasks for parallel processing
    tasks = []
    for i, pair in enumerate(pairs):
        task_args = (seq_dir, seq, i, pair, out_rgb_dir, out_depth_dir)
        tasks.append(task_args)

    # Process pairs in parallel
    num_workers = max(1, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_pair, t): t for t in tasks}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc=f"Processing sequence {seq}"
        ):
            error = future.result()
            if error:
                tqdm.write(error)


def main():
    parser = argparse.ArgumentParser(description="Preprocess SmartPortraits dataset.")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to the directory containing all sequences with association.txt files.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to the directory where processed results will be saved.",
    )
    args = parser.parse_args()

    # Gather sequences
    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Input directory not found: {args.input_dir}")

    seqs = sorted(
        [
            d
            for d in os.listdir(args.input_dir)
            if os.path.isdir(os.path.join(args.input_dir, d))
        ]
    )

    if not seqs:
        raise ValueError(f"No valid subdirectories found in {args.input_dir}")

    # Process each sequence
    for seq in tqdm(seqs, desc="Sequences"):
        process_sequence(seq, args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
