#!/usr/bin/env python3

# --------------------------------------------------------
# Script to pre-process the COP3D dataset.
# Usage:
#   python3 preprocess_cop3d.py --cop3d_dir /path/to/cop3d \
#       --output_dir /path/to/processed_cop3d
# --------------------------------------------------------

import argparse
import random
import gzip
import json
import os
import os.path as osp

import torch
import PIL.Image
import numpy as np
import cv2

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import src.dust3r.datasets.utils.cropping as cropping

# Define the object categories. (These are used for seeding.)
CATEGORIES = ["cat", "dog"]
CATEGORIES_IDX = {cat: i for i, cat in enumerate(CATEGORIES)}


def get_parser():
    """Set up the argument parser."""
    parser = argparse.ArgumentParser(
        description="Preprocess the CO3D dataset and output processed images, masks, and metadata."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory for processed CO3D data.",
    )
    parser.add_argument(
        "--cop3d_dir",
        type=str,
        default="",
        help="Directory containing the raw CO3D data.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--min_quality",
        type=float,
        default=0.5,
        help="Minimum viewpoint quality score.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
        help=(
            "Lower dimension will be >= img_size * 3/4, and max dimension will be >= img_size"
        ),
    )
    return parser


def convert_ndc_to_pinhole(focal_length, principal_point, image_size):
    """Convert normalized device coordinates to a pinhole camera intrinsic matrix."""
    focal_length = np.array(focal_length)
    principal_point = np.array(principal_point)
    image_size_wh = np.array([image_size[1], image_size[0]])
    half_image_size = image_size_wh / 2
    rescale = half_image_size.min()
    principal_point_px = half_image_size - principal_point * rescale
    focal_length_px = focal_length * rescale
    fx, fy = focal_length_px[0], focal_length_px[1]
    cx, cy = principal_point_px[0], principal_point_px[1]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def opencv_from_cameras_projection(R, T, focal, p0, image_size):
    """
    Convert camera projection parameters from CO3D (NDC) to OpenCV coordinates.

    Returns:
        R, tvec, camera_matrix: OpenCV-style rotation matrix, translation vector, and intrinsic matrix.
    """
    R = torch.from_numpy(R)[None, :, :]
    T = torch.from_numpy(T)[None, :]
    focal = torch.from_numpy(focal)[None, :]
    p0 = torch.from_numpy(p0)[None, :]
    image_size = torch.from_numpy(image_size)[None, :]

    # Convert to PyTorch3D convention.
    R_pytorch3d = R.clone()
    T_pytorch3d = T.clone()
    focal_pytorch3d = focal
    p0_pytorch3d = p0
    T_pytorch3d[:, :2] *= -1
    R_pytorch3d[:, :, :2] *= -1
    tvec = T_pytorch3d
    R = R_pytorch3d.permute(0, 2, 1)

    # Retype image_size (flip to width, height).
    image_size_wh = image_size.to(R).flip(dims=(1,))

    # Compute scale and principal point.
    scale = image_size_wh.to(R).min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = image_size_wh / 2.0
    principal_point = -p0_pytorch3d * scale + c0
    focal_length = focal_pytorch3d * scale

    camera_matrix = torch.zeros_like(R)
    camera_matrix[:, :2, 2] = principal_point
    camera_matrix[:, 2, 2] = 1.0
    camera_matrix[:, 0, 0] = focal_length[:, 0]
    camera_matrix[:, 1, 1] = focal_length[:, 1]
    return R[0], tvec[0], camera_matrix[0]


def get_set_list(category_dir, split):
    """Obtain a list of sequences for a given category and split."""
    listfiles = os.listdir(osp.join(category_dir, "set_lists"))
    subset_list_files = [f for f in listfiles if "manyview" in f]
    if len(subset_list_files) <= 0:
        subset_list_files = [f for f in listfiles if "fewview" in f]

    sequences_all = []
    for subset_list_file in subset_list_files:
        with open(osp.join(category_dir, "set_lists", subset_list_file)) as f:
            subset_lists_data = json.load(f)
            sequences_all.extend(subset_lists_data[split])
    return sequences_all


def prepare_sequences(
    category, cop3d_dir, output_dir, img_size, split, min_quality, seed
):
    """
    Process sequences for a given category and split.

    This function loads per-frame and per-sequence annotations,
    filters sequences based on quality, crops and rescales images,
    and saves metadata for each frame.

    Returns a dictionary mapping sequence names to lists of selected frame indices.
    """
    random.seed(seed)
    category_dir = osp.join(cop3d_dir, category)
    category_output_dir = osp.join(output_dir, category)
    sequences_all = get_set_list(category_dir, split)

    # Get unique sequence names.
    sequences_numbers = sorted(set(seq_name for seq_name, _, _ in sequences_all))

    # Load frame and sequence annotation files.
    frame_file = osp.join(category_dir, "frame_annotations.jgz")
    sequence_file = osp.join(category_dir, "sequence_annotations.jgz")

    with gzip.open(frame_file, "r") as fin:
        frame_data = json.loads(fin.read())
    with gzip.open(sequence_file, "r") as fin:
        sequence_data = json.loads(fin.read())

    # Organize frame annotations per sequence.
    frame_data_processed = {}
    for f_data in frame_data:
        sequence_name = f_data["sequence_name"]
        frame_data_processed.setdefault(sequence_name, {})[
            f_data["frame_number"]
        ] = f_data

    # Select sequences with quality above the threshold.
    good_quality_sequences = set()
    for seq_data in sequence_data:
        if seq_data["viewpoint_quality_score"] > min_quality:
            good_quality_sequences.add(seq_data["sequence_name"])
    sequences_numbers = [
        seq_name for seq_name in sequences_numbers if seq_name in good_quality_sequences
    ]
    selected_sequences_numbers = sequences_numbers
    selected_sequences_numbers_dict = {
        seq_name: [] for seq_name in selected_sequences_numbers
    }

    # Filter frames to only those from selected sequences.
    sequences_all = [
        (seq_name, frame_number, filepath)
        for seq_name, frame_number, filepath in sequences_all
        if seq_name in selected_sequences_numbers_dict
    ]

    # Process each frame.
    for seq_name, frame_number, filepath in tqdm(
        sequences_all, desc="Processing frames"
    ):
        frame_idx = int(filepath.split("/")[-1][5:-4])
        selected_sequences_numbers_dict[seq_name].append(frame_idx)
        mask_path = filepath.replace("images", "masks").replace(".jpg", ".png")
        frame_data_entry = frame_data_processed[seq_name][frame_number]
        focal_length = frame_data_entry["viewpoint"]["focal_length"]
        principal_point = frame_data_entry["viewpoint"]["principal_point"]
        image_size = frame_data_entry["image"]["size"]
        K = convert_ndc_to_pinhole(focal_length, principal_point, image_size)
        R, tvec, camera_intrinsics = opencv_from_cameras_projection(
            np.array(frame_data_entry["viewpoint"]["R"]),
            np.array(frame_data_entry["viewpoint"]["T"]),
            np.array(focal_length),
            np.array(principal_point),
            np.array(image_size),
        )

        # Load input image and mask.
        image_path = osp.join(cop3d_dir, filepath)
        mask_path_full = osp.join(cop3d_dir, mask_path)
        input_rgb_image = PIL.Image.open(image_path).convert("RGB")
        input_mask = plt.imread(mask_path_full)
        H, W = input_mask.shape

        camera_intrinsics = camera_intrinsics.numpy()
        cx, cy = camera_intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)

        # Crop the image, mask, and adjust intrinsics.
        input_rgb_image, input_mask, input_camera_intrinsics = (
            cropping.crop_image_depthmap(
                input_rgb_image, input_mask, camera_intrinsics, crop_bbox
            )
        )
        scale_final = ((img_size * 3 // 4) / min(H, W)) + 1e-8
        output_resolution = np.floor(np.array([W, H]) * scale_final).astype(int)
        if max(output_resolution) < img_size:
            scale_final = (img_size / max(H, W)) + 1e-8
            output_resolution = np.floor(np.array([W, H]) * scale_final).astype(int)
        input_rgb_image, input_mask, input_camera_intrinsics = (
            cropping.rescale_image_depthmap(
                input_rgb_image, input_mask, input_camera_intrinsics, output_resolution
            )
        )

        # Generate and adjust camera pose.
        camera_pose = np.eye(4, dtype=np.float32)
        camera_pose[:3, :3] = R
        camera_pose[:3, 3] = tvec
        camera_pose = np.linalg.inv(camera_pose)

        # Save processed image and mask.
        save_img_path = osp.join(output_dir, filepath)
        save_mask_path = osp.join(output_dir, mask_path)
        os.makedirs(osp.split(save_img_path)[0], exist_ok=True)
        os.makedirs(osp.split(save_mask_path)[0], exist_ok=True)
        input_rgb_image.save(save_img_path)
        cv2.imwrite(save_mask_path, (input_mask * 255).astype(np.uint8))

        # Save metadata (intrinsics and pose).
        save_meta_path = save_img_path.replace("jpg", "npz")
        np.savez(
            save_meta_path,
            camera_intrinsics=input_camera_intrinsics,
            camera_pose=camera_pose,
        )

    return selected_sequences_numbers_dict


def main():
    parser = get_parser()
    args = parser.parse_args()
    assert (
        args.cop3d_dir != args.output_dir
    ), "Input and output directories must differ."
    categories = CATEGORIES
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each split separately.
    for split in ["train", "test"]:
        selected_sequences_path = osp.join(
            args.output_dir, f"selected_seqs_{split}.json"
        )
        if os.path.isfile(selected_sequences_path):
            continue

        all_selected_sequences = {}
        for category in categories:
            category_output_dir = osp.join(args.output_dir, category)
            os.makedirs(category_output_dir, exist_ok=True)
            category_selected_sequences_path = osp.join(
                category_output_dir, f"selected_seqs_{split}.json"
            )
            if os.path.isfile(category_selected_sequences_path):
                with open(category_selected_sequences_path, "r") as fid:
                    category_selected_sequences = json.load(fid)
            else:
                print(f"Processing {split} - category = {category}")
                category_selected_sequences = prepare_sequences(
                    category=category,
                    cop3d_dir=args.cop3d_dir,
                    output_dir=args.output_dir,
                    img_size=args.img_size,
                    split=split,
                    min_quality=args.min_quality,
                    seed=args.seed + CATEGORIES_IDX[category],
                )
                with open(category_selected_sequences_path, "w") as file:
                    json.dump(category_selected_sequences, file)

            all_selected_sequences[category] = category_selected_sequences

        with open(selected_sequences_path, "w") as file:
            json.dump(all_selected_sequences, file)


if __name__ == "__main__":
    main()
