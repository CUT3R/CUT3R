#!/usr/bin/env python3
"""
Preprocess the Dynamic Replica dataset.

This script reads frame annotations (stored in compressed JSON files),
loads images, depth maps, optical flow, and camera parameters, and saves
processed images, depth maps, flow files, and camera metadata (intrinsics and poses)
to an output directory organized by split, sequence, and camera view.

Usage:
    python preprocess_dynamic_replica.py --root_dir /path/to/data_dynamic_replica \
                                           --out_dir /path/to/processed_dynamic_replica \
                                           [--splits train valid test] \
                                           [--num_processes 8]
"""

import argparse
import gzip
import json
import os
import os.path as osp
import re
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
from PIL import Image
from pytorch3d.implicitron.dataset.types import (
    FrameAnnotation as ImplicitronFrameAnnotation,
    load_dataclass,
)
from tqdm import tqdm
import imageio

# Enable OpenEXR support in OpenCV.
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

TAG_CHAR = np.array([202021.25], np.float32)


def readFlow(fn):
    """Read .flo file in Middlebury format."""
    with open(fn, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print("Magic number incorrect. Invalid .flo file")
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            return np.resize(data, (int(h), int(w), 2))


def readPFM(file):
    with open(file, "rb") as f:
        header = f.readline().rstrip()
        if header == b"PF":
            color = True
        elif header == b"Pf":
            color = False
        else:
            raise Exception("Not a PFM file.")

        dim_match = re.match(rb"^(\d+)\s(\d+)\s$", f.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(f.readline().rstrip())
        endian = "<" if scale < 0 else ">"
        if scale < 0:
            scale = -scale

        data = np.fromfile(f, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data


def read_gen(file_name, pil=False):
    ext = osp.splitext(file_name)[-1].lower()
    if ext in [".png", ".jpeg", ".ppm", ".jpg"]:
        return Image.open(file_name)
    elif ext in [".bin", ".raw"]:
        return np.load(file_name)
    elif ext == ".flo":
        return readFlow(file_name).astype(np.float32)
    elif ext == ".pfm":
        flow = readPFM(file_name).astype(np.float32)
        return flow if len(flow.shape) == 2 else flow[:, :, :-1]
    return []


def _load_16big_png_depth(depth_png):
    with Image.open(depth_png) as depth_pil:
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth


@dataclass
class DynamicReplicaFrameAnnotation(ImplicitronFrameAnnotation):
    """A dataclass used to load annotations from .json for Dynamic Replica."""

    camera_name: Optional[str] = None
    instance_id_map_path: Optional[str] = None
    flow_forward: Optional[str] = None
    flow_forward_mask: Optional[str] = None
    flow_backward: Optional[str] = None
    flow_backward_mask: Optional[str] = None
    trajectories: Optional[str] = None


def _get_pytorch3d_camera(entry_viewpoint, image_size, scale: float):
    """
    Convert the camera parameters stored in an annotation to PyTorch3D convention.

    Returns:
        R, tvec, focal, principal_point
    """
    assert entry_viewpoint is not None
    principal_point = torch.tensor(entry_viewpoint.principal_point, dtype=torch.float)
    focal_length = torch.tensor(entry_viewpoint.focal_length, dtype=torch.float)
    half_image_size_wh_orig = (
        torch.tensor(list(reversed(image_size)), dtype=torch.float) / 2.0
    )

    fmt = entry_viewpoint.intrinsics_format
    if fmt.lower() == "ndc_norm_image_bounds":
        rescale = half_image_size_wh_orig
    elif fmt.lower() == "ndc_isotropic":
        rescale = half_image_size_wh_orig.min()
    else:
        raise ValueError(f"Unknown intrinsics format: {fmt}")

    principal_point_px = half_image_size_wh_orig - principal_point * rescale
    focal_length_px = focal_length * rescale

    # Prepare rotation and translation for PyTorch3D
    R = torch.tensor(entry_viewpoint.R, dtype=torch.float)
    T = torch.tensor(entry_viewpoint.T, dtype=torch.float)
    R_pytorch3d = R.clone()
    T_pytorch3d = T.clone()
    T_pytorch3d[..., :2] *= -1
    R_pytorch3d[..., :, :2] *= -1
    tvec = T_pytorch3d
    R = R_pytorch3d.T

    return R, tvec, focal_length_px, principal_point_px


# Global configuration for splits and output.
SPLITS = ["train", "valid", "test"]
# (You can override the default root and out_dir via command-line arguments.)


def process_split_data(args):
    """
    Process all frames for a given split.

    Reads the frame annotation file for the given split, groups frames per sequence
    and camera, and for each frame loads the image, depth map, optical flows (if available),
    computes the camera intrinsics and pose (using _get_pytorch3d_camera), and saves the data.
    """
    split, root_dir, out_dir = args
    split_dir = osp.join(root_dir, split)
    # The frame annotations are stored in a compressed json file.
    frame_annotations_file = osp.join(split_dir, f"frame_annotations_{split}.jgz")
    with gzip.open(frame_annotations_file, "rt", encoding="utf8") as zipfile:
        frame_annots_list = load_dataclass(zipfile, List[DynamicReplicaFrameAnnotation])

    # Group frames by sequence and camera.
    seq_annot = defaultdict(lambda: defaultdict(list))
    for frame_annot in frame_annots_list:
        seq_annot[frame_annot.sequence_name][frame_annot.camera_name].append(
            frame_annot
        )

    # Process each sequence.
    for seq_name in tqdm(seq_annot.keys(), desc=f"Processing split '{split}'"):
        # For each camera (e.g., 'left', 'right'), create output directories.
        for cam in ["left", "right"]:
            out_img_dir = osp.join(out_dir, split, seq_name, cam, "rgb")
            out_depth_dir = osp.join(out_dir, split, seq_name, cam, "depth")
            out_fflow_dir = osp.join(out_dir, split, seq_name, cam, "flow_forward")
            out_bflow_dir = osp.join(out_dir, split, seq_name, cam, "flow_backward")
            out_cam_dir = osp.join(out_dir, split, seq_name, cam, "cam")
            os.makedirs(out_img_dir, exist_ok=True)
            os.makedirs(out_depth_dir, exist_ok=True)
            os.makedirs(out_fflow_dir, exist_ok=True)
            os.makedirs(out_bflow_dir, exist_ok=True)
            os.makedirs(out_cam_dir, exist_ok=True)

            for framedata in tqdm(
                seq_annot[seq_name][cam], desc=f"Seq {seq_name} [{cam}]", leave=False
            ):
                timestamp = framedata.frame_timestamp
                im_path = osp.join(split_dir, framedata.image.path)
                depth_path = osp.join(split_dir, framedata.depth.path)
                if framedata.flow_forward["path"]:
                    flow_forward_path = osp.join(
                        split_dir, framedata.flow_forward["path"]
                    )
                    flow_forward_mask_path = osp.join(
                        split_dir, framedata.flow_forward_mask["path"]
                    )
                if framedata.flow_backward["path"]:
                    flow_backward_path = osp.join(
                        split_dir, framedata.flow_backward["path"]
                    )
                    flow_backward_mask_path = osp.join(
                        split_dir, framedata.flow_backward_mask["path"]
                    )

                # Ensure required files exist.
                assert os.path.isfile(im_path), im_path
                assert os.path.isfile(depth_path), depth_path
                if framedata.flow_forward["path"]:
                    assert os.path.isfile(flow_forward_path), flow_forward_path
                    assert os.path.isfile(
                        flow_forward_mask_path
                    ), flow_forward_mask_path
                if framedata.flow_backward["path"]:
                    assert os.path.isfile(flow_backward_path), flow_backward_path
                    assert os.path.isfile(
                        flow_backward_mask_path
                    ), flow_backward_mask_path

                viewpoint = framedata.viewpoint
                # Load depth map.
                depth = _load_16big_png_depth(depth_path)

                # Process optical flow if available.
                if framedata.flow_forward["path"]:
                    flow_forward = cv2.imread(flow_forward_path, cv2.IMREAD_UNCHANGED)
                    flow_forward_mask = cv2.imread(
                        flow_forward_mask_path, cv2.IMREAD_UNCHANGED
                    )
                    np.savez(
                        osp.join(out_fflow_dir, f"{timestamp}.npz"),
                        flow=flow_forward,
                        mask=flow_forward_mask,
                    )
                if framedata.flow_backward["path"]:
                    flow_backward = cv2.imread(flow_backward_path, cv2.IMREAD_UNCHANGED)
                    flow_backward_mask = cv2.imread(
                        flow_backward_mask_path, cv2.IMREAD_UNCHANGED
                    )
                    np.savez(
                        osp.join(out_bflow_dir, f"{timestamp}.npz"),
                        flow=flow_backward,
                        mask=flow_backward_mask,
                    )

                # Get camera parameters.
                R, t, focal, pp = _get_pytorch3d_camera(
                    viewpoint, framedata.image.size, scale=1.0
                )
                intrinsics = np.eye(3)
                intrinsics[0, 0] = focal[0].item()
                intrinsics[1, 1] = focal[1].item()
                intrinsics[0, 2] = pp[0].item()
                intrinsics[1, 2] = pp[1].item()
                pose = np.eye(4)
                # Invert the camera pose.
                pose[:3, :3] = R.numpy().T
                pose[:3, 3] = -R.numpy().T @ t.numpy()

                # Define output file paths.
                out_img_path = osp.join(out_img_dir, f"{timestamp}.png")
                out_depth_path = osp.join(out_depth_dir, f"{timestamp}.npy")
                out_cam_path = osp.join(out_cam_dir, f"{timestamp}.npz")

                # Copy RGB image.
                shutil.copy(im_path, out_img_path)
                # Save depth.
                np.save(out_depth_path, depth)
                # Save camera metadata.
                np.savez(out_cam_path, intrinsics=intrinsics, pose=pose)
    # (Optionally, you could return some summary information.)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Dynamic Replica dataset: convert raw annotations, images, "
        "depth, and flow files to a processed format."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory of the Dynamic Replica data.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for processed data.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=SPLITS,
        help="List of splits to process (default: train valid test).",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=cpu_count(),
        help="Number of processes to use (default: number of CPU cores).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tasks = [(split, args.root_dir, args.out_dir) for split in args.splits]

    print("Processing splits:", args.splits)
    with Pool(processes=args.num_processes) as pool:
        list(
            tqdm(
                pool.imap(process_split_data, tasks),
                total=len(tasks),
                desc="Overall Progress",
            )
        )


if __name__ == "__main__":
    main()
