#!/usr/bin/env python3
"""
Preprocess the Hypersim dataset.

This script reads camera parameters from a CSV file, converts an OpenGL-style
projection matrix into a camera intrinsic matrix, applies tone mapping, and
saves processed RGB images, depth maps, and camera metadata into an output
directory. Processing is done per scene and per camera view.

Usage:
    python preprocess_hypersim.py --hypersim_dir /path/to/hypersim \
                                  --output_dir /path/to/processed_hypersim
"""

import argparse
import os
import shutil
import time

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Ensure OpenEXR support for OpenCV.
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def get_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess the Hypersim dataset by converting projection "
        "matrices, applying tone mapping, and saving processed outputs."
    )
    parser.add_argument(
        "--hypersim_dir",
        default="/path/to/hypersim",
        help="Root directory of the Hypersim dataset.",
    )
    parser.add_argument(
        "--output_dir",
        default="/path/to/processed_hypersim",
        help="Output directory for processed Hypersim data.",
    )
    return parser


def opengl_to_intrinsics(proj_matrix, width_pixels, height_pixels):
    # Extract parameters from the projection matrix.
    K00 = proj_matrix[0, 0] * width_pixels / 2.0
    K01 = -proj_matrix[0, 1] * width_pixels / 2.0
    K02 = (1.0 - proj_matrix[0, 2]) * width_pixels / 2.0
    K11 = proj_matrix[1, 1] * height_pixels / 2.0
    K12 = (1.0 + proj_matrix[1, 2]) * height_pixels / 2.0
    return np.array([[K00, K01, K02], [0.0, K11, K12], [0.0, 0.0, 1.0]])


def process_scene(args):
    rootdir, outdir, scene_name = args
    scene_outdir = os.path.join(outdir, scene_name)
    os.makedirs(scene_outdir, exist_ok=True)
    seq_dir = os.path.join(rootdir, scene_name)
    seq_detail_dir = os.path.join(seq_dir, "_detail")
    seq_images_dir = os.path.join(seq_dir, "images")

    # Read global camera parameters from the CSV file.
    all_metafile = os.path.join(rootdir, "metadata_camera_parameters.csv")
    df_camera_parameters = pd.read_csv(all_metafile, index_col="scene_name")
    df_ = df_camera_parameters.loc[scene_name]

    width_pixels = int(df_["settings_output_img_width"])
    height_pixels = int(df_["settings_output_img_height"])

    M_proj = np.array(
        [
            [df_["M_proj_00"], df_["M_proj_01"], df_["M_proj_02"], df_["M_proj_03"]],
            [df_["M_proj_10"], df_["M_proj_11"], df_["M_proj_12"], df_["M_proj_13"]],
            [df_["M_proj_20"], df_["M_proj_21"], df_["M_proj_22"], df_["M_proj_23"]],
            [df_["M_proj_30"], df_["M_proj_31"], df_["M_proj_32"], df_["M_proj_33"]],
        ]
    )

    camera_intrinsics = opengl_to_intrinsics(
        M_proj, width_pixels, height_pixels
    ).astype(np.float32)
    if camera_intrinsics[0, 1] != 0:
        print(f"camera_intrinsics[0, 1] != 0: {camera_intrinsics[0, 1]}")
        return

    # Read world scale and camera IDs.
    worldscale = (
        pd.read_csv(
            os.path.join(seq_detail_dir, "metadata_scene.csv"),
            index_col="parameter_name",
        )
        .to_numpy()
        .flatten()[0]
        .astype(np.float32)
    )
    camera_ids = (
        pd.read_csv(
            os.path.join(seq_detail_dir, "metadata_cameras.csv"),
            header=None,
            skiprows=1,
        )
        .to_numpy()
        .flatten()
    )

    # Tone mapping parameters.
    gamma = 1.0 / 2.2  # Standard gamma correction exponent.
    inv_gamma = 1.0 / gamma
    percentile = 90  # Desired percentile brightness in the unmodified image.
    brightness_nth_percentile_desired = 0.8  # Desired brightness after scaling.

    for camera_id in camera_ids:
        subscene_dir = os.path.join(scene_outdir, f"{camera_id}")
        os.makedirs(subscene_dir, exist_ok=True)
        camera_dir = os.path.join(seq_detail_dir, camera_id)
        if not os.path.exists(camera_dir):
            print(f"{camera_dir} does not exist.")
            continue
        color_dir = os.path.join(seq_images_dir, f"scene_{camera_id}_final_hdf5")
        geometry_dir = os.path.join(seq_images_dir, f"scene_{camera_id}_geometry_hdf5")
        if not (os.path.exists(color_dir) and os.path.exists(geometry_dir)):
            print(f"{color_dir} or {geometry_dir} does not exist.")
            continue

        camera_positions_hdf5_file = os.path.join(
            camera_dir, "camera_keyframe_positions.hdf5"
        )
        camera_orientations_hdf5_file = os.path.join(
            camera_dir, "camera_keyframe_orientations.hdf5"
        )

        with h5py.File(camera_positions_hdf5_file, "r") as f:
            camera_positions = f["dataset"][:]
        with h5py.File(camera_orientations_hdf5_file, "r") as f:
            camera_orientations = f["dataset"][:]

        assert len(camera_positions) == len(
            camera_orientations
        ), f"len(camera_positions)={len(camera_positions)} != len(camera_orientations)={len(camera_orientations)}"

        rgbs = sorted([f for f in os.listdir(color_dir) if f.endswith(".color.hdf5")])
        depths = sorted(
            [f for f in os.listdir(geometry_dir) if f.endswith(".depth_meters.hdf5")]
        )
        assert len(rgbs) == len(
            depths
        ), f"len(rgbs)={len(rgbs)} != len(depths)={len(depths)}"
        exist_frame_ids = [int(f.split(".")[1]) for f in rgbs]
        valid_camera_positions = camera_positions[exist_frame_ids]
        valid_camera_orientations = camera_orientations[exist_frame_ids]

        for i, (rgb, depth) in enumerate(tqdm(zip(rgbs, depths), total=len(rgbs))):
            frame_id = int(rgb.split(".")[1])
            assert frame_id == int(
                depth.split(".")[1]
            ), f"frame_id={frame_id} != {int(depth.split('.')[1])}"
            # Tone mapping.
            render_entity = os.path.join(
                geometry_dir,
                depth.replace("depth_meters.hdf5", "render_entity_id.hdf5"),
            )
            with h5py.File(os.path.join(color_dir, rgb), "r") as f:
                color = f["dataset"][:]
            with h5py.File(os.path.join(geometry_dir, depth), "r") as f:
                distance = f["dataset"][:]
            R_cam2world = valid_camera_orientations[i]
            R_cam2world = R_cam2world @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            t_cam2world = valid_camera_positions[i] * worldscale
            T_cam2world = np.eye(4)
            T_cam2world[:3, :3] = R_cam2world
            T_cam2world[:3, 3] = t_cam2world

            if not np.isfinite(T_cam2world).all():
                print(f"frame_id={frame_id} T_cam2world is not finite.")
                continue

            focal = (camera_intrinsics[0, 0] + camera_intrinsics[1, 1]) / 2.0
            ImageplaneX = (
                np.linspace(
                    (-0.5 * width_pixels) + 0.5,
                    (0.5 * width_pixels) - 0.5,
                    width_pixels,
                )
                .reshape(1, width_pixels)
                .repeat(height_pixels, 0)
                .astype(np.float32)[:, :, None]
            )
            ImageplaneY = (
                np.linspace(
                    (-0.5 * height_pixels) + 0.5,
                    (0.5 * height_pixels) - 0.5,
                    height_pixels,
                )
                .reshape(height_pixels, 1)
                .repeat(width_pixels, 1)
                .astype(np.float32)[:, :, None]
            )
            ImageplaneZ = np.full([height_pixels, width_pixels, 1], focal, np.float32)
            Imageplane = np.concatenate([ImageplaneX, ImageplaneY, ImageplaneZ], axis=2)

            depth = distance / np.linalg.norm(Imageplane, axis=2) * focal

            with h5py.File(render_entity, "r") as f:
                render_entity_id = f["dataset"][:].astype(np.int32)
            assert (render_entity_id != 0).all()
            valid_mask = render_entity_id != -1

            if np.sum(valid_mask) == 0:
                scale = 1.0  # If there are no valid pixels, set scale to 1.0.
            else:
                brightness = (
                    0.3 * color[:, :, 0] + 0.59 * color[:, :, 1] + 0.11 * color[:, :, 2]
                )
                brightness_valid = brightness[valid_mask]
                eps = 0.0001  # Avoid division by zero.
                brightness_nth_percentile_current = np.percentile(
                    brightness_valid, percentile
                )
                if brightness_nth_percentile_current < eps:
                    scale = 0.0
                else:
                    scale = (
                        np.power(brightness_nth_percentile_desired, inv_gamma)
                        / brightness_nth_percentile_current
                    )

            color = np.power(np.maximum(scale * color, 0), gamma)
            color = np.clip(color, 0.0, 1.0)

            out_rgb_path = os.path.join(subscene_dir, f"{frame_id:06d}_rgb.png")
            Image.fromarray((color * 255).astype(np.uint8)).save(out_rgb_path)
            out_depth_path = os.path.join(subscene_dir, f"{frame_id:06d}_depth.npy")
            np.save(out_depth_path, depth.astype(np.float32))
            out_cam_path = os.path.join(subscene_dir, f"{frame_id:06d}_cam.npz")
            np.savez(
                out_cam_path,
                intrinsics=camera_intrinsics,
                pose=T_cam2world.astype(np.float32),
            )


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Use placeholder paths to avoid personal/private information.
    rootdir = args.hypersim_dir  # e.g., '/path/to/hypersim'
    outdir = args.output_dir  # e.g., '/path/to/processed_hypersim'
    os.makedirs(outdir, exist_ok=True)

    import multiprocessing

    scenes = sorted(
        [f for f in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir, f))]
    )
    # Process each scene sequentially (or use multiprocessing if desired)
    for scene in scenes:
        process_scene((rootdir, outdir, scene))


if __name__ == "__main__":
    main()
