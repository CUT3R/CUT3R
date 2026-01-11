#!/usr/bin/env python3
"""
Process Bedlam scenes by computing camera intrinsics and extrinsics
from extracted data. The script reads per-scene CSV and image/depth files,
computes the necessary camera parameters, and saves the resulting camera
files (as .npz files) in an output directory.

Usage:
    python preprocess_bedlam.py --root /path/to/extracted_data \
                             --outdir /path/to/processed_bedlam \
                             [--num_workers 4]
"""

import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
import shutil
import OpenEXR  # Ensure OpenEXR is installed
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse

# Enable OpenEXR support in OpenCV.
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Global constants
IMG_FORMAT = ".png"
rotate_flag = False
SENSOR_W = 36
SENSOR_H = 20.25
IMG_W = 1280
IMG_H = 720

# -----------------------------------------------------------------------------
# Helper functions for camera parameter conversion
# -----------------------------------------------------------------------------


def focalLength_mm2px(focalLength, dslr_sens, focalPoint):
    focal_pixel = (focalLength / dslr_sens) * focalPoint * 2
    return focal_pixel


def get_cam_int(fl, sens_w, sens_h, cx, cy):
    flx = focalLength_mm2px(fl, sens_w, cx)
    fly = focalLength_mm2px(fl, sens_h, cy)
    cam_mat = np.array([[flx, 0, cx], [0, fly, cy], [0, 0, 1]])
    return cam_mat


def unreal2cv2(points):
    # Permute coordinates: x --> y, y --> z, z --> x
    points = np.roll(points, 2, axis=1)
    # Invert the y-axis
    points = points * np.array([1.0, -1.0, 1.0])
    return points


def get_cam_trans(body_trans, cam_trans):
    cam_trans = np.array(cam_trans) / 100
    cam_trans = unreal2cv2(np.reshape(cam_trans, (1, 3)))
    body_trans = np.array(body_trans) / 100
    body_trans = unreal2cv2(np.reshape(body_trans, (1, 3)))
    trans = body_trans - cam_trans
    return trans


def get_cam_rotmat(pitch, yaw, roll):
    rotmat_yaw, _ = cv2.Rodrigues(np.array([[0, (yaw / 180) * np.pi, 0]], dtype=float))
    rotmat_pitch, _ = cv2.Rodrigues(np.array([pitch / 180 * np.pi, 0, 0]).reshape(3, 1))
    rotmat_roll, _ = cv2.Rodrigues(np.array([0, 0, roll / 180 * np.pi]).reshape(3, 1))
    final_rotmat = rotmat_roll @ (rotmat_pitch @ rotmat_yaw)
    return final_rotmat


def get_global_orient(cam_pitch, cam_yaw, cam_roll):
    pitch_rotmat, _ = cv2.Rodrigues(
        np.array([cam_pitch / 180 * np.pi, 0, 0]).reshape(3, 1)
    )
    roll_rotmat, _ = cv2.Rodrigues(
        np.array([0, 0, cam_roll / 180 * np.pi]).reshape(3, 1)
    )
    final_rotmat = roll_rotmat @ pitch_rotmat
    return final_rotmat


def convert_translation_to_opencv(x, y, z):
    t_cv = np.array([y, -z, x])
    return t_cv


def rotation_matrix_unreal(yaw, pitch, roll):
    yaw_rad = np.deg2rad(yaw)
    pitch_rad = np.deg2rad(pitch)
    roll_rad = np.deg2rad(roll)
    # Yaw (left-handed)
    R_yaw = np.array(
        [
            [np.cos(-yaw_rad), -np.sin(-yaw_rad), 0],
            [np.sin(-yaw_rad), np.cos(-yaw_rad), 0],
            [0, 0, 1],
        ]
    )
    # Pitch (right-handed)
    R_pitch = np.array(
        [
            [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
            [0, 1, 0],
            [-np.sin(pitch_rad), 0, np.cos(pitch_rad)],
        ]
    )
    # Roll (right-handed)
    R_roll = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0, np.sin(roll_rad), np.cos(roll_rad)],
        ]
    )
    R_unreal = R_roll @ R_pitch @ R_yaw
    return R_unreal


def convert_rotation_to_opencv(R_unreal):
    # Transformation matrix from Unreal to OpenCV coordinate system.
    C = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]])
    R_cv = C @ R_unreal @ C.T
    return R_cv


def get_rot_unreal(yaw, pitch, roll):
    yaw_rad = np.deg2rad(yaw)
    pitch_rad = np.deg2rad(pitch)
    roll_rad = np.deg2rad(roll)
    R_yaw = np.array(
        [
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1],
        ]
    )
    R_pitch = np.array(
        [
            [np.cos(pitch_rad), 0, -np.sin(pitch_rad)],
            [0, 1, 0],
            [np.sin(pitch_rad), 0, np.cos(pitch_rad)],
        ]
    )
    R_roll = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll_rad), np.sin(roll_rad)],
            [0, -np.sin(roll_rad), np.cos(roll_rad)],
        ]
    )
    R_unreal = R_yaw @ R_pitch @ R_roll
    return R_unreal


def get_extrinsics_unreal(R_unreal, t_unreal):
    cam_trans = np.array(t_unreal)
    ext = np.eye(4)
    ext[:3, :3] = R_unreal
    ext[:3, 3] = cam_trans.reshape(1, 3)
    return ext


def get_extrinsics_opencv(yaw, pitch, roll, x, y, z):
    R_unreal = get_rot_unreal(yaw, pitch, roll)
    t_unreal = np.array([x / 100.0, y / 100.0, z / 100.0])
    T_u2wu = get_extrinsics_unreal(R_unreal, t_unreal)
    T_opencv2unreal = np.array(
        [[0, 0, -1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    T_wu2ou = np.array(
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    return np.linalg.inv(T_opencv2unreal @ T_u2wu @ T_wu2ou)


# -----------------------------------------------------------------------------
# Get camera parameters from the extracted images and CSV data.
# -----------------------------------------------------------------------------


def get_params(
    image_folder,
    fl,
    trans_body,
    cam_x,
    cam_y,
    cam_z,
    fps,
    cam_pitch_,
    cam_roll_,
    cam_yaw_,
):
    all_images = sorted(glob(os.path.join(image_folder, "*" + IMG_FORMAT)))
    imgnames, cam_ext, cam_int = [], [], []

    for img_ind, image_path in enumerate(all_images):
        # Process every 5th frame.
        if img_ind % 5 != 0:
            continue
        cam_ind = img_ind

        cam_pitch_ind = cam_pitch_[cam_ind]
        cam_yaw_ind = cam_yaw_[cam_ind]
        cam_roll_ind = cam_roll_[cam_ind]

        CAM_INT = get_cam_int(fl[cam_ind], SENSOR_W, SENSOR_H, IMG_W / 2.0, IMG_H / 2.0)

        rot_unreal = rotation_matrix_unreal(cam_yaw_ind, cam_pitch_ind, cam_roll_ind)
        rot_cv = convert_rotation_to_opencv(rot_unreal)
        trans_cv = convert_translation_to_opencv(
            cam_x[cam_ind] / 100.0, cam_y[cam_ind] / 100.0, cam_z[cam_ind] / 100.0
        )
        cam_ext_ = np.eye(4)
        cam_ext_[:3, :3] = rot_cv
        # The camera pose is computed as the inverse of the transformed translation.
        cam_ext_[:3, 3] = -rot_cv @ trans_cv

        imgnames.append(
            os.path.join(image_path.split("/")[-2], image_path.split("/")[-1])
        )
        cam_ext.append(cam_ext_)
        cam_int.append(CAM_INT)
    return imgnames, cam_ext, cam_int


# -----------------------------------------------------------------------------
# Processing per sequence.
# -----------------------------------------------------------------------------


def process_seq(args):
    """
    Process a single sequence task. For each image, load the corresponding
    depth and image files, and save the computed camera intrinsics and the inverse
    of the extrinsic matrix (i.e. the camera pose in world coordinates) as an NPZ file.
    """
    (
        scene,
        seq_name,
        outdir,
        image_folder_base,
        depth_folder_base,
        imgnames,
        cam_ext,
        cam_int,
    ) = args

    out_rgb_dir = os.path.join(outdir, '_'.join([scene, seq_name]), 'rgb')
    out_depth_dir = os.path.join(outdir, '_'.join([scene, seq_name]), 'depth')
    out_cam_dir = os.path.join(outdir, "_".join([scene, seq_name]), "cam")
    os.makedirs(out_rgb_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)
    os.makedirs(out_cam_dir, exist_ok=True)

    assert (
        len(imgnames) == len(cam_ext) == len(cam_int)
    ), f"Inconsistent lengths for {scene}_{seq_name}"
    for imgname, ext, intr in zip(imgnames, cam_ext, cam_int):
        depthname = imgname.replace(".png", "_depth.exr")
        imgpath = os.path.join(image_folder_base, imgname)
        depthpath = os.path.join(depth_folder_base, depthname)
        depth= OpenEXR.File(depthpath).parts[0].channels['Depth'].pixels
        depth = depth.astype(np.float32)/100.0
        
        outimg_path = os.path.join(out_rgb_dir, os.path.basename(imgpath))
        outdepth_path = os.path.join(out_depth_dir, os.path.basename(imgpath).replace('.png','.npy'))
        outcam_path = os.path.join(
            out_cam_dir, os.path.basename(imgpath).replace(".png", ".npz")
        )

        shutil.copy(imgpath, outimg_path)
        np.save(outdepth_path, depth)
        np.savez(outcam_path, intrinsics=intr, pose=np.linalg.inv(ext))
    return None


# -----------------------------------------------------------------------------
# Main entry point.
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Process Bedlam scenes: compute camera intrinsics and extrinsics, "
        "and save processed camera files."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory of the extracted data (scenes).",
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Output directory for processed data."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes (default: os.cpu_count()//2).",
    )
    args = parser.parse_args()

    root = args.root
    outdir = args.outdir
    num_workers = (
        args.num_workers if args.num_workers is not None else (os.cpu_count() or 4) // 2
    )

    # Get scene directories from the root folder.
    scenes = sorted(
        [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    )
    # Exclude HDRI scenes.
    hdri_scenes = [
        "20221010_3_1000_batch01hand",
        "20221017_3_1000_batch01hand",
        "20221018_3-8_250_batch01hand",
        "20221019_3_250_highbmihand",
    ]
    scenes = np.setdiff1d(scenes, hdri_scenes)

    tasks = []
    for scene in tqdm(scenes, desc="Collecting tasks"):
        # Skip closeup scenes.
        if "closeup" in scene:
            continue
        base_folder = os.path.join(root, scene)
        image_folder_base = os.path.join(root, scene, "png")
        depth_folder_base = os.path.join(root, scene, "depth")
        csv_path = os.path.join(base_folder, "be_seq.csv")
        if not os.path.exists(csv_path):
            continue
        csv_data = pd.read_csv(csv_path)
        csv_data = csv_data.to_dict("list")
        cam_csv_base = os.path.join(base_folder, "ground_truth", "camera")

        # Look for a row in the CSV with a "sequence_name" comment.
        for idx, comment in enumerate(csv_data.get("Comment", [])):
            if "sequence_name" in comment:
                seq_name = comment.split(";")[0].split("=")[-1]
                cam_csv_path = os.path.join(cam_csv_base, seq_name + "_camera.csv")
                if not os.path.exists(cam_csv_path):
                    continue
                cam_csv_data = pd.read_csv(cam_csv_path)
                cam_csv_data = cam_csv_data.to_dict("list")
                cam_x = cam_csv_data["x"]
                cam_y = cam_csv_data["y"]
                cam_z = cam_csv_data["z"]
                cam_yaw_ = cam_csv_data["yaw"]
                cam_pitch_ = cam_csv_data["pitch"]
                cam_roll_ = cam_csv_data["roll"]
                fl = cam_csv_data["focal_length"]
                image_folder = os.path.join(image_folder_base, seq_name)
                trans_body = None  # Not used here.
                imgnames, cam_ext, cam_int = get_params(
                    image_folder,
                    fl,
                    trans_body,
                    cam_x,
                    cam_y,
                    cam_z,
                    6,
                    cam_pitch_=cam_pitch_,
                    cam_roll_=cam_roll_,
                    cam_yaw_=cam_yaw_,
                )
                tasks.append(
                    (
                        scene,
                        seq_name,
                        outdir,
                        image_folder_base,
                        depth_folder_base,
                        imgnames,
                        cam_ext,
                        cam_int,
                    )
                )
                # Process only the first valid sequence for this scene.
                break

    # Process each task in parallel.
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_seq, task): task for task in tasks}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing sequences"
        ):
            error = future.result()
            if error:
                print(error)


if __name__ == "__main__":
    main()
