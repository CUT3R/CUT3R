import argparse
import random
import gzip
import json
import os
import os.path as osp
import torch
import PIL.Image
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
import src.dust3r.datasets.utils.cropping as cropping  # noqa
from scipy.spatial.transform import Rotation as R


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unreal4k_dir",
        default="",
    )
    parser.add_argument(
        "--output_dir",
        default="",
    )
    return parser


def parse_extrinsics(file_path):
    """
    Parse the extrinsics file to extract the intrinsics and pose matrices.

    Args:
    file_path (str): The path to the file containing the extrinsics data.

    Returns:
    tuple: A tuple containing the intrinsics matrix (3x3) and pose matrix (3x4).
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

        # Parse the intrinsics matrix
        intrinsics_data = list(map(float, lines[0].strip().split()))
        intrinsics_matrix = np.array(intrinsics_data).reshape(3, 3)

        # Parse the pose matrix
        cam2world = np.eye(4)
        pose_data = list(map(float, lines[1].strip().split()))
        pose_matrix = np.array(pose_data).reshape(3, 4)
        cam2world[:3] = pose_matrix
        cam2world = np.linalg.inv(cam2world)

        return intrinsics_matrix, cam2world


def main(rootdir, outdir):
    os.makedirs(outdir, exist_ok=True)
    envs = [
        f for f in sorted(os.listdir(rootdir)) if os.path.isdir(osp.join(rootdir, f))
    ]
    for env in tqdm(envs):
        subscenes = ["0", "1"]
        for subscene in tqdm(subscenes):
            frame_dir = osp.join(rootdir, env)
            rgb_dir = osp.join(frame_dir, f"Image{subscene}")
            disp_dir = osp.join(frame_dir, f"Disp{subscene}")
            ext_dir = osp.join(frame_dir, f"Extrinsics{subscene}")

            frame_num = len(os.listdir(rgb_dir))
            os.makedirs(osp.join(outdir, env, subscene), exist_ok=True)
            for i in tqdm(range(frame_num)):
                rgb_path = osp.join(rgb_dir, f"{i:05d}.png")
                out_rgb_path = osp.join(outdir, env, subscene, f"{i:05d}_rgb.png")
                disp_path = osp.join(disp_dir, f"{i:05d}.npy")
                out_depth_path = osp.join(outdir, env, subscene, f"{i:05d}_depth.npy")
                out_cam_path = osp.join(outdir, env, subscene, f"{i:05d}.npz")
                ext_path0 = osp.join(frame_dir, f"Extrinsics0", f"{i:05d}.txt")
                ext_path1 = osp.join(frame_dir, f"Extrinsics1", f"{i:05d}.txt")
                K0, c2w0 = parse_extrinsics(ext_path0)
                K1, c2w1 = parse_extrinsics(ext_path1)
                if subscene == "0":
                    K = K0
                    c2w = c2w0
                else:
                    K = K1
                    c2w = c2w1

                img = Image.open(rgb_path).convert("RGB")
                disp = np.load(disp_path).astype(np.float32)
                baseline = (np.linalg.inv(c2w0) @ c2w1)[0, 3]
                depth = baseline * K[0, 0] / disp

                image, depthmap, camera_intrinsics = cropping.rescale_image_depthmap(
                    img, depth, K, output_resolution=(512, 384)
                )

                image.save(out_rgb_path)
                np.save(out_depth_path, depthmap)
                np.savez(out_cam_path, intrinsics=camera_intrinsics, cam2world=c2w)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.unreal4k_dir, args.output_dir)
