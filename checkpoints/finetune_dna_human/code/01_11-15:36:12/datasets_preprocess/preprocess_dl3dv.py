import argparse
import random
import gzip
import json
import os
import sys

import os.path as osp

import torch
import PIL.Image
from PIL import Image
import numpy as np
import cv2

from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
from read_write_model import run

import torch
import torchvision


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dl3dv_dir", default="../DL3DV-Dense/3K/")  # TODO
    parser.add_argument("--output_dir", default="../processed_dl3dv/3K/")  # TODO
    return parser


from scipy.spatial.transform import Rotation as R


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def main(rootdir, outdir):
    os.makedirs(outdir, exist_ok=True)

    envs = [f for f in os.listdir(rootdir) if os.path.isdir(osp.join(rootdir, f))]
    for env in tqdm(envs):
        subseqs = [
            f
            for f in os.listdir(osp.join(rootdir, env))
            if os.path.isdir(osp.join(rootdir, env, f)) and f.startswith("dense")
        ]
        for subseq in subseqs:
            sparse_dir = osp.join(rootdir, env, subseq, "sparse")
            images_dir = osp.join(rootdir, env, subseq, "images")
            # depth_dir = osp.join(rootdir, env, subseq, "stereo", "depth_maps")
            if (
                (not os.path.exists(sparse_dir))
                or (not os.path.exists(images_dir))
                # or (not os.path.exists(depth_dir))
            ):
                continue
            intrins_file = sparse_dir + "/cameras.txt"
            poses_file = sparse_dir + "/images.txt"
            if os.path.exists(intrins_file) and os.path.exists(poses_file):
                continue
            run(sparse_dir, sparse_dir)

            cam_params = {}
            with open(intrins_file, "r") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split()
                    if len(parts) == 0:
                        continue
                    cam_id = int(parts[0])
                    fx = float(parts[4])
                    fy = float(parts[5])
                    cx = float(parts[6])
                    cy = float(parts[7])
                    cam_params[cam_id] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            poses = []
            images = []
            intrinsics = []

            with open(poses_file, "r") as f:
                for i, line in enumerate(f):
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split()
                    if len(parts) == 0:
                        continue
                    if "." in parts[0]:
                        continue

                    img_name = parts[-1]
                    w, x, y, z = map(float, parts[1:5])
                    R = np.array(
                        [
                            [
                                1 - 2 * y * y - 2 * z * z,
                                2 * x * y - 2 * z * w,
                                2 * x * z + 2 * y * w,
                            ],
                            [
                                2 * x * y + 2 * z * w,
                                1 - 2 * x * x - 2 * z * z,
                                2 * y * z - 2 * x * w,
                            ],
                            [
                                2 * x * z - 2 * y * w,
                                2 * y * z + 2 * x * w,
                                1 - 2 * x * x - 2 * y * y,
                            ],
                        ]
                    )
                    tx, ty, tz = map(float, parts[5:8])
                    cam_id = int(parts[-2])
                    pose = np.eye(4)
                    pose[:3, :3] = R
                    pose[:3, 3] = [tx, ty, tz]
                    poses.append(np.linalg.inv(pose))
                    images.append(img_name)
                    intrinsics.append(cam_params[cam_id])

            os.makedirs(osp.join(outdir, env, subseq), exist_ok=True)
            os.makedirs(osp.join(outdir, env, subseq, "rgb"), exist_ok=True)
            # os.makedirs(osp.join(outdir, env, subseq, "depth"), exist_ok=True)
            os.makedirs(osp.join(outdir, env, subseq, "cam"), exist_ok=True)

            for i, img_name in enumerate(tqdm(images)):
                basename = img_name.split("/")[-1]
                if os.path.exists(
                    osp.join(
                        outdir, env, subseq, "cam", basename.replace(".png", ".npz")
                    )
                ):
                    print("Exist!")
                    continue
                img_path = os.path.join(images_dir, img_name)
                # depth_path = os.path.join(depth_dir, img_name + ".geometric.bin")
                if not os.path.exists(depth_path) or not os.path.exists(img_path):
                    continue
                try:
                    rgb = Image.open(img_path)
                    # depth = read_array(depth_path)
                except:
                    continue
                intrinsic = intrinsics[i]
                pose = poses[i]

                # save all

                rgb.save(osp.join(outdir, env, subseq, "rgb", basename))
                # np.save(
                #     osp.join(
                #         outdir, env, subseq, "depth", basename.replace(".png", ".npy")
                #     ),
                #     depth,
                # )
                np.savez(
                    osp.join(
                        outdir, env, subseq, "cam", basename.replace(".png", ".npz")
                    ),
                    intrinsic=intrinsic,
                    pose=pose,
                )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.dl3dv_dir, args.output_dir)
