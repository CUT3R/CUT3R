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
        "--tartanair_dir",
        default="data/tartanair",
    )
    parser.add_argument(
        "--output_dir",
        default="data/mast3r_data/processed_tartanair",
    )
    return parser


def main(rootdir, outdir):
    os.makedirs(outdir, exist_ok=True)
    envs = [
        f for f in sorted(os.listdir(rootdir)) if os.path.isdir(osp.join(rootdir, f))
    ]
    for env in tqdm(envs):
        for difficulty in ["Easy", "Hard"]:
            subscenes = [
                f
                for f in os.listdir(osp.join(rootdir, env, difficulty))
                if os.path.isdir(osp.join(rootdir, env, difficulty, f))
            ]
            for subscene in tqdm(subscenes):
                frame_dir = osp.join(rootdir, env, difficulty, subscene)
                rgb_dir = osp.join(frame_dir, "image_left")
                depth_dir = osp.join(frame_dir, "depth_left")
                flow_dir = osp.join(frame_dir, "flow")
                intrinsics = np.array(
                    [[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]]
                ).astype(np.float32)
                poses = np.loadtxt(osp.join(frame_dir, "pose_left.txt"))
                frame_num = len(poses)
                os.makedirs(osp.join(outdir, env, difficulty, subscene), exist_ok=True)
                assert (
                    len(os.listdir(rgb_dir))
                    == len(os.listdir(depth_dir))
                    == len(os.listdir(flow_dir)) // 2 + 1
                    == frame_num
                )
                for i in tqdm(range(frame_num)):
                    rgb_path = osp.join(rgb_dir, f"{i:06d}_left.png")
                    out_rgb_path = osp.join(
                        outdir, env, difficulty, subscene, f"{i:06d}_rgb.png"
                    )
                    depth_path = osp.join(depth_dir, f"{i:06d}_left_depth.npy")
                    out_depth_path = osp.join(
                        outdir, env, difficulty, subscene, f"{i:06d}_depth.npy"
                    )
                    if i < frame_num - 1:
                        fflow_path = osp.join(flow_dir, f"{i:06d}_{i+1:06d}_flow.npy")
                        mask_path = osp.join(flow_dir, f"{i:06d}_{i+1:06d}_mask.npy")
                    else:
                        fflow_path = None
                        mask_path = None
                    out_fflow_path = (
                        osp.join(outdir, env, difficulty, subscene, f"{i:06d}_flow.npy")
                        if fflow_path is not None
                        else None
                    )
                    out_mask_path = (
                        osp.join(outdir, env, difficulty, subscene, f"{i:06d}_mask.npy")
                        if mask_path is not None
                        else None
                    )
                    pose = poses[i]
                    x, y, z, qx, qy, qz, qw = pose
                    rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
                    c2w = np.eye(4)
                    c2w[:3, :3] = rotation
                    c2w[:3, 3] = [x, y, z]
                    w2c = np.linalg.inv(c2w)
                    w2c = w2c[[1, 2, 0, 3]]
                    c2w = np.linalg.inv(w2c)
                    K = intrinsics
                    # copy
                    shutil.copy(rgb_path, out_rgb_path)
                    shutil.copy(depth_path, out_depth_path)
                    if fflow_path is not None:
                        shutil.copy(fflow_path, out_fflow_path)
                    if mask_path is not None:
                        shutil.copy(mask_path, out_mask_path)
                    np.savez(
                        osp.join(outdir, env, difficulty, subscene, f"{i:06d}_cam.npz"),
                        camera_pose=c2w.astype(np.float32),
                        camera_intrinsics=K.astype(np.float32),
                    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.tartanair_dir, args.output_dir)
