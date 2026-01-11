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
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
import path_to_root  # noqa
import datasets_preprocess.utils.cropping as cropping  # noqa


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scannet_dir", default="data/data_scannet")
    parser.add_argument("--output_dir", default="data/dust3r_data/processed_scannet")
    return parser


def process_scene(args):
    rootdir, outdir, split, scene = args
    frame_dir = osp.join(rootdir, split, scene)
    rgb_dir = osp.join(frame_dir, "color")
    depth_dir = osp.join(frame_dir, "depth")
    pose_dir = osp.join(frame_dir, "pose")
    depth_intrinsic = np.loadtxt(
        osp.join(frame_dir, "intrinsic", "intrinsic_depth.txt")
    )[:3, :3].astype(np.float32)
    color_intrinsic = np.loadtxt(
        osp.join(frame_dir, "intrinsic", "intrinsic_color.txt")
    )[:3, :3].astype(np.float32)
    if not np.isfinite(depth_intrinsic).all() or not np.isfinite(color_intrinsic).all():
        return
    os.makedirs(osp.join(outdir, split, scene), exist_ok=True)
    frame_num = len(os.listdir(rgb_dir))
    assert frame_num == len(os.listdir(depth_dir)) == len(os.listdir(pose_dir))
    out_rgb_dir = osp.join(outdir, split, scene, "color")
    out_depth_dir = osp.join(outdir, split, scene, "depth")
    out_cam_dir = osp.join(outdir, split, scene, "cam")

    os.makedirs(out_rgb_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)
    os.makedirs(out_cam_dir, exist_ok=True)
    for i in tqdm(range(frame_num)):
        rgb_path = osp.join(rgb_dir, f"{i}.jpg")
        depth_path = osp.join(depth_dir, f"{i}.png")
        pose_path = osp.join(pose_dir, f"{i}.txt")

        rgb = Image.open(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        rgb = rgb.resize(depth.shape[::-1], resample=Image.Resampling.LANCZOS)
        pose = np.loadtxt(pose_path).reshape(4, 4).astype(np.float32)
        if not np.isfinite(pose).all():
            continue

        out_rgb_path = osp.join(out_rgb_dir, f"{i:05d}.jpg")
        out_depth_path = osp.join(out_depth_dir, f"{i:05d}.png")
        out_cam_path = osp.join(out_cam_dir, f"{i:05d}.npz")
        np.savez(out_cam_path, intrinsics=depth_intrinsic, pose=pose)
        rgb.save(out_rgb_path)
        cv2.imwrite(out_depth_path, depth)


def main(rootdir, outdir):
    os.makedirs(outdir, exist_ok=True)
    splits = ["scans_test", "scans_train"]
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    for split in splits:
        scenes = [
            f
            for f in os.listdir(os.path.join(rootdir, split))
            if os.path.isdir(osp.join(rootdir, split, f))
        ]
        pool.map(process_scene, [(rootdir, outdir, split, scene) for scene in scenes])
    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.scannet_dir, args.output_dir)
