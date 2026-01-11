import os


import os.path as osp

from PIL import Image
import numpy as np


from tqdm import tqdm
from read_write_model import run


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mapfree_dir", default="")  # TODO
    parser.add_argument("--output_dir", default="test_preprocess")  # TODO
    return parser


def main(rootdir, outdir):
    os.makedirs(outdir, exist_ok=True)

    envs = [f for f in os.listdir(rootdir) if os.path.isdir(osp.join(rootdir, f))]
    for env in tqdm(envs):
        subseqs = [
            f
            for f in os.listdir(osp.join(rootdir, env))
            if os.path.isdir(osp.join(rootdir, env, f))
        ]
        for subseq in subseqs:
            sparse_dir = osp.join(rootdir, env, subseq, "sparse")
            images_dir = osp.join(rootdir, env, subseq, "images")
            run(sparse_dir, sparse_dir)
            intrins_file = sparse_dir + "/cameras.txt"
            poses_file = sparse_dir + "/images.txt"

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
            os.makedirs(osp.join(outdir, env, subseq, "cam"), exist_ok=True)

            for i, img_name in enumerate(tqdm(images)):
                img_path = os.path.join(images_dir, img_name)
                rgb = Image.open(img_path)
                intrinsic = intrinsics[i]
                pose = poses[i]
                # save all
                basename = img_name.split("/")[-1]
                rgb.save(osp.join(outdir, env, subseq, "rgb", basename))
                np.savez(
                    osp.join(
                        outdir, env, subseq, "cam", basename.replace(".jpg", ".npz")
                    ),
                    intrinsic=intrinsic,
                    pose=pose,
                )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.mapfree_dir, args.output_dir)
