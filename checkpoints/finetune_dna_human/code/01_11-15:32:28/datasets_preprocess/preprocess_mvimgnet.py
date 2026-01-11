#!/usr/bin/env python3
"""
Preprocess the MVImgNet dataset.

This script processes MVImgNet sequences by:
  - Loading a sparse SFM reconstruction.
  - Undistorting and rescaling RGB images.
  - Converting COLMAP intrinsics between conventions.
  - Saving the processed images and camera metadata.

Usage:
  python preprocess_mvimgnet.py --data_dir /path/to/MVImgNet_data \
                                --pcd_dir /path/to/MVPNet \
                                --output_dir /path/to/processed_mvimgnet
"""

import os
import os.path as osp
import argparse
import numpy as np
import open3d as o3d
import pyrender
import PIL.Image as Image
import cv2
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import your custom SFM processing function.
from read_write_model import run  # Assumed to be available

# Try to set up resampling filters from PIL.
try:
    lanczos = Image.Resampling.LANCZOS
    bicubic = Image.Resampling.BICUBIC
except AttributeError:
    lanczos = Image.LANCZOS
    bicubic = Image.BICUBIC

# Conversion matrix from COLMAP (or OpenGL) to OpenCV conventions.
OPENGL_TO_OPENCV = np.float32(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
)


# -----------------------------------------------------------------------------
# Helper Classes and Functions
# -----------------------------------------------------------------------------
class ImageList:
    """Convenience class to apply operations to a list of images."""

    def __init__(self, images):
        if not isinstance(images, (list, tuple)):
            images = [images]
        self.images = []
        for image in images:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def to_pil(self):
        return tuple(self.images) if len(self.images) > 1 else self.images[0]

    @property
    def size(self):
        sizes = [im.size for im in self.images]
        assert all(s == sizes[0] for s in sizes)
        return sizes[0]

    def resize(self, *args, **kwargs):
        return ImageList([im.resize(*args, **kwargs) for im in self.images])

    def crop(self, *args, **kwargs):
        return ImageList([im.crop(*args, **kwargs) for im in self.images])


def colmap_to_opencv_intrinsics(K):
    """
    Convert COLMAP intrinsics (with pixel centers at (0.5, 0.5)) to OpenCV convention.
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def opencv_to_colmap_intrinsics(K):
    """
    Convert OpenCV intrinsics (with pixel centers at (0, 0)) to COLMAP convention.
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K


def rescale_image_depthmap(
    image, depthmap, camera_intrinsics, output_resolution, force=True
):
    """
    Jointly rescale an image (and its depthmap) so that the output resolution is at least the desired value.

    Args:
      image: Input image (as a PIL.Image or compatible object).
      depthmap: A corresponding depth map (or None).
      camera_intrinsics: A 3x3 NumPy array of intrinsics.
      output_resolution: (width, height) desired resolution.
      force: If True, always rescale even if the image is smaller.

    Returns:
      Tuple of (rescaled image, rescaled depthmap, updated intrinsics).
    """
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (W, H)
    output_resolution = np.array(output_resolution)
    if depthmap is not None:
        assert tuple(depthmap.shape[:2]) == image.size[::-1]
    scale_final = max(output_resolution / image.size) + 1e-8
    if scale_final >= 1 and not force:
        return image.to_pil(), depthmap, camera_intrinsics
    output_resolution = np.floor(input_resolution * scale_final).astype(int)
    image = image.resize(
        tuple(output_resolution), resample=lanczos if scale_final < 1 else bicubic
    )
    if depthmap is not None:
        depthmap = cv2.resize(
            depthmap, tuple(output_resolution), interpolation=cv2.INTER_NEAREST
        )
    camera_intrinsics = camera_matrix_of_crop(
        camera_intrinsics, input_resolution, output_resolution, scaling=scale_final
    )
    return image.to_pil(), depthmap, camera_intrinsics


def camera_matrix_of_crop(
    input_camera_matrix,
    input_resolution,
    output_resolution,
    scaling=1,
    offset_factor=0.5,
    offset=None,
):
    """
    Update the camera intrinsics to account for a rescaling (or cropping) of the image.
    """
    margins = np.asarray(input_resolution) * scaling - output_resolution
    assert np.all(margins >= 0.0)
    if offset is None:
        offset = offset_factor * margins
    output_camera_matrix_colmap = opencv_to_colmap_intrinsics(input_camera_matrix)
    output_camera_matrix_colmap[:2, :] *= scaling
    output_camera_matrix_colmap[:2, 2] -= offset
    output_camera_matrix = colmap_to_opencv_intrinsics(output_camera_matrix_colmap)
    return output_camera_matrix


def pose_from_qwxyz_txyz(elems):
    """
    Convert a quaternion (qw, qx, qy, qz) and translation (tx, ty, tz) to a 4x4 pose.
    Returns the inverse of the computed pose (i.e. cam2world).
    """
    from scipy.spatial.transform import Rotation

    qw, qx, qy, qz, tx, ty, tz = map(float, elems)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat((qx, qy, qz, qw)).as_matrix()
    pose[:3, 3] = (tx, ty, tz)
    return np.linalg.inv(pose)


def load_sfm(sfm_dir):
    """
    Load sparse SFM data from COLMAP output files.

    Returns a tuple (img_idx, img_infos) where:
      - img_idx: A dict mapping image filename to index.
      - img_infos: A dict of image information (including intrinsics, file path, and camera pose).
    """
    with open(osp.join(sfm_dir, "cameras.txt"), "r") as f:
        raw = f.read().splitlines()[3:]  # skip header
    intrinsics = {}
    for camera in raw:
        camera = camera.split(" ")
        intrinsics[int(camera[0])] = [camera[1]] + [float(x) for x in camera[2:]]
    with open(osp.join(sfm_dir, "images.txt"), "r") as f:
        raw = f.read().splitlines()
        raw = [line for line in raw if not line.startswith("#")]
    img_idx = {}
    img_infos = {}
    for image, points in zip(raw[0::2], raw[1::2]):
        image = image.split(" ")
        points = points.split(" ")
        idx = image[0]
        img_name = image[-1]
        assert img_name not in img_idx, f"Duplicate image: {img_name}"
        img_idx[img_name] = idx
        current_points2D = {
            int(i): (float(x), float(y))
            for i, x, y in zip(points[2::3], points[0::3], points[1::3])
            if i != "-1"
        }
        img_infos[idx] = dict(
            intrinsics=intrinsics[int(image[-2])],
            path=img_name,
            frame_id=img_name,
            cam_to_world=pose_from_qwxyz_txyz(image[1:-2]),
            sparse_pts2d=current_points2D,
        )
    return img_idx, img_infos


def undistort_images(intrinsics, rgb):
    """
    Given camera intrinsics (in COLMAP convention) and an RGB image, compute and return
    the corresponding OpenCV intrinsics along with the (unchanged) image.
    """
    width = int(intrinsics[1])
    height = int(intrinsics[2])
    fx = intrinsics[3]
    fy = intrinsics[4]
    cx = intrinsics[5]
    cy = intrinsics[6]
    K = np.zeros([3, 3])
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy
    K[2, 2] = 1
    return width, height, K, rgb


# -----------------------------------------------------------------------------
# Processing Functions
# -----------------------------------------------------------------------------
def process_sequence(category, obj, data_dir, output_dir):
    """
    Process a single sequence from MVImgNet.

    Steps:
      1. Load the point cloud (from the MVPNet directory) and create a mesh (using Pyrender) for visualization.
      2. Load the SFM reconstruction from COLMAP files.
      3. For each image in the SFM output:
         a. Load the image.
         b. Undistort and rescale it.
         c. Update the camera intrinsics.
         d. Save the processed image and camera metadata.
    """

    # Define directories.
    seq_dir = osp.join(data_dir, "MVImgNet_by_categories", category, obj[:-4])
    rgb_dir = osp.join(seq_dir, "images")
    sfm_dir = osp.join(seq_dir, "sparse", "0")

    output_scene_dir = osp.join(output_dir, f"{category}_{obj[:-4]}")
    output_rgb_dir = osp.join(output_scene_dir, "rgb")
    output_cam_dir = osp.join(output_scene_dir, "cam")
    os.makedirs(output_rgb_dir, exist_ok=True)
    os.makedirs(output_cam_dir, exist_ok=True)

    # Run custom SFM processing.
    run(sfm_dir, sfm_dir)
    img_idx, img_infos = load_sfm(sfm_dir)

    for imgname in img_idx:
        idx = img_idx[imgname]
        info = img_infos[idx]
        rgb_path = osp.join(rgb_dir, info["path"])
        if not osp.exists(rgb_path):
            continue
        rgb = np.array(Image.open(rgb_path))
        _, _, K, rgb = undistort_images(info["intrinsics"], rgb)
        intrinsics = colmap_to_opencv_intrinsics(K)
        # Rescale image to a target resolution (e.g., 640x480) preserving aspect ratio.
        image, _, intrinsics = rescale_image_depthmap(
            rgb, None, intrinsics, (640, int(640 * 3.0 / 4))
        )
        intrinsics = opencv_to_colmap_intrinsics(intrinsics)
        out_img_path = osp.join(output_rgb_dir, info["path"][:-3] + "jpg")
        image.save(out_img_path)
        out_cam_path = osp.join(output_cam_dir, info["path"][:-3] + "npz")
        np.savez(out_cam_path, intrinsics=intrinsics, pose=info["cam_to_world"])


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MVImgNet dataset: undistort, rescale images, and save camera parameters."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/path/to/MVImgNet_data",
        help="Directory containing MVImgNet data (images and point clouds).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/path/to/processed_mvimgnet",
        help="Directory where processed data will be saved.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    # Get list of categories.
    categories = sorted(
        [
            d
            for d in os.listdir(osp.join(data_dir, "MVImgNet_by_categories"))
            if osp.isdir(osp.join(data_dir, "MVImgNet_by_categories", d))
        ]
    )
    for cat in categories:
        objects = sorted(os.listdir(osp.join(data_dir, "MVImgNet_by_categories", cat)))
        for obj in objects:
            process_sequence(cat, obj, data_dir, output_dir)


if __name__ == "__main__":
    main()
