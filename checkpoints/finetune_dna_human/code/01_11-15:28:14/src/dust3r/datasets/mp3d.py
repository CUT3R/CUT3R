import os.path as osp
import os
import sys
import itertools

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
import cv2
import numpy as np

from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


class MP3D_Multi(BaseMultiViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = False
        self.is_metric = True
        super().__init__(*args, **kwargs)

        self.loaded_data = self._load_data()

    def _load_data(self):
        scenes = os.listdir(self.ROOT)
        offset = 0
        overlaps = {scene: [] for scene in scenes}
        scene_img_list = {scene: [] for scene in scenes}
        images = []

        j = 0
        for scene in scenes:
            scene_dir = osp.join(self.ROOT, scene)
            rgb_dir = osp.join(scene_dir, "rgb")
            basenames = sorted(
                [f[:-4] for f in os.listdir(rgb_dir) if f.endswith(".png")]
            )
            overlap = np.load(osp.join(scene_dir, "overlap.npy"))
            overlaps[scene] = overlap
            num_imgs = len(basenames)

            images.extend(
                [(scene, i, basename) for i, basename in enumerate(basenames)]
            )
            scene_img_list[scene] = np.arange(num_imgs) + offset
            offset += num_imgs
            j += 1

        self.scenes = scenes
        self.scene_img_list = scene_img_list
        self.images = images
        self.overlaps = overlaps

    def __len__(self):
        return len(self.images)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        num_views_posible = 0
        num_unique = num_views if not self.allow_repeat else max(num_views // 3, 3)
        while num_views_posible < num_unique - 1:
            scene, img_idx, _ = self.images[idx]
            overlap = self.overlaps[scene]
            sel_img_idx = np.where(overlap[:, 0] == img_idx)[0]
            overlap_sel = overlap[sel_img_idx]
            overlap_sel = overlap_sel[
                (overlap_sel[:, 2] > 0.01) * (overlap_sel[:, 2] < 1)
            ]
            num_views_posible = len(overlap_sel)
            if num_views_posible >= num_unique - 1:
                break
            idx = rng.choice(len(self.images))

        ref_id = self.scene_img_list[scene][img_idx]
        ids = self.scene_img_list[scene][overlap_sel[:, 1].astype(np.int64)]
        replace = False if not self.allow_repeat else True
        image_idxs = rng.choice(
            ids,
            num_views - 1,
            replace=replace,
            p=overlap_sel[:, 2] / np.sum(overlap_sel[:, 2]),
        )
        image_idxs = np.concatenate([[ref_id], image_idxs])

        ordered_video = False
        views = []
        for v, view_idx in enumerate(image_idxs):
            scene, _, basename = self.images[view_idx]
            scene_dir = osp.join(self.ROOT, scene)
            rgb_path = osp.join(scene_dir, "rgb", basename + ".png")
            depth_path = osp.join(scene_dir, "depth", basename + ".npy")
            cam_path = osp.join(scene_dir, "cam", basename + ".npz")

            rgb_image = imread_cv2(rgb_path, cv2.IMREAD_COLOR)
            depthmap = np.load(depth_path).astype(np.float32)
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            cam_file = np.load(cam_path)
            intrinsics = cam_file["intrinsics"]
            camera_pose = cam_file["pose"]

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            # generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.85, 0.1, 0.05]
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="mp3d",
                    label=scene + "_" + rgb_path,
                    instance=f"{str(idx)}_{str(view_idx)}",
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(0.99, dtype=np.float32),
                    img_mask=img_mask,
                    ray_mask=ray_mask,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                )
            )
        assert len(views) == num_views
        return views
