import os.path as osp
import cv2
import numpy as np
import itertools
import os
import sys

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from tqdm import tqdm
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2

import re


def extract_number(filename):
    match = re.search(r"\d+", filename)
    if match:
        return int(match.group())
    return 0


class UASOL_Multi(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = 40
        super().__init__(*args, **kwargs)
        self.loaded_data = self._load_data()

    def _load_data(self):
        self.scenes = os.listdir(self.ROOT)

        offset = 0
        scenes = []
        sceneids = []
        scene_img_list = []
        images = []
        start_img_ids = []

        j = 0
        for scene in tqdm(self.scenes):
            scene_dir = osp.join(self.ROOT, scene)
            rgb_dir = osp.join(scene_dir, "rgb")
            basenames = sorted(
                [f[:-4] for f in os.listdir(rgb_dir) if f.endswith(".png")],
                key=extract_number,
            )
            num_imgs = len(basenames)
            img_ids = list(np.arange(num_imgs) + offset)
            # start_img_ids_ = img_ids[:-self.num_views+1]
            cut_off = (
                self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            )
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

            if num_imgs < cut_off:
                print(f"Skipping {scene}")
                continue

            start_img_ids.extend(start_img_ids_)
            sceneids.extend([j] * num_imgs)
            images.extend(basenames)
            scenes.append(scene)
            scene_img_list.append(img_ids)

            # offset groups
            offset += num_imgs
            j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        start_id = self.start_img_ids[idx]
        all_image_ids = self.scene_img_list[self.sceneids[start_id]]
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_image_ids,
            rng,
            max_interval=self.max_interval,
            video_prob=0.75,
            fix_interval_prob=0.75,
        )
        image_idxs = np.array(all_image_ids)[pos]

        views = []
        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])
            rgb_dir = osp.join(scene_dir, "rgb")
            depth_dir = osp.join(scene_dir, "depth")
            cam_dir = osp.join(scene_dir, "cam")

            basename = self.images[view_idx]

            # Load RGB image
            rgb_image = imread_cv2(osp.join(rgb_dir, basename + ".png"))
            # Load depthmap
            depthmap = np.load(osp.join(depth_dir, basename + ".npy"))
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            depthmap[depthmap >= 20] = 0  # invalid

            cam = np.load(osp.join(cam_dir, basename + ".npz"))
            camera_pose = cam["pose"]
            intrinsics = cam["intrinsics"]
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            # generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.75, 0.2, 0.05]
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="UASOL",
                    label=self.scenes[scene_id] + "_" + basename,
                    instance=osp.join(rgb_dir, basename + ".png"),
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(0.9, dtype=np.float32),
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
