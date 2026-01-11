import os.path as osp
import cv2
import numpy as np
import itertools
import os
import sys
import json

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


class OmniObject3D_Multi(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = False
        self.is_metric = False  # True
        super().__init__(*args, **kwargs)

        self.loaded_data = self._load_data()

    def _load_data(self):
        self.scenes = [
            d
            for d in os.listdir(self.ROOT)
            if os.path.isdir(os.path.join(self.ROOT, d))
        ]
        with open(os.path.join(self.ROOT, "scale.json"), "r") as f:
            self.scales = json.load(f)
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
            cut_off = (
                self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            )

            if num_imgs < cut_off:
                print(f"Skipping {scene}")
                continue
            img_ids = list(np.arange(num_imgs) + offset)
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

            start_img_ids.extend([(scene, id) for id in start_img_ids_])
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
        scene, start_id = self.start_img_ids[idx]
        all_image_ids = self.scene_img_list[self.sceneids[start_id]]
        pos, ordered_video = self.get_seq_from_start_id(
            num_views, start_id, all_image_ids, rng, max_interval=100, video_prob=0.0
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
            depthmap = np.load(osp.join(depth_dir, basename + ".npy"))
            cam = np.load(osp.join(cam_dir, basename + ".npz"))
            camera_pose = cam["pose"]
            intrinsics = cam["intrinsics"]
            scale = self.scales[self.scenes[scene_id]]
            depthmap = depthmap / scale / 1000.0
            camera_pose[:3, 3] = camera_pose[:3, 3] / scale / 1000.0

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.8, 0.15, 0.05]
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="OmniObject3D",
                    label=self.scenes[scene_id] + "_" + basename,
                    instance=f"{str(idx)}_{str(view_idx)}",
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(1.0, dtype=np.float32),
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
