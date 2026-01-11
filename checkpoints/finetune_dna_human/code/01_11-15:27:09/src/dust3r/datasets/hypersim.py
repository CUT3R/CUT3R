import os.path as osp
import os
import sys
import itertools

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
import cv2
import numpy as np

from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


class HyperSim_Multi(BaseMultiViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = 4
        super().__init__(*args, **kwargs)

        self.loaded_data = self._load_data()

    def _load_data(self):
        self.all_scenes = sorted(
            [f for f in os.listdir(self.ROOT) if os.path.isdir(osp.join(self.ROOT, f))]
        )
        subscenes = []
        for scene in self.all_scenes:
            # not empty
            subscenes.extend(
                [
                    osp.join(scene, f)
                    for f in os.listdir(osp.join(self.ROOT, scene))
                    if os.path.isdir(osp.join(self.ROOT, scene, f))
                    and len(os.listdir(osp.join(self.ROOT, scene, f))) > 0
                ]
            )

        offset = 0
        scenes = []
        sceneids = []
        images = []
        start_img_ids = []
        scene_img_list = []
        j = 0
        for scene_idx, scene in enumerate(subscenes):
            scene_dir = osp.join(self.ROOT, scene)
            rgb_paths = sorted([f for f in os.listdir(scene_dir) if f.endswith(".png")])
            assert len(rgb_paths) > 0, f"{scene_dir} is empty."
            num_imgs = len(rgb_paths)
            cut_off = (
                self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            )
            if num_imgs < cut_off:
                print(f"Skipping {scene}")
                continue
            img_ids = list(np.arange(num_imgs) + offset)
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

            scenes.append(scene)
            scene_img_list.append(img_ids)
            sceneids.extend([j] * num_imgs)
            images.extend(rgb_paths)
            start_img_ids.extend(start_img_ids_)
            offset += num_imgs
            j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.scene_img_list = scene_img_list
        self.start_img_ids = start_img_ids

    def __len__(self):
        return len(self.start_img_ids) * 10

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        idx = idx // 10
        start_id = self.start_img_ids[idx]
        scene_id = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_id]
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_image_ids,
            rng,
            max_interval=self.max_interval,
            block_shuffle=16,
        )
        image_idxs = np.array(all_image_ids)[pos]
        views = []
        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])

            rgb_path = self.images[view_idx]
            depth_path = rgb_path.replace("rgb.png", "depth.npy")
            cam_path = rgb_path.replace("rgb.png", "cam.npz")

            rgb_image = imread_cv2(osp.join(scene_dir, rgb_path), cv2.IMREAD_COLOR)
            depthmap = np.load(osp.join(scene_dir, depth_path)).astype(np.float32)
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            cam_file = np.load(osp.join(scene_dir, cam_path))
            intrinsics = cam_file["intrinsics"].astype(np.float32)
            camera_pose = cam_file["pose"].astype(np.float32)

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
                    dataset="hypersim",
                    label=self.scenes[scene_id] + "_" + rgb_path,
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
