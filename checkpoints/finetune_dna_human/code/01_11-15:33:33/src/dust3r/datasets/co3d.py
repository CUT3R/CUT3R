import os.path as osp
import json
import itertools
from collections import deque
import sys

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
import cv2
import numpy as np
import time

from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


class Co3d_Multi(BaseMultiViewDataset):
    def __init__(self, mask_bg="rand", *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, "rand")
        self.mask_bg = mask_bg
        self.is_metric = False
        self.dataset_label = "Co3d_v2"

        # load all scenes
        with open(osp.join(self.ROOT, f"selected_seqs_{self.split}.json"), "r") as f:
            self.scenes = json.load(f)
            self.scenes = {k: v for k, v in self.scenes.items() if len(v) > 0}
            self.scenes = {
                (k, k2): v2 for k, v in self.scenes.items() for k2, v2 in v.items()
            }
        self.scene_list = list(self.scenes.keys())
        cut_off = (
            self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
        )
        self.cut_off = cut_off
        self.all_ref_imgs = [
            (key, value)
            for key, values in self.scenes.items()
            for value in values[: len(values) - cut_off + 1]
        ]
        self.invalidate = {scene: {} for scene in self.scene_list}
        self.invalid_scenes = {scene: False for scene in self.scene_list}

    def __len__(self):
        return len(self.all_ref_imgs)

    def _get_metadatapath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, "images", f"frame{view_idx:06n}.npz")

    def _get_impath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, "images", f"frame{view_idx:06n}.jpg")

    def _get_depthpath(self, obj, instance, view_idx):
        return osp.join(
            self.ROOT, obj, instance, "depths", f"frame{view_idx:06n}.jpg.geometric.png"
        )

    def _get_maskpath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, "masks", f"frame{view_idx:06n}.png")

    def _read_depthmap(self, depthpath, input_metadata):
        depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
        depthmap = (depthmap.astype(np.float32) / 65535) * np.nan_to_num(
            input_metadata["maximum_depth"]
        )
        return depthmap

    def _get_views(self, idx, resolution, rng, num_views):
        invalid_seq = True
        scene_info, ref_img_idx = self.all_ref_imgs[idx]

        while invalid_seq:
            while self.invalid_scenes[scene_info]:
                idx = rng.integers(low=0, high=len(self.all_ref_imgs))
                scene_info, ref_img_idx = self.all_ref_imgs[idx]

            obj, instance = scene_info

            image_pool = self.scenes[obj, instance]
            if len(image_pool) < self.cut_off:
                print("Invalid scene!")
                self.invalid_scenes[scene_info] = True
                continue

            imgs_idxs, ordered_video = self.get_seq_from_start_id(
                num_views, ref_img_idx, image_pool, rng
            )

            if resolution not in self.invalidate[obj, instance]:  # flag invalid images
                self.invalidate[obj, instance][resolution] = [
                    False for _ in range(len(image_pool))
                ]
            # decide now if we mask the bg
            mask_bg = (self.mask_bg == True) or (
                self.mask_bg == "rand" and rng.choice(2, p=[0.9, 0.1])
            )
            views = []

            imgs_idxs = deque(imgs_idxs)

            while len(imgs_idxs) > 0:  # some images (few) have zero depth
                if (
                    len(image_pool) - sum(self.invalidate[obj, instance][resolution])
                    < self.cut_off
                ):
                    print("Invalid scene!")
                    invalid_seq = True
                    self.invalid_scenes[scene_info] = True
                    break

                im_idx = imgs_idxs.pop()
                if self.invalidate[obj, instance][resolution][im_idx]:
                    # search for a valid image
                    ordered_video = False
                    random_direction = 2 * rng.choice(2) - 1
                    for offset in range(1, len(image_pool)):
                        tentative_im_idx = (im_idx + (random_direction * offset)) % len(
                            image_pool
                        )
                        if not self.invalidate[obj, instance][resolution][
                            tentative_im_idx
                        ]:
                            im_idx = tentative_im_idx
                            break
                view_idx = image_pool[im_idx]
                impath = self._get_impath(obj, instance, view_idx)
                depthpath = self._get_depthpath(obj, instance, view_idx)

                # load camera params
                metadata_path = self._get_metadatapath(obj, instance, view_idx)
                input_metadata = np.load(metadata_path)
                camera_pose = input_metadata["camera_pose"].astype(np.float32)
                intrinsics = input_metadata["camera_intrinsics"].astype(np.float32)

                # load image and depth
                rgb_image = imread_cv2(impath)
                depthmap = self._read_depthmap(depthpath, input_metadata)

                if mask_bg:
                    # load object mask
                    maskpath = self._get_maskpath(obj, instance, view_idx)
                    maskmap = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED).astype(
                        np.float32
                    )
                    maskmap = (maskmap / 255.0) > 0.1

                    # update the depthmap with mask
                    depthmap *= maskmap
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath
                )
                num_valid = (depthmap > 0.0).sum()
                if num_valid == 0:
                    # problem, invalidate image and retry
                    self.invalidate[obj, instance][resolution][im_idx] = True
                    imgs_idxs.append(im_idx)
                    continue

                # generate img mask and raymap mask
                img_mask, ray_mask = self.get_img_and_ray_masks(
                    self.is_metric, len(views), rng
                )

                views.append(
                    dict(
                        img=rgb_image,
                        depthmap=depthmap,
                        camera_pose=camera_pose,
                        camera_intrinsics=intrinsics,
                        dataset=self.dataset_label,
                        label=osp.join(obj, instance),
                        instance=osp.split(impath)[1],
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

            if len(views) == num_views and not all(
                [view["instance"] == views[0]["instance"] for view in views]
            ):
                invalid_seq = False
        return views
