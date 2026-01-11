import os.path as osp
import sys

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
import cv2
import numpy as np

from dust3r.datasets.co3d import Co3d_Multi
from dust3r.utils.image import imread_cv2


class Cop3D_Multi(Co3d_Multi):
    def __init__(self, mask_bg="rand", *args, ROOT, **kwargs):
        super().__init__(mask_bg, *args, ROOT=ROOT, **kwargs)
        self.dataset_label = "Cop3D"
        self.is_metric = False

    def _get_metadatapath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, "images", f"frame{view_idx:06n}.npz")

    def _get_impath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, "images", f"frame{view_idx:06n}.jpg")

    def _get_depthpath(self, obj, instance, view_idx):
        # no depth, pseduo path just for getting the right resolution
        return osp.join(self.ROOT, obj, instance, "images", f"frame{view_idx:06n}.jpg")

    def _get_maskpath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, "masks", f"frame{view_idx:06n}.png")

    def _read_depthmap(self, impath, input_metadata):
        # no depth, set to all ones
        img = imread_cv2(impath, cv2.IMREAD_UNCHANGED)
        depthmap = np.ones_like(img[..., 0], dtype=np.float32)
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
            if len(image_pool) < self.num_views:
                print("Invalid scene!")
                self.invalid_scenes[scene_info] = True
                continue

            imgs_idxs, ordered_video = self.get_seq_from_start_id(
                num_views,
                ref_img_idx,
                image_pool,
                rng,
                max_interval=5,
                video_prob=1.0,
                fix_interval_prob=0.9,
            )

            views = []

            for im_idx in imgs_idxs:
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

                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath
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
                        quantile=np.array(0.96, dtype=np.float32),
                        img_mask=True,
                        ray_mask=False,
                        camera_only=True,
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
