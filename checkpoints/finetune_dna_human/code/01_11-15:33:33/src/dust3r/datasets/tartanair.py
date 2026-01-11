import os.path as osp
import numpy as np
import cv2
import numpy as np
import itertools
import os
import sys

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))

from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


class TartanAir_Multi(BaseMultiViewDataset):

    def __init__(self, ROOT, *args, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = 20
        super().__init__(*args, **kwargs)
        # loading all
        assert self.split is None
        self._load_data()

    def _load_data(self):
        scene_dirs = sorted(
            [
                d
                for d in os.listdir(self.ROOT)
                if os.path.isdir(os.path.join(self.ROOT, d))
            ]
        )

        offset = 0
        scenes = []
        sceneids = []
        images = []
        scene_img_list = []
        start_img_ids = []
        j = 0

        for scene in scene_dirs:
            for mode in ["Easy", "Hard"]:
                seq_dirs = sorted(
                    [
                        os.path.join(self.ROOT, scene, mode, d)
                        for d in os.listdir(os.path.join(self.ROOT, scene, mode))
                        if os.path.isdir(os.path.join(self.ROOT, scene, mode, d))
                    ]
                )
                for seq_dir in seq_dirs:
                    basenames = sorted(
                        [f[:-8] for f in os.listdir(seq_dir) if f.endswith(".png")]
                    )
                    num_imgs = len(basenames)
                    cut_off = (
                        self.num_views
                        if not self.allow_repeat
                        else max(self.num_views // 3, 3)
                    )

                    if num_imgs < cut_off:
                        print(f"Skipping {scene}")
                        continue
                    img_ids = list(np.arange(num_imgs) + offset)
                    start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

                    scenes.append(seq_dir)
                    scene_img_list.append(img_ids)
                    sceneids.extend([j] * num_imgs)
                    images.extend(basenames)
                    start_img_ids.extend(start_img_ids_)
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

    def get_stats(self):
        return f"{len(self)} groups of views"

    def _get_views(self, idx, resolution, rng, num_views):
        start_id = self.start_img_ids[idx]
        scene_id = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_id]
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_image_ids,
            rng,
            max_interval=self.max_interval,
            video_prob=0.8,
            fix_interval_prob=0.8,
            block_shuffle=16,
        )
        image_idxs = np.array(all_image_ids)[pos]

        views = []

        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = self.scenes[scene_id]
            basename = self.images[view_idx]

            img = basename + "_rgb.png"
            image = imread_cv2(osp.join(scene_dir, img))
            depthmap = np.load(osp.join(scene_dir, basename + "_depth.npy"))
            camera_params = np.load(osp.join(scene_dir, basename + "_cam.npz"))

            intrinsics = camera_params["camera_intrinsics"]
            camera_pose = camera_params["camera_pose"]

            sky_mask = depthmap >= 1000
            depthmap[sky_mask] = -1.0  # sky
            depthmap = np.nan_to_num(depthmap, nan=0, posinf=0, neginf=0)
            threshold = (
                np.percentile(depthmap[depthmap > 0], 98)
                if depthmap[depthmap > 0].size > 0
                else 0
            )
            depthmap[depthmap > threshold] = 0.0

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(scene_dir, img)
            )

            # generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.75, 0.2, 0.05]
            )

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,  # cam2world
                    camera_intrinsics=intrinsics,
                    dataset="TartanAir",
                    label=scene_dir,
                    is_metric=self.is_metric,
                    instance=scene_dir + "_" + img,
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
