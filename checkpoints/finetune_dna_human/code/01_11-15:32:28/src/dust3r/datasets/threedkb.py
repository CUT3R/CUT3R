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


class ThreeDKenBurns(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = False
        self.is_metric = False
        super().__init__(*args, **kwargs)
        self.loaded_data = self._load_data()

    def _load_data(self):
        self.scenes = os.listdir(self.ROOT)

        offset = 0
        scenes = []
        sceneids = []
        images = []
        img_ids = []

        j = 0
        for scene in tqdm(self.scenes):
            scene_dir = osp.join(self.ROOT, scene)
            rgb_dir = osp.join(scene_dir, "rgb")
            basenames = sorted(
                [f[:-4] for f in os.listdir(rgb_dir) if f.endswith(".png")]
            )

            num_imgs = len(basenames)
            img_ids_ = list(np.arange(num_imgs) + offset)

            img_ids.extend(img_ids_)
            sceneids.extend([j] * num_imgs)
            images.extend(basenames)
            scenes.append(scene)

            # offset groups
            offset += num_imgs
            j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.img_ids = img_ids

    def __len__(self):
        return len(self.img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        new_seed = rng.integers(0, 2**32) + idx
        new_rng = np.random.default_rng(new_seed)
        image_idxs = new_rng.choice(self.img_ids, num_views, replace=False)

        views = []
        for view_idx in image_idxs:
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])
            rgb_dir = osp.join(scene_dir, "rgb")
            depth_dir = osp.join(scene_dir, "depth")
            cam_dir = osp.join(scene_dir, "cam")

            basename = self.images[view_idx]

            # Load RGB image
            rgb_image = imread_cv2(osp.join(rgb_dir, basename + ".png"))
            depthmap = imread_cv2(osp.join(depth_dir, basename + ".exr"))
            depthmap[depthmap > 20000] = 0.0
            depthmap = depthmap / 1000.0
            cam = np.load(osp.join(cam_dir, basename + ".npz"))
            intrinsics = cam["intrinsics"]
            camera_pose = np.eye(4)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="3DKenBurns",
                    label=self.scenes[scene_id] + "_" + basename,
                    instance=f"{str(idx)}_{str(view_idx)}",
                    is_metric=self.is_metric,
                    is_video=False,
                    quantile=np.array(1.0, dtype=np.float32),
                    img_mask=True,
                    ray_mask=False,
                    camera_only=False,
                    depth_only=False,
                    single_view=True,
                    reset=True,
                )
            )
        assert len(views) == num_views
        return views
