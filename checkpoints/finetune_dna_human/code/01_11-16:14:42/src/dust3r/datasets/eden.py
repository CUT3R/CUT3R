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


class EDEN_Multi(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        super().__init__(*args, **kwargs)
        self.loaded_data = self._load_data()

    def _load_data(self):
        scenes = os.listdir(self.ROOT)
        img_names = []
        for scene in scenes:
            scene_dir = osp.join(self.ROOT, scene)
            rgb_dir = osp.join(scene_dir, "rgb")
            basenames = sorted(
                [f[:-4] for f in os.listdir(rgb_dir) if f.endswith(".png")]
            )
            img_names.extend([(scene, basename) for basename in basenames])

        self.img_names = img_names

    def __len__(self):
        return len(self.img_names)

    def get_image_num(self):
        return len(self.img_names)

    def _get_views(self, idx, resolution, rng, num_views):
        new_seed = rng.integers(0, 2**32) + idx
        new_rng = np.random.default_rng(new_seed)
        img_names = new_rng.permutation(self.img_names)

        views = []
        i = 0
        while len(views) < num_views:
            # Load RGB image
            scene, img_name = img_names[i]
            try:
                rgb_image = imread_cv2(
                    osp.join(self.ROOT, scene, "rgb", f"{img_name}.png")
                )
                depthmap = np.load(
                    osp.join(self.ROOT, scene, "depth", f"{img_name}.npy")
                )
                depthmap = np.nan_to_num(depthmap, nan=0, posinf=0, neginf=0)

                intrinsics = np.load(
                    osp.join(self.ROOT, scene, "cam", f"{img_name}.npz")
                )["intrinsics"]
                # camera pose is not provided, placeholder
                camera_pose = np.eye(4)
            except:
                i += 1
                continue

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=img_name
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="EDEN",
                    label=img_name,
                    instance=osp.join(self.ROOT, scene, "rgb", f"{img_name}.png"),
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
            i += 1
        return views
