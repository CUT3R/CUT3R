import os.path as osp
import sys

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
import cv2
import numpy as np

from dust3r.datasets.co3d import Co3d_Multi
from dust3r.utils.image import imread_cv2


class WildRGBD_Multi(Co3d_Multi):
    def __init__(self, mask_bg="rand", *args, ROOT, **kwargs):
        super().__init__(mask_bg, *args, ROOT=ROOT, **kwargs)
        self.dataset_label = "WildRGBD"
        self.is_metric = True
        # load all scenes
        self.scenes.pop(("box", "scenes/scene_257"), None)
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

    def _get_metadatapath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, "metadata", f"{view_idx:0>5d}.npz")

    def _get_impath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, "rgb", f"{view_idx:0>5d}.jpg")

    def _get_depthpath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, "depth", f"{view_idx:0>5d}.png")

    def _get_maskpath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, "masks", f"{view_idx:0>5d}.png")

    def _read_depthmap(self, depthpath, input_metadata):
        # We store depths in the depth scale of 1000.
        # That is, when we load depth image and divide by 1000, we could get depth in meters.
        depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
        depthmap = depthmap.astype(np.float32) / 1000.0
        return depthmap

    def _get_views(self, idx, resolution, rng, num_views):
        views = super()._get_views(idx, resolution, rng, num_views)
        for view in views:
            assert view["is_metric"]
            view["quantile"] = np.array(0.96, dtype=np.float32)
        return views
