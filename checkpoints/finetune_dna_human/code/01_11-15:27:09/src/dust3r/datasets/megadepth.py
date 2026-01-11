import os.path as osp
import numpy as np
import itertools
import os
import sys

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


class MegaDepth_Multi(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self._load_data(self.split)
        self.is_metric = False
        if self.split is None:
            pass
        elif self.split == "train":
            self.select_scene(("0015", "0022"), opposite=True)
        elif self.split == "val":
            self.select_scene(("0015", "0022"))
        else:
            raise ValueError(f"bad {self.split=}")

    def _load_data(self, split):
        with np.load(
            osp.join(self.ROOT, "megadepth_sets_64.npz"), allow_pickle=True
        ) as data:
            self.all_scenes = data["scenes"]
            self.all_images = data["images"]
            self.sets = data["sets"]

    def __len__(self):
        return len(self.sets)

    def get_image_num(self):
        return len(self.all_images)

    def get_stats(self):
        return f"{len(self)} groups from {len(self.all_scenes)} scenes"

    def select_scene(self, scene, *instances, opposite=False):
        scenes = (scene,) if isinstance(scene, str) else tuple(scene)
        scene_id = [s.startswith(scenes) for s in self.all_scenes]
        assert any(scene_id), "no scene found"
        valid = np.in1d(self.sets[:, 0], np.nonzero(scene_id)[0])
        if instances:
            raise NotImplementedError("selecting instances not implemented")
        if opposite:
            valid = ~valid
        assert valid.any()
        self.sets = self.sets[valid]

    def _get_views(self, idx, resolution, rng, num_views):
        scene_id = self.sets[idx][0]
        image_idxs = self.sets[idx][1:65]
        replace = False if not self.allow_repeat else True
        image_idxs = rng.choice(image_idxs, num_views, replace=replace)
        scene, subscene = self.all_scenes[scene_id].split()
        seq_path = osp.join(self.ROOT, scene, subscene)
        views = []
        for im_id in image_idxs:
            img = self.all_images[im_id]
            try:
                image = imread_cv2(osp.join(seq_path, img + ".jpg"))
                depthmap = imread_cv2(osp.join(seq_path, img + ".exr"))
                camera_params = np.load(osp.join(seq_path, img + ".npz"))
            except Exception as e:
                raise OSError(f"cannot load {img}, got exception {e}")
            intrinsics = np.float32(camera_params["intrinsics"])
            camera_pose = np.float32(camera_params["cam2world"])
            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(seq_path, img)
            )
            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,  # cam2world
                    camera_intrinsics=intrinsics,
                    dataset="MegaDepth",
                    label=osp.relpath(seq_path, self.ROOT),
                    is_metric=self.is_metric,
                    instance=img,
                    is_video=False,
                    quantile=np.array(0.96, dtype=np.float32),
                    img_mask=True,
                    ray_mask=False,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                )
            )
        assert len(views) == num_views
        return views
