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


class PointOdyssey_Multi(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = 4
        super().__init__(*args, **kwargs)
        assert self.split in ["train", "test", "val"]
        self.scenes_to_use = [
            # 'cab_h_bench_3rd', 'cab_h_bench_ego1', 'cab_h_bench_ego2',
            "cnb_dlab_0215_3rd",
            "cnb_dlab_0215_ego1",
            "cnb_dlab_0225_3rd",
            "cnb_dlab_0225_ego1",
            "dancing",
            "dancingroom0_3rd",
            "footlab_3rd",
            "footlab_ego1",
            "footlab_ego2",
            "girl",
            "girl_egocentric",
            "human_egocentric",
            "human_in_scene",
            "human_in_scene1",
            "kg",
            "kg_ego1",
            "kg_ego2",
            "kitchen_gfloor",
            "kitchen_gfloor_ego1",
            "kitchen_gfloor_ego2",
            "scene_carb_h_tables",
            "scene_carb_h_tables_ego1",
            "scene_carb_h_tables_ego2",
            "scene_j716_3rd",
            "scene_j716_ego1",
            "scene_j716_ego2",
            "scene_recording_20210910_S05_S06_0_3rd",
            "scene_recording_20210910_S05_S06_0_ego2",
            "scene1_0129",
            "scene1_0129_ego",
            "seminar_h52_3rd",
            "seminar_h52_ego1",
            "seminar_h52_ego2",
        ]
        self.loaded_data = self._load_data(self.split)

    def _load_data(self, split):
        root = os.path.join(self.ROOT, split)
        self.scenes = []

        offset = 0
        scenes = []
        sceneids = []
        scene_img_list = []
        images = []
        start_img_ids = []

        j = 0
        for scene in tqdm(os.listdir(root)):
            if scene not in self.scenes_to_use:
                continue
            scene_dir = osp.join(root, scene)
            rgb_dir = osp.join(scene_dir, "rgb")
            basenames = sorted(
                [f[:-4] for f in os.listdir(rgb_dir) if f.endswith(".jpg")]
            )
            num_imgs = len(basenames)
            img_ids = list(np.arange(num_imgs) + offset)
            cut_off = (
                self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            )
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]
            # start_img_ids_ = img_ids[:-self.num_views+1]

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
            video_prob=1.0,
            fix_interval_prob=1.0,
        )
        image_idxs = np.array(all_image_ids)[pos]

        views = []
        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.split, self.scenes[scene_id])
            rgb_dir = osp.join(scene_dir, "rgb")
            depth_dir = osp.join(scene_dir, "depth")
            cam_dir = osp.join(scene_dir, "cam")

            basename = self.images[view_idx]

            # Load RGB image
            rgb_image = imread_cv2(osp.join(rgb_dir, basename + ".jpg"))
            # Load depthmap
            depthmap = np.load(osp.join(depth_dir, basename + ".npy"))
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            depthmap[depthmap > 1000] = 0.0

            cam = np.load(osp.join(cam_dir, basename + ".npz"))
            camera_pose = cam["pose"]
            intrinsics = cam["intrinsics"]
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            # generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.9, 0.05, 0.05]
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="PointOdyssey",
                    label=self.scenes[scene_id] + "_" + basename,
                    instance=osp.join(rgb_dir, basename + ".jpg"),
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
