import os.path as osp
import cv2
import numpy as np
import itertools
import os
import sys

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))

from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


class ScanNetpp_Multi(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = 3
        super().__init__(*args, **kwargs)
        assert self.split == "train"
        self.loaded_data = self._load_data()

    def _load_data(self):
        with np.load(osp.join(self.ROOT, "all_metadata.npz")) as data:
            self.scenes = data["scenes"]
        offset = 0
        scenes = []
        sceneids = []
        images = []
        intrinsics = []
        trajectories = []
        groups = []
        id_ranges = []
        j = 0
        self.image_num = 0
        for scene in self.scenes:
            scene_dir = osp.join(self.ROOT, scene)
            with np.load(
                osp.join(scene_dir, "new_scene_metadata.npz"), allow_pickle=True
            ) as data:
                imgs = data["images"]
                self.image_num += len(imgs)
                img_ids = np.arange(len(imgs)).tolist()
                intrins = data["intrinsics"]
                traj = data["trajectories"]
                imgs_on_disk = sorted(os.listdir(osp.join(scene_dir, "images")))
                imgs_on_disk = list(map(lambda x: x[:-4], imgs_on_disk))

                dslr_ids = [
                    i + offset
                    for i in img_ids
                    if imgs[i].startswith("DSC") and imgs[i] in imgs_on_disk
                ]
                iphone_ids = [
                    i + offset
                    for i in img_ids
                    if imgs[i].startswith("frame") and imgs[i] in imgs_on_disk
                ]

                num_imgs = len(imgs)
                assert max(dslr_ids) < min(iphone_ids)
                assert "image_collection" in data

                img_groups = []
                img_id_ranges = []

                for ref_id, group in data["image_collection"].item().items():
                    if len(group) + 1 < self.num_views:
                        continue
                    group.insert(0, (ref_id, 1.0))
                    sorted_group = sorted(group, key=lambda x: x[1], reverse=True)
                    group = [int(x[0] + offset) for x in sorted_group]
                    img_groups.append(sorted(group))

                    if imgs[ref_id].startswith("frame"):
                        img_id_ranges.append(dslr_ids)
                    else:
                        img_id_ranges.append(iphone_ids)

                if len(img_groups) == 0:
                    print(f"Skipping {scene}")
                    continue
                scenes.append(scene)
                sceneids.extend([j] * num_imgs)
                images.extend(imgs)
                intrinsics.append(intrins)
                trajectories.append(traj)

                # offset groups
                groups.extend(img_groups)
                id_ranges.extend(img_id_ranges)
                offset += num_imgs
                j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.intrinsics = np.concatenate(intrinsics, axis=0)
        self.trajectories = np.concatenate(trajectories, axis=0)
        self.id_ranges = id_ranges
        self.groups = groups

    def __len__(self):
        return len(self.groups) * 10

    def get_image_num(self):
        return self.image_num

    def _get_views(self, idx, resolution, rng, num_views):
        idx = idx // 10
        image_idxs = self.groups[idx]
        rand_val = rng.random()

        image_idxs_video = self.id_ranges[idx]
        cut_off = num_views if not self.allow_repeat else max(num_views // 3, 3)
        start_image_idxs = image_idxs_video[: len(image_idxs_video) - cut_off + 1]

        if rand_val < 0.7 and len(start_image_idxs) > 0:
            start_id = rng.choice(start_image_idxs)
            pos, ordered_video = self.get_seq_from_start_id(
                num_views,
                start_id,
                image_idxs_video,
                rng,
                max_interval=self.max_interval,
                video_prob=0.8,
                fix_interval_prob=0.5,
                block_shuffle=16,
            )
            image_idxs = np.array(image_idxs_video)[pos]

        else:
            ordered_video = True
            # ordered video with varying intervals
            num_candidates = len(image_idxs)
            max_id = min(num_candidates, int(num_views * (2 + 2 * rng.random())))
            image_idxs = sorted(rng.permutation(image_idxs[:max_id])[:num_views])
            if rand_val > 0.75:
                ordered_video = False
                image_idxs = rng.permutation(image_idxs)

        views = []
        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])

            intrinsics = self.intrinsics[view_idx]
            camera_pose = self.trajectories[view_idx]
            basename = self.images[view_idx]

            # Load RGB image
            rgb_image = imread_cv2(osp.join(scene_dir, "images", basename + ".jpg"))
            # Load depthmap
            depthmap = imread_cv2(
                osp.join(scene_dir, "depth", basename + ".png"), cv2.IMREAD_UNCHANGED
            )
            depthmap = depthmap.astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0  # invalid

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
                    dataset="ScanNet++",
                    label=self.scenes[scene_id] + "_" + basename,
                    instance=f"{str(idx)}_{str(view_idx)}",
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(0.99, dtype=np.float32),
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
