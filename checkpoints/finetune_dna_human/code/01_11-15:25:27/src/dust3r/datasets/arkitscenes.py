import os.path as osp
import os
import sys
import itertools

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
import cv2
import numpy as np

from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


def stratified_sampling(indices, num_samples, rng=None):
    if num_samples > len(indices):
        raise ValueError("num_samples cannot exceed the number of available indices.")
    elif num_samples == len(indices):
        return indices

    sorted_indices = sorted(indices)
    stride = len(sorted_indices) / num_samples
    sampled_indices = []
    if rng is None:
        rng = np.random.default_rng()

    for i in range(num_samples):
        start = int(i * stride)
        end = int((i + 1) * stride)
        # Ensure end does not exceed the list
        end = min(end, len(sorted_indices))
        if start < end:
            # Randomly select within the current stratum
            rand_idx = rng.integers(start, end)
            sampled_indices.append(sorted_indices[rand_idx])
        else:
            # In case of any rounding issues, select the last index
            sampled_indices.append(sorted_indices[-1])

    return rng.permutation(sampled_indices)


class ARKitScenes_Multi(BaseMultiViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = 8
        super().__init__(*args, **kwargs)
        if split == "train":
            self.split = "Training"
        elif split == "test":
            self.split = "Test"
        else:
            raise ValueError("")

        self.loaded_data = self._load_data(self.split)

    def _load_data(self, split):
        with np.load(osp.join(self.ROOT, split, "all_metadata.npz")) as data:
            self.scenes: np.ndarray = data["scenes"]
            high_res_list = np.array(
                [
                    d
                    for d in os.listdir(
                        os.path.join(
                            self.ROOT.rstrip("/") + "_highres",
                            split if split == "Training" else "Validation",
                        )
                    )
                    if os.path.join(self.ROOT + "_highres", split, d)
                ]
            )
            self.scenes = np.setdiff1d(self.scenes, high_res_list)
        offset = 0
        counts = []
        scenes = []
        sceneids = []
        images = []
        intrinsics = []
        trajectories = []
        groups = []
        id_ranges = []
        j = 0
        for scene_idx, scene in enumerate(self.scenes):
            scene_dir = osp.join(self.ROOT, self.split, scene)
            with np.load(
                osp.join(scene_dir, "new_scene_metadata.npz"), allow_pickle=True
            ) as data:
                imgs = data["images"]
                intrins = data["intrinsics"]
                traj = data["trajectories"]
                min_seq_len = (
                    self.num_views
                    if not self.allow_repeat
                    else max(self.num_views // 3, 3)
                )
                if len(imgs) < min_seq_len:
                    print(f"Skipping {scene}")
                    continue

                collections = {}
                assert "image_collection" in data, "Image collection not found"
                collections["image"] = data["image_collection"]

                num_imgs = imgs.shape[0]
                img_groups = []
                min_group_len = (
                    self.num_views
                    if not self.allow_repeat
                    else max(self.num_views // 3, 3)
                )
                for ref_id, group in collections["image"].item().items():
                    if len(group) + 1 < min_group_len:
                        continue

                    # groups are (idx, score)s
                    group.insert(0, (ref_id, 1.0))
                    group = [int(x[0] + offset) for x in group]
                    img_groups.append(sorted(group))

                if len(img_groups) == 0:
                    print(f"Skipping {scene}")
                    continue

                scenes.append(scene)
                sceneids.extend([j] * num_imgs)
                id_ranges.extend([(offset, offset + num_imgs) for _ in range(num_imgs)])
                images.extend(imgs)
                K = np.expand_dims(np.eye(3), 0).repeat(num_imgs, 0)

                K[:, 0, 0] = [fx for _, _, fx, _, _, _ in intrins]
                K[:, 1, 1] = [fy for _, _, _, fy, _, _ in intrins]
                K[:, 0, 2] = [cx for _, _, _, _, cx, _ in intrins]
                K[:, 1, 2] = [cy for _, _, _, _, _, cy in intrins]
                intrinsics.extend(list(K))
                trajectories.extend(list(traj))

                # offset groups
                groups.extend(img_groups)
                counts.append(offset)
                offset += num_imgs
                j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.id_ranges = id_ranges
        self.images = images
        self.intrinsics = intrinsics
        self.trajectories = trajectories
        self.groups = groups

    def __len__(self):
        return len(self.groups)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):

        if rng.choice([True, False]):
            image_idxs = np.arange(self.id_ranges[idx][0], self.id_ranges[idx][1])
            cut_off = num_views if not self.allow_repeat else max(num_views // 3, 3)
            start_image_idxs = image_idxs[: len(image_idxs) - cut_off + 1]
            start_id = rng.choice(start_image_idxs)
            pos, ordered_video = self.get_seq_from_start_id(
                num_views,
                start_id,
                image_idxs.tolist(),
                rng,
                max_interval=self.max_interval,
                video_prob=0.8,
                fix_interval_prob=0.5,
                block_shuffle=16,
            )
            image_idxs = np.array(image_idxs)[pos]
        else:
            ordered_video = False
            image_idxs = self.groups[idx]
            image_idxs = rng.permutation(image_idxs)
            if len(image_idxs) > num_views:
                image_idxs = image_idxs[:num_views]
            else:
                if rng.random() < 0.8:
                    image_idxs = rng.choice(image_idxs, size=num_views, replace=True)
                else:
                    repeat_num = num_views // len(image_idxs) + 1
                    image_idxs = np.tile(image_idxs, repeat_num)[:num_views]

        views = []
        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.split, self.scenes[scene_id])

            intrinsics = self.intrinsics[view_idx]
            camera_pose = self.trajectories[view_idx]
            basename = self.images[view_idx]
            assert (
                basename[:8] == self.scenes[scene_id]
            ), f"{basename}, {self.scenes[scene_id]}"
            # print(scene_dir, basename)
            # Load RGB image
            rgb_image = imread_cv2(
                osp.join(scene_dir, "vga_wide", basename.replace(".png", ".jpg"))
            )
            # Load depthmap
            depthmap = imread_cv2(
                osp.join(scene_dir, "lowres_depth", basename), cv2.IMREAD_UNCHANGED
            )
            depthmap = depthmap.astype(np.float32) / 1000.0
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
                    dataset="arkitscenes",
                    label=self.scenes[scene_id] + "_" + basename,
                    instance=f"{str(idx)}_{str(view_idx)}",
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(0.98, dtype=np.float32),
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
