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


class RE10K_Multi(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = False
        self.max_interval = 128
        super().__init__(*args, **kwargs)
        self.loaded_data = self._load_data()

    def _load_data(self):
        self.scenes = os.listdir(self.ROOT)

        offset = 0
        scenes = []
        sceneids = []
        scene_img_list = []
        images = []
        start_img_ids = []

        j = 0
        for scene in tqdm(self.scenes):
            scene_dir = osp.join(self.ROOT, scene)
            rgb_dir = osp.join(scene_dir, "rgb")
            basenames = sorted(
                [f[:-4] for f in os.listdir(rgb_dir) if f.endswith(".png")],
                key=lambda x: int(x),
            )

            num_imgs = len(basenames)
            img_ids = list(np.arange(num_imgs) + offset)
            cut_off = (
                self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            )
            if num_imgs < cut_off:
                print(f"Skipping {scene}")
                continue
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

            start_img_ids.extend([(scene, id) for id in start_img_ids_])
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

        self.invalid_scenes = {scene: False for scene in self.scenes}

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        invalid_seq = True
        scene, start_id = self.start_img_ids[idx]

        while invalid_seq:
            while self.invalid_scenes[scene]:
                idx = rng.integers(low=0, high=len(self.start_img_ids))
                scene, start_id = self.start_img_ids[idx]

            all_image_ids = self.scene_img_list[self.sceneids[start_id]]
            pos, ordered_video = self.get_seq_from_start_id(
                num_views, start_id, all_image_ids, rng, max_interval=self.max_interval
            )
            image_idxs = np.array(all_image_ids)[pos]

            views = []
            for view_idx in image_idxs:
                scene_id = self.sceneids[view_idx]
                scene_dir = osp.join(self.ROOT, self.scenes[scene_id])
                rgb_dir = osp.join(scene_dir, "rgb")
                cam_dir = osp.join(scene_dir, "cam")

                basename = self.images[view_idx]

                try:
                    # Load RGB image
                    rgb_image = imread_cv2(osp.join(rgb_dir, basename + ".png"))
                    # Load depthmap, no depth, set to all ones
                    depthmap = np.ones_like(rgb_image[..., 0], dtype=np.float32)
                    cam = np.load(osp.join(cam_dir, basename + ".npz"))
                    intrinsics = cam["intrinsics"]
                    camera_pose = cam["pose"]
                except:
                    print(f"Error loading {scene} {basename}, skipping")
                    self.invalid_scenes[scene] = True
                    break

                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
                )

                views.append(
                    dict(
                        img=rgb_image,
                        depthmap=depthmap.astype(np.float32),
                        camera_pose=camera_pose.astype(np.float32),
                        camera_intrinsics=intrinsics.astype(np.float32),
                        dataset="realestate10k",
                        label=self.scenes[scene_id] + "_" + basename,
                        instance=f"{str(idx)}_{str(view_idx)}",
                        is_metric=self.is_metric,
                        is_video=ordered_video,
                        quantile=np.array(0.98, dtype=np.float32),
                        img_mask=True,
                        ray_mask=False,
                        camera_only=True,
                        depth_only=False,
                        single_view=False,
                        reset=False,
                    )
                )
            if len(views) == num_views:
                invalid_seq = False
        return views
