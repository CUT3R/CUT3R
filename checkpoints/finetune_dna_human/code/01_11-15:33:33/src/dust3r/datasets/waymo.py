import os.path as osp
import os
import numpy as np
import sys

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
import h5py
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


class Waymo_Multi(BaseMultiViewDataset):
    """Dataset of outdoor street scenes, 5 images each time"""

    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.max_interval = 8
        self.video = True
        self.is_metric = True
        super().__init__(*args, **kwargs)
        assert self.split is None
        self._load_data()

    def load_invalid_dict(self, h5_file_path):
        invalid_dict = {}
        with h5py.File(h5_file_path, "r") as h5f:
            for scene in h5f:
                data = h5f[scene]["invalid_pairs"][:]
                invalid_pairs = set(
                    tuple(pair.decode("utf-8").split("_")) for pair in data
                )
                invalid_dict[scene] = invalid_pairs
        return invalid_dict

    def _load_data(self):
        invalid_dict = self.load_invalid_dict(
            os.path.join(self.ROOT, "invalid_files.h5")
        )
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
        start_img_ids = []
        scene_img_list = []
        is_video = []
        j = 0

        for scene in scene_dirs:
            scene_dir = osp.join(self.ROOT, scene)
            invalid_pairs = invalid_dict.get(scene, set())
            seq2frames = {}
            for f in os.listdir(scene_dir):
                if not f.endswith(".jpg"):
                    continue
                basename = f[:-4]
                frame_id = basename.split("_")[0]
                seq_id = basename.split("_")[1]
                if seq_id == "5":
                    continue
                if (seq_id, frame_id) in invalid_pairs:
                    continue  # Skip invalid files
                if seq_id not in seq2frames:
                    seq2frames[seq_id] = []
                seq2frames[seq_id].append(frame_id)

            for seq_id, frame_ids in seq2frames.items():
                frame_ids = sorted(frame_ids)
                num_imgs = len(frame_ids)
                img_ids = list(np.arange(num_imgs) + offset)
                cut_off = (
                    self.num_views
                    if not self.allow_repeat
                    else max(self.num_views // 3, 3)
                )
                start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

                if num_imgs < cut_off:
                    print(f"Skipping {scene}_{seq_id}")
                    continue

                scenes.append((scene, seq_id))
                sceneids.extend([j] * num_imgs)
                images.extend(frame_ids)
                start_img_ids.extend(start_img_ids_)
                scene_img_list.append(img_ids)

                offset += num_imgs
                j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list
        self.is_video = is_video

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def get_stats(self):
        return f"{len(self)} groups of views"

    def _get_views(self, idx, resolution, rng, num_views):
        start_id = self.start_img_ids[idx]
        all_image_ids = self.scene_img_list[self.sceneids[start_id]]
        _, seq_id = self.scenes[self.sceneids[start_id]]
        max_interval = self.max_interval // 2 if seq_id == "4" else self.max_interval
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_image_ids,
            rng,
            max_interval=max_interval,
            video_prob=0.9,
            fix_interval_prob=0.9,
            block_shuffle=16,
        )
        image_idxs = np.array(all_image_ids)[pos]
        views = []
        ordered_video = True

        views = []

        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir, seq_id = self.scenes[scene_id]
            scene_dir = osp.join(self.ROOT, scene_dir)
            frame_id = self.images[view_idx]

            impath = f"{frame_id}_{seq_id}"
            image = imread_cv2(osp.join(scene_dir, impath + ".jpg"))
            depthmap = imread_cv2(osp.join(scene_dir, impath + ".exr"))
            camera_params = np.load(osp.join(scene_dir, impath + ".npz"))

            intrinsics = np.float32(camera_params["intrinsics"])
            camera_pose = np.float32(camera_params["cam2world"])

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(scene_dir, impath)
            )

            # generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.85, 0.10, 0.05]
            )

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,  # cam2world
                    camera_intrinsics=intrinsics,
                    dataset="Waymo",
                    label=osp.relpath(scene_dir, self.ROOT),
                    is_metric=self.is_metric,
                    instance=osp.join(scene_dir, impath + ".jpg"),
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

        return views
