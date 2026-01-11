import os.path as osp
import numpy as np
import os
import sys

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2
import h5py
from tqdm import tqdm


class BlendedMVS_Multi(BaseMultiViewDataset):
    """Dataset of outdoor street scenes, 5 images each time"""

    def __init__(self, *args, ROOT, split=None, **kwargs):
        self.ROOT = ROOT
        self.video = False
        self.is_metric = False
        super().__init__(*args, **kwargs)
        # assert split is None
        self._load_data()

    def _load_data(self):
        self.data_dict = self.read_h5_file(os.path.join(self.ROOT, "new_overlap.h5"))
        self.num_imgs = sum(
            [len(self.data_dict[s]["basenames"]) for s in self.data_dict.keys()]
        )
        self.num_scenes = len(self.data_dict.keys())
        self.invalid_scenes = []
        self.is_reachable_cache = {scene: {} for scene in self.data_dict.keys()}

    def read_h5_file(self, h5_file_path):
        data_dict = {}
        self.all_ref_imgs = []
        with h5py.File(h5_file_path, "r") as f:
            for scene_dir in tqdm(f.keys()):
                group = f[scene_dir]
                basenames = group["basenames"][:]
                indices = group["indices"][:]
                values = group["values"][:]
                shape = group.attrs["shape"]
                # Reconstruct the sparse matrix
                score_matrix = np.zeros(shape, dtype=np.float32)
                score_matrix[indices[0], indices[1]] = values
                data_dict[scene_dir] = {
                    "basenames": basenames,
                    "score_matrix": self.build_adjacency_list(score_matrix),
                }
                self.all_ref_imgs.extend(
                    [(scene_dir, b) for b in range(len(basenames))]
                )
        return data_dict

    @staticmethod
    def build_adjacency_list(S, thresh=0.2):
        adjacency_list = [[] for _ in range(len(S))]
        S = S - thresh
        S[S < 0] = 0
        rows, cols = np.nonzero(S)
        for i, j in zip(rows, cols):
            adjacency_list[i].append((j, S[i][j]))
        return adjacency_list

    @staticmethod
    def is_reachable(adjacency_list, start_index, k):
        visited = set()
        stack = [start_index]
        while stack and len(visited) < k:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                for neighbor in adjacency_list[node]:
                    if neighbor[0] not in visited:
                        stack.append(neighbor[0])
        return len(visited) >= k

    @staticmethod
    def random_sequence_no_revisit_with_backtracking(
        adjacency_list, k, start_index, rng: np.random.Generator
    ):
        path = [start_index]
        visited = set([start_index])

        neighbor_iterators = []
        # Initialize the iterator for the start index
        neighbors = adjacency_list[start_index]
        neighbor_idxs = [n[0] for n in neighbors]
        neighbor_weights = [n[1] for n in neighbors]
        neighbor_idxs = rng.choice(
            neighbor_idxs,
            size=len(neighbor_idxs),
            replace=False,
            p=np.array(neighbor_weights) / np.sum(neighbor_weights),
        ).tolist()
        neighbor_iterators.append(iter(neighbor_idxs))

        while len(path) < k:
            if not neighbor_iterators:
                # No possible sequence
                return None
            current_iterator = neighbor_iterators[-1]
            try:
                next_index = next(current_iterator)
                if next_index not in visited:
                    path.append(next_index)
                    visited.add(next_index)

                    # Prepare iterator for the next node
                    neighbors = adjacency_list[next_index]
                    neighbor_idxs = [n[0] for n in neighbors]
                    neighbor_weights = [n[1] for n in neighbors]
                    neighbor_idxs = rng.choice(
                        neighbor_idxs,
                        size=len(neighbor_idxs),
                        replace=False,
                        p=np.array(neighbor_weights) / np.sum(neighbor_weights),
                    ).tolist()
                    neighbor_iterators.append(iter(neighbor_idxs))
            except StopIteration:
                # No more neighbors to try at this node, backtrack
                neighbor_iterators.pop()
                visited.remove(path.pop())
        return path

    @staticmethod
    def random_sequence_with_optional_repeats(
        adjacency_list,
        k,
        start_index,
        rng: np.random.Generator,
        max_k=None,
        max_attempts=100,
    ):
        if max_k is None:
            max_k = k
        path = [start_index]
        visited = set([start_index])
        current_index = start_index
        attempts = 0

        while len(path) < max_k and attempts < max_attempts:
            attempts += 1
            neighbors = adjacency_list[current_index]
            neighbor_idxs = [n[0] for n in neighbors]
            neighbor_weights = [n[1] for n in neighbors]

            if not neighbor_idxs:
                # No neighbors, cannot proceed further
                break

            # Try to find unvisited neighbors
            unvisited_neighbors = [
                (idx, wgt)
                for idx, wgt in zip(neighbor_idxs, neighbor_weights)
                if idx not in visited
            ]
            if unvisited_neighbors:
                # Select among unvisited neighbors
                unvisited_idxs = [idx for idx, _ in unvisited_neighbors]
                unvisited_weights = [wgt for _, wgt in unvisited_neighbors]
                probabilities = np.array(unvisited_weights) / np.sum(unvisited_weights)
                next_index = rng.choice(unvisited_idxs, p=probabilities)
                visited.add(next_index)
            else:
                # All neighbors visited, but we need to reach length max_k
                # So we can revisit nodes
                probabilities = np.array(neighbor_weights) / np.sum(neighbor_weights)
                next_index = rng.choice(neighbor_idxs, p=probabilities)

            path.append(next_index)
            current_index = next_index

        if len(set(path)) >= k:
            # If path is shorter than max_k, extend it by repeating existing elements
            while len(path) < max_k:
                # Randomly select nodes from the existing path to repeat
                next_index = rng.choice(path)
                path.append(next_index)
            return path
        else:
            # Could not reach k unique nodes
            return None

    def __len__(self):
        return len(self.all_ref_imgs)

    def get_image_num(self):
        return self.num_imgs

    def get_stats(self):
        return f"{len(self)} imgs from {self.num_scenes} scenes"

    def generate_sequence(
        self, scene, adj_list, num_views, start_index, rng, allow_repeat=False
    ):
        cutoff = num_views if not allow_repeat else max(num_views // 5, 3)
        if start_index in self.is_reachable_cache[scene]:
            if not self.is_reachable_cache[scene][start_index]:
                print(
                    f"Cannot reach {num_views} unique elements from index {start_index}."
                )
                return None
        else:
            self.is_reachable_cache[scene][start_index] = self.is_reachable(
                adj_list, start_index, cutoff
            )
            if not self.is_reachable_cache[scene][start_index]:
                print(
                    f"Cannot reach {num_views} unique elements from index {start_index}."
                )
                return None
        if not allow_repeat:
            sequence = self.random_sequence_no_revisit_with_backtracking(
                adj_list, cutoff, start_index, rng
            )
        else:
            sequence = self.random_sequence_with_optional_repeats(
                adj_list, cutoff, start_index, rng, max_k=num_views
            )
        if not sequence:
            self.is_reachable_cache[scene][start_index] = False
            print("Failed to generate a sequence without revisiting.")
        return sequence

    def _get_views(self, idx, resolution, rng: np.random.Generator, num_views):
        scene_info, ref_img_idx = self.all_ref_imgs[idx]
        invalid_seq = True
        ordered_video = False

        while invalid_seq:
            basenames = self.data_dict[scene_info]["basenames"]
            if (
                sum(
                    [
                        (1 - int(x))
                        for x in list(self.is_reachable_cache[scene_info].values())
                    ]
                )
                > len(basenames) - self.num_views
            ):
                self.invalid_scenes.append(scene_info)
            while scene_info in self.invalid_scenes:
                idx = rng.integers(low=0, high=len(self.all_ref_imgs))
                scene_info, ref_img_idx = self.all_ref_imgs[idx]
                basenames = self.data_dict[scene_info]["basenames"]

            score_matrix = self.data_dict[scene_info]["score_matrix"]
            imgs_idxs = self.generate_sequence(
                scene_info, score_matrix, num_views, ref_img_idx, rng, self.allow_repeat
            )

            if imgs_idxs is None:
                random_direction = 2 * rng.choice(2) - 1
                for offset in range(1, len(basenames)):
                    tentative_im_idx = (
                        ref_img_idx + (random_direction * offset)
                    ) % len(basenames)
                    if (
                        tentative_im_idx not in self.is_reachable_cache[scene_info]
                        or self.is_reachable_cache[scene_info][tentative_im_idx]
                    ):
                        ref_img_idx = tentative_im_idx
                        break
            else:
                invalid_seq = False
        views = []
        for view_idx in imgs_idxs:
            scene_dir = osp.join(self.ROOT, scene_info)
            impath = basenames[view_idx].decode("utf-8")
            image = imread_cv2(osp.join(scene_dir, impath + ".jpg"))
            depthmap = imread_cv2(osp.join(scene_dir, impath + ".exr"))
            camera_params = np.load(osp.join(scene_dir, impath + ".npz"))

            intrinsics = np.float32(camera_params["intrinsics"])
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3, :3] = camera_params["R_cam2world"]
            camera_pose[:3, 3] = camera_params["t_cam2world"]

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(scene_dir, impath)
            )

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,  # cam2world
                    camera_intrinsics=intrinsics,
                    dataset="BlendedMVS",
                    label=osp.relpath(scene_dir, self.ROOT),
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    instance=osp.join(scene_dir, impath + ".jpg"),
                    quantile=np.array(0.97, dtype=np.float32),
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
