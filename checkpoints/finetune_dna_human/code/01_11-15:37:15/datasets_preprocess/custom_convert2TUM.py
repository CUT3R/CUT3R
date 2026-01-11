import os
import json
import shutil
import numpy as np
import cv2 as cv
import imageio
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import open3d as o3d
import scipy.ndimage
import pickle

# Set environment variable to limit OpenBLAS threads
os.environ["OPENBLAS_NUM_THREADS"] = "1"

DEPTH_SCALE_FACTOR = 5000


# Point cloud from depth
def pointcloudify_depth(depth, intrinsics, dist_coeff, undistort=True):
    shape = depth.shape[::-1]

    if undistort:
        undist_intrinsics, _ = cv.getOptimalNewCameraMatrix(
            intrinsics, dist_coeff, shape, 1, shape
        )
        inv_undist_intrinsics = np.linalg.inv(undist_intrinsics)

        map_x, map_y = cv.initUndistortRectifyMap(
            intrinsics, dist_coeff, None, undist_intrinsics, shape, cv.CV_32FC1
        )
        undist_depth = cv.remap(depth, map_x, map_y, cv.INTER_NEAREST)
    else:
        inv_undist_intrinsics = np.linalg.inv(intrinsics)
        undist_depth = depth

    # Generate x,y grid for H x W image
    grid_x, grid_y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    grid = np.stack((grid_x, grid_y, np.ones_like(grid_x)), axis=-1)

    # Reshape and compute local grid
    grid_flat = grid.reshape(-1, 3).T
    local_grid = inv_undist_intrinsics @ grid_flat

    # Multiply by depth
    local_grid = local_grid.T * undist_depth.reshape(-1, 1)

    return local_grid.astype(np.float32)


def project_pcd_to_depth(pcd, undist_intrinsics, img_size, config):
    h, w = img_size
    points = np.asarray(pcd.points)
    d = points[:, 2]
    normalized_points = points / points[:, 2][:, np.newaxis]
    proj_pcd = np.round((undist_intrinsics @ normalized_points.T).T).astype(np.int64)
    proj_mask = (
        (proj_pcd[:, 0] >= 0)
        & (proj_pcd[:, 0] < w)
        & (proj_pcd[:, 1] >= 0)
        & (proj_pcd[:, 1] < h)
    )
    proj_pcd = proj_pcd[proj_mask]
    d = d[proj_mask]
    pcd_image = np.zeros((config["res_h"], config["res_w"]), dtype=np.float32)
    pcd_image[proj_pcd[:, 1], proj_pcd[:, 0]] = d
    return pcd_image


def smooth_depth(depth):
    MAX_DEPTH_VAL = 1e5
    KERNEL_SIZE = 11
    depth = depth.copy()
    depth[depth == 0] = MAX_DEPTH_VAL
    smoothed_depth = scipy.ndimage.minimum_filter(depth, KERNEL_SIZE)
    smoothed_depth[smoothed_depth == MAX_DEPTH_VAL] = 0
    return smoothed_depth


def align_rgb_depth(rgb, depth, roi, config, rgb_cnf, config_dict, T):
    # Undistort rgb image
    undist_rgb = cv.undistort(
        rgb,
        rgb_cnf["intrinsics"],
        rgb_cnf["dist_coeff"],
        None,
        rgb_cnf["undist_intrinsics"],
    )

    # Create point cloud from depth
    pcd = o3d.geometry.PointCloud()
    points = pointcloudify_depth(
        depth, config_dict["depth"]["dist_mtx"], config_dict["depth"]["dist_coef"]
    )
    pcd.points = o3d.utility.Vector3dVector(points)
    # Align point cloud with depth reference frame
    pcd.transform(T)

    # Project aligned point cloud to rgb
    aligned_depth = project_pcd_to_depth(
        pcd, rgb_cnf["undist_intrinsics"], rgb.shape[:2], config
    )

    smoothed_aligned_depth = smooth_depth(aligned_depth)
    x, y, w, h = roi

    depth_res = smoothed_aligned_depth[y : y + h, x : x + w]
    rgb_res = undist_rgb[y : y + h, x : x + w]
    return rgb_res, depth_res, rgb_cnf["undist_intrinsics"]


def process_pair(args):
    (
        pair,
        smartphone_folder,
        azure_depth_folder,
        final_folder,
        config,
        rgb_cnf,
        config_dict,
        T,
    ) = args
    try:
        rgb_image = cv.imread(os.path.join(smartphone_folder, f"{pair[0]}.png"))
        depth_array = np.load(
            os.path.join(azure_depth_folder, f"{pair[1]}.npy"), allow_pickle=True
        )

        rgb_image_aligned, depth_array_aligned, intrinsics = align_rgb_depth(
            rgb_image,
            depth_array,
            (0, 0, config["res_w"], config["res_h"]),
            config,
            rgb_cnf,
            config_dict,
            T,
        )
        # Save rgb as 8-bit png
        cv.imwrite(
            os.path.join(final_folder, "rgb", f"{pair[0]}.png"), rgb_image_aligned
        )

        # # Save depth as 16-bit unsigned int with scale factor
        # depth_array_aligned = (depth_array_aligned *
        #                        DEPTH_SCALE_FACTOR).astype(np.uint16)
        # imageio.imwrite(os.path.join(final_folder, 'depth', f"{pair[1]}.png"), depth_array_aligned)
        np.save(
            os.path.join(final_folder, "depth", f"{pair[0]}.npy"), depth_array_aligned
        )
        np.savez(
            os.path.join(final_folder, "cam", f"{pair[0]}.npz"), intrinsics=intrinsics
        )
    except Exception as e:
        return f"Error processing pair {pair}: {e}"
    return None


def main():
    DATA_DIR_ = "data_smartportraits/SmartPortraits"  # REPLACE WITH YOUR OWN DATA PATH!
    DATA_DIR = DATA_DIR_.rstrip("/")
    print(f"{DATA_DIR_} {DATA_DIR}/")

    # Folder where the data in TUM format will be put
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(curr_dir, "config.json")) as conf_f:
        config = json.load(conf_f)

    # Pre-load shared data
    with open(os.path.join(curr_dir, config["depth_conf"]), "rb") as config_f:
        config_dict = pickle.load(config_f)

    rgb_cnf = np.load(
        os.path.join(curr_dir, config["rgb_intristics"]), allow_pickle=True
    ).item()

    T = np.load(os.path.join(curr_dir, config["transform_intristics"]))

    final_root = "processed_smartportraits1"  # REPLACE WITH YOUR OWN DATA PATH!

    seqs = []
    for scene in os.listdir(DATA_DIR):
        scene_path = os.path.join(DATA_DIR, scene)
        if not os.path.isdir(scene_path):
            continue
        for s in os.listdir(scene_path):
            s_path = os.path.join(scene_path, s)
            if not os.path.isdir(s_path):
                continue
            for date in os.listdir(s_path):
                date_path = os.path.join(s_path, date)
                if os.path.isdir(date_path):
                    seqs.append((scene, s, date))

    for seq in tqdm(seqs):
        scene, s, date = seq
        dataset_path = os.path.join(DATA_DIR, scene, s, date)
        final_folder = os.path.join(final_root, "_".join([scene, s, date]))

        azure_depth_folder = os.path.join(dataset_path, "_azure_depth_image_raw")
        smartphone_folder = os.path.join(dataset_path, "smartphone_video_frames")

        depth_files = [
            file for file in os.listdir(azure_depth_folder) if file.endswith(".npy")
        ]
        depth_ts = np.array([int(file.split(".")[0]) for file in depth_files])
        depth_ts.sort()

        rgb_files = [
            file for file in os.listdir(smartphone_folder) if file.endswith(".png")
        ]
        rgb_ts = np.array([int(file.split(".")[0]) for file in rgb_files])
        rgb_ts.sort()

        print(
            f"Depth timestamps from {depth_ts[0]} to {depth_ts[-1]} (cnt {len(depth_ts)})"
        )
        print(f"RGB timestamps from {rgb_ts[0]} to {rgb_ts[-1]} (cnt {len(rgb_ts)})")

        # Build correspondences between depth and rgb by nearest neighbour algorithm
        rgbd_pairs = []
        for depth_t in depth_ts:
            idx = np.argmin(np.abs(rgb_ts - depth_t))
            closest_rgb_t = rgb_ts[idx]
            rgbd_pairs.append((closest_rgb_t, depth_t))

        # Prepare folder infrastructure
        if os.path.exists(final_folder):
            shutil.rmtree(final_folder)
        os.makedirs(os.path.join(final_folder, "depth"), exist_ok=True)
        os.makedirs(os.path.join(final_folder, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(final_folder, "cam"), exist_ok=True)

        # Prepare arguments for processing
        tasks = [
            (
                pair,
                smartphone_folder,
                azure_depth_folder,
                final_folder,
                config,
                rgb_cnf,
                config_dict,
                T,
            )
            for pair in rgbd_pairs
        ]

        num_workers = os.cpu_count()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_pair, task): task[0] for task in tasks}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Processing {scene}_{s}_{date}",
            ):
                error = future.result()
                if error:
                    print(error)


if __name__ == "__main__":
    main()
