import os
import json
import os.path as osp
import decimal
import argparse
import math
from bisect import bisect_left
from PIL import Image
import numpy as np
import quaternion
from scipy import interpolate
import cv2
from tqdm import tqdm
from multiprocessing import Pool


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arkitscenes_dir",
        default="",
    )
    parser.add_argument(
        "--output_dir",
        default="data/dust3r_data/processed_arkitscenes_highres",
    )
    return parser


def value_to_decimal(value, decimal_places):
    decimal.getcontext().rounding = decimal.ROUND_HALF_UP  # define rounding method
    return decimal.Decimal(str(float(value))).quantize(
        decimal.Decimal("1e-{}".format(decimal_places))
    )


def closest(value, sorted_list):
    index = bisect_left(sorted_list, value)
    if index == 0:
        return sorted_list[0]
    elif index == len(sorted_list):
        return sorted_list[-1]
    else:
        value_before = sorted_list[index - 1]
        value_after = sorted_list[index]
        if value_after - value < value - value_before:
            return value_after
        else:
            return value_before


def get_up_vectors(pose_device_to_world):
    return np.matmul(pose_device_to_world, np.array([[0.0], [-1.0], [0.0], [0.0]]))


def get_right_vectors(pose_device_to_world):
    return np.matmul(pose_device_to_world, np.array([[1.0], [0.0], [0.0], [0.0]]))


def read_traj(traj_path):
    quaternions = []
    poses = []
    timestamps = []
    poses_p_to_w = []
    with open(traj_path) as f:
        traj_lines = f.readlines()
        for line in traj_lines:
            tokens = line.split()
            assert len(tokens) == 7
            traj_timestamp = float(tokens[0])

            timestamps_decimal_value = value_to_decimal(traj_timestamp, 3)
            timestamps.append(
                float(timestamps_decimal_value)
            )  # for spline interpolation

            angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
            r_w_to_p, _ = cv2.Rodrigues(np.asarray(angle_axis))
            t_w_to_p = np.asarray(
                [float(tokens[4]), float(tokens[5]), float(tokens[6])]
            )

            pose_w_to_p = np.eye(4)
            pose_w_to_p[:3, :3] = r_w_to_p
            pose_w_to_p[:3, 3] = t_w_to_p

            pose_p_to_w = np.linalg.inv(pose_w_to_p)

            r_p_to_w_as_quat = quaternion.from_rotation_matrix(pose_p_to_w[:3, :3])
            t_p_to_w = pose_p_to_w[:3, 3]
            poses_p_to_w.append(pose_p_to_w)
            poses.append(t_p_to_w)
            quaternions.append(r_p_to_w_as_quat)
    return timestamps, poses, quaternions, poses_p_to_w


def main(rootdir, outdir):
    os.makedirs(outdir, exist_ok=True)
    subdirs = ["Validation", "Training"]
    for subdir in subdirs:
        outsubdir = osp.join(outdir, subdir)
        scene_dirs = sorted(
            [
                d
                for d in os.listdir(osp.join(rootdir, subdir))
                if osp.isdir(osp.join(rootdir, subdir, d))
            ]
        )

        with Pool() as pool:
            results = list(
                tqdm(
                    pool.imap(
                        process_scene,
                        [
                            (rootdir, outdir, subdir, scene_subdir)
                            for scene_subdir in scene_dirs
                        ],
                    ),
                    total=len(scene_dirs),
                )
            )

        # Filter None results and other post-processing
        valid_scenes = [result for result in results if result is not None]
        outlistfile = osp.join(outsubdir, "scene_list.json")
        with open(outlistfile, "w") as f:
            json.dump(valid_scenes, f)


def process_scene(args):
    rootdir, outdir, subdir, scene_subdir = args
    # Unpack paths
    scene_dir = osp.join(rootdir, subdir, scene_subdir)
    outsubdir = osp.join(outdir, subdir)
    out_scene_subdir = osp.join(outsubdir, scene_subdir)

    # Validation if necessary resources exist
    if (
        not osp.exists(osp.join(scene_dir, "highres_depth"))
        or not osp.exists(osp.join(scene_dir, "vga_wide"))
        or not osp.exists(osp.join(scene_dir, "vga_wide_intrinsics"))
        or not osp.exists(osp.join(scene_dir, "lowres_wide.traj"))
    ):
        return None

    depth_dir = osp.join(scene_dir, "highres_depth")
    rgb_dir = osp.join(scene_dir, "vga_wide")
    intrinsics_dir = osp.join(scene_dir, "vga_wide_intrinsics")
    traj_path = osp.join(scene_dir, "lowres_wide.traj")

    depth_files = sorted(os.listdir(depth_dir))
    img_files = sorted(os.listdir(rgb_dir))

    out_scene_subdir = osp.join(outsubdir, scene_subdir)

    # STEP 3: parse the scene and export the list of valid (K, pose, rgb, depth) and convert images
    scene_metadata_path = osp.join(out_scene_subdir, "scene_metadata.npz")
    if osp.isfile(scene_metadata_path):
        print(f"Skipping {scene_subdir}")
    else:
        print(f"parsing {scene_subdir}")
        # loads traj
        timestamps, poses, quaternions, poses_cam_to_world = read_traj(traj_path)

        poses = np.array(poses)
        quaternions = np.array(quaternions, dtype=np.quaternion)
        quaternions = quaternion.unflip_rotors(quaternions)
        timestamps = np.array(timestamps)

        all_depths = sorted(
            [
                (basename, basename.split(".png")[0].split("_")[1])
                for basename in depth_files
            ],
            key=lambda x: float(x[1]),
        )

        selected_depths = []
        timestamps_selected = []
        timestamp_min = timestamps.min()
        timestamp_max = timestamps.max()
        for basename, frame_id in all_depths:
            frame_id = float(frame_id)
            if frame_id < timestamp_min or frame_id > timestamp_max:
                continue
            selected_depths.append((basename, frame_id))
            timestamps_selected.append(frame_id)

        sky_direction_scene, trajectories, intrinsics, images, depths = (
            convert_scene_metadata(
                scene_subdir,
                intrinsics_dir,
                timestamps,
                quaternions,
                poses,
                poses_cam_to_world,
                img_files,
                selected_depths,
                timestamps_selected,
            )
        )

        if len(images) == 0:
            print(f"Skipping {scene_subdir}")
            return None

        os.makedirs(out_scene_subdir, exist_ok=True)

        os.makedirs(os.path.join(out_scene_subdir, "vga_wide"), exist_ok=True)
        os.makedirs(os.path.join(out_scene_subdir, "highres_depth"), exist_ok=True)
        assert isinstance(sky_direction_scene, str)

        for image_path, depth_path in zip(images, depths):
            img_out = os.path.join(
                out_scene_subdir, "vga_wide", image_path.replace(".png", ".jpg")
            )
            depth_out = os.path.join(out_scene_subdir, "highres_depth", depth_path)
            if osp.isfile(img_out) and osp.isfile(depth_out):
                continue

            vga_wide_path = osp.join(rgb_dir, image_path)
            depth_path = osp.join(depth_dir, depth_path)

            if not osp.isfile(vga_wide_path) or not osp.isfile(depth_path):
                continue

            img = Image.open(vga_wide_path)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            # rotate the image
            if sky_direction_scene == "RIGHT":
                try:
                    img = img.transpose(Image.Transpose.ROTATE_90)
                except Exception:
                    img = img.transpose(Image.ROTATE_90)
                depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)

            elif sky_direction_scene == "LEFT":
                try:
                    img = img.transpose(Image.Transpose.ROTATE_270)
                except Exception:
                    img = img.transpose(Image.ROTATE_270)
                depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)

            elif sky_direction_scene == "DOWN":
                try:
                    img = img.transpose(Image.Transpose.ROTATE_180)
                except Exception:
                    img = img.transpose(Image.ROTATE_180)
                depth = cv2.rotate(depth, cv2.ROTATE_180)

            W, H = img.size
            if not osp.isfile(img_out):
                img.save(img_out)

            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            if not osp.isfile(
                depth_out
            ):  # avoid destroying the base dataset when you mess up the paths
                cv2.imwrite(depth_out, depth)

        # save at the end
        np.savez(
            scene_metadata_path,
            trajectories=trajectories,
            intrinsics=intrinsics,
            images=images,
        )


def convert_scene_metadata(
    scene_subdir,
    intrinsics_dir,
    timestamps,
    quaternions,
    poses,
    poses_cam_to_world,
    all_images,
    selected_depths,
    timestamps_selected,
):
    # find scene orientation
    sky_direction_scene, rotated_to_cam = find_scene_orientation(poses_cam_to_world)

    # find/compute pose for selected timestamps
    # most images have a valid timestamp / exact pose associated
    timestamps_selected = np.array(timestamps_selected)
    spline = interpolate.interp1d(timestamps, poses, kind="linear", axis=0)
    interpolated_rotations = quaternion.squad(
        quaternions, timestamps, timestamps_selected
    )
    interpolated_positions = spline(timestamps_selected)

    trajectories = []
    intrinsics = []
    images = []
    depths = []
    for i, (basename, frame_id) in enumerate(selected_depths):
        intrinsic_fn = osp.join(intrinsics_dir, f"{scene_subdir}_{frame_id}.pincam")
        search_interval = int(0.1 / 0.001)
        for timestamp in range(-search_interval, search_interval + 1):
            if osp.exists(intrinsic_fn):
                break
            intrinsic_fn = osp.join(
                intrinsics_dir,
                f"{scene_subdir}_{float(frame_id) + timestamp * 0.001:.3f}.pincam",
            )
        if not osp.exists(intrinsic_fn):
            print(f"Skipping {intrinsic_fn}")
            continue

        image_path = "{}_{}.png".format(scene_subdir, frame_id)
        search_interval = int(0.001 / 0.001)
        for timestamp in range(-search_interval, search_interval + 1):
            if image_path in all_images:
                break
            image_path = "{}_{}.png".format(
                scene_subdir, float(frame_id) + timestamp * 0.001
            )
        if image_path not in all_images:
            print(f"Skipping {scene_subdir} {frame_id}")
            continue

        w, h, fx, fy, hw, hh = np.loadtxt(intrinsic_fn)  # PINHOLE

        pose = np.eye(4)
        pose[:3, :3] = quaternion.as_rotation_matrix(interpolated_rotations[i])
        pose[:3, 3] = interpolated_positions[i]

        images.append(basename)
        depths.append(basename)
        if sky_direction_scene == "RIGHT" or sky_direction_scene == "LEFT":
            intrinsics.append([h, w, fy, fx, hh, hw])  # swapped intrinsics
        else:
            intrinsics.append([w, h, fx, fy, hw, hh])
        trajectories.append(
            pose @ rotated_to_cam
        )  # pose_cam_to_world @ rotated_to_cam = rotated(cam) to world

    return sky_direction_scene, trajectories, intrinsics, images, depths


def find_scene_orientation(poses_cam_to_world):
    if len(poses_cam_to_world) > 0:
        up_vector = sum(get_up_vectors(p) for p in poses_cam_to_world) / len(
            poses_cam_to_world
        )
        right_vector = sum(get_right_vectors(p) for p in poses_cam_to_world) / len(
            poses_cam_to_world
        )
        up_world = np.array([[0.0], [0.0], [1.0], [0.0]])
    else:
        up_vector = np.array([[0.0], [-1.0], [0.0], [0.0]])
        right_vector = np.array([[1.0], [0.0], [0.0], [0.0]])
        up_world = np.array([[0.0], [0.0], [1.0], [0.0]])

    # value between 0, 180
    device_up_to_world_up_angle = (
        np.arccos(np.clip(np.dot(np.transpose(up_world), up_vector), -1.0, 1.0)).item()
        * 180.0
        / np.pi
    )
    device_right_to_world_up_angle = (
        np.arccos(
            np.clip(np.dot(np.transpose(up_world), right_vector), -1.0, 1.0)
        ).item()
        * 180.0
        / np.pi
    )

    up_closest_to_90 = abs(device_up_to_world_up_angle - 90.0) < abs(
        device_right_to_world_up_angle - 90.0
    )
    if up_closest_to_90:
        assert abs(device_up_to_world_up_angle - 90.0) < 45.0
        # LEFT
        if device_right_to_world_up_angle > 90.0:
            sky_direction_scene = "LEFT"
            cam_to_rotated_q = quaternion.from_rotation_vector(
                [0.0, 0.0, math.pi / 2.0]
            )
        else:
            # note that in metadata.csv RIGHT does not exist, but again it's not accurate...
            # well, turns out there are scenes oriented like this
            # for example Training/41124801
            sky_direction_scene = "RIGHT"
            cam_to_rotated_q = quaternion.from_rotation_vector(
                [0.0, 0.0, -math.pi / 2.0]
            )
    else:
        # right is close to 90
        assert abs(device_right_to_world_up_angle - 90.0) < 45.0
        if device_up_to_world_up_angle > 90.0:
            sky_direction_scene = "DOWN"
            cam_to_rotated_q = quaternion.from_rotation_vector([0.0, 0.0, math.pi])
        else:
            sky_direction_scene = "UP"
            cam_to_rotated_q = quaternion.quaternion(1, 0, 0, 0)
    cam_to_rotated = np.eye(4)
    cam_to_rotated[:3, :3] = quaternion.as_rotation_matrix(cam_to_rotated_q)
    rotated_to_cam = np.linalg.inv(cam_to_rotated)
    return sky_direction_scene, rotated_to_cam


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.arkitscenes_dir, args.output_dir)
