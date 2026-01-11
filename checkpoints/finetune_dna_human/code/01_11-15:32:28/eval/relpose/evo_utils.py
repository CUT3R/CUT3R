import os
import re
from copy import deepcopy
from pathlib import Path

import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
import matplotlib.pyplot as plt
import numpy as np
from evo.core import sync
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import file_interface, plot
from scipy.spatial.transform import Rotation
from evo.core import metrics


def sintel_cam_read(filename):
    """Read camera data, return (M,N) tuple.

    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    TAG_FLOAT = 202021.25

    f = open(filename, "rb")
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert (
        check == TAG_FLOAT
    ), " cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? ".format(
        TAG_FLOAT, check
    )
    M = np.fromfile(f, dtype="float64", count=9).reshape((3, 3))
    N = np.fromfile(f, dtype="float64", count=12).reshape((3, 4))
    return M, N


def load_replica_traj(gt_file):
    traj_w_c = np.loadtxt(gt_file)
    assert traj_w_c.shape[1] == 12 or traj_w_c.shape[1] == 16
    poses = [
        np.array(
            [
                [r[0], r[1], r[2], r[3]],
                [r[4], r[5], r[6], r[7]],
                [r[8], r[9], r[10], r[11]],
                [0, 0, 0, 1],
            ]
        )
        for r in traj_w_c
    ]

    pose_path = PosePath3D(poses_se3=poses)
    timestamps_mat = np.arange(traj_w_c.shape[0]).astype(float)

    traj = PoseTrajectory3D(poses_se3=pose_path.poses_se3, timestamps=timestamps_mat)
    xyz = traj.positions_xyz
    # shift -1 column -> w in back column
    # quat = np.roll(traj.orientations_quat_wxyz, -1, axis=1)
    # uncomment this line if the quaternion is in scalar-first format
    quat = traj.orientations_quat_wxyz

    traj_tum = np.column_stack((xyz, quat))
    return (traj_tum, timestamps_mat)


def load_sintel_traj(gt_file):  # './data/sintel/training/camdata_left/alley_2'
    # Refer to ParticleSfM
    gt_pose_lists = sorted(os.listdir(gt_file))
    gt_pose_lists = [
        os.path.join(gt_file, x) for x in gt_pose_lists if x.endswith(".cam")
    ]
    tstamps = [float(x.split("/")[-1][:-4].split("_")[-1]) for x in gt_pose_lists]
    gt_poses = [
        sintel_cam_read(f)[1] for f in gt_pose_lists
    ]  # [1] means get the extrinsic
    xyzs, wxyzs = [], []
    tum_gt_poses = []
    for gt_pose in gt_poses:
        gt_pose = np.concatenate([gt_pose, np.array([[0, 0, 0, 1]])], 0)
        gt_pose_inv = np.linalg.inv(gt_pose)  # world2cam -> cam2world
        xyz = gt_pose_inv[:3, -1]
        xyzs.append(xyz)
        R = Rotation.from_matrix(gt_pose_inv[:3, :3])
        xyzw = R.as_quat()  # scalar-last for scipy
        wxyz = np.array([xyzw[-1], xyzw[0], xyzw[1], xyzw[2]])
        wxyzs.append(wxyz)
        tum_gt_pose = np.concatenate([xyz, wxyz], 0)  # TODO: check if this is correct
        tum_gt_poses.append(tum_gt_pose)

    tum_gt_poses = np.stack(tum_gt_poses, 0)
    tum_gt_poses[:, :3] = tum_gt_poses[:, :3] - np.mean(
        tum_gt_poses[:, :3], 0, keepdims=True
    )
    tt = np.expand_dims(np.stack(tstamps, 0), -1)
    return tum_gt_poses, tt


def load_traj(gt_traj_file, traj_format="sintel", skip=0, stride=1, num_frames=None):
    """Read trajectory format. Return in TUM-RGBD format.
    Returns:
        traj_tum (N, 7): camera to world poses in (x,y,z,qx,qy,qz,qw)
        timestamps_mat (N, 1): timestamps
    """
    if traj_format == "replica":
        traj_tum, timestamps_mat = load_replica_traj(gt_traj_file)
    elif traj_format == "sintel":
        traj_tum, timestamps_mat = load_sintel_traj(gt_traj_file)
    elif traj_format in ["tum", "tartanair"]:
        traj = file_interface.read_tum_trajectory_file(gt_traj_file)
        xyz = traj.positions_xyz
        quat = traj.orientations_quat_wxyz
        timestamps_mat = traj.timestamps
        traj_tum = np.column_stack((xyz, quat))
    else:
        raise NotImplementedError

    traj_tum = traj_tum[skip::stride]
    timestamps_mat = timestamps_mat[skip::stride]
    if num_frames is not None:
        traj_tum = traj_tum[:num_frames]
        timestamps_mat = timestamps_mat[:num_frames]
    return traj_tum, timestamps_mat


def update_timestamps(gt_file, traj_format, skip=0, stride=1):
    """Update timestamps given a"""
    if traj_format == "tum":
        traj_t_map_file = gt_file.replace("groundtruth.txt", "rgb.txt")
        timestamps = load_timestamps(traj_t_map_file, traj_format)
        return timestamps[skip::stride]
    elif traj_format == "tartanair":
        traj_t_map_file = gt_file.replace("gt_pose.txt", "times.txt")
        timestamps = load_timestamps(traj_t_map_file, traj_format)
        return timestamps[skip::stride]


def load_timestamps(time_file, traj_format="replica"):
    if traj_format in ["tum", "tartanair"]:
        with open(time_file, "r+") as f:
            lines = f.readlines()
        timestamps_mat = [
            float(x.split(" ")[0]) for x in lines if not x.startswith("#")
        ]
        return timestamps_mat


def make_traj(args) -> PoseTrajectory3D:
    if isinstance(args, tuple) or isinstance(args, list):
        traj, tstamps = args
        return PoseTrajectory3D(
            positions_xyz=traj[:, :3],
            orientations_quat_wxyz=traj[:, 3:],
            timestamps=tstamps,
        )
    assert isinstance(args, PoseTrajectory3D), type(args)
    return deepcopy(args)


def eval_metrics(pred_traj, gt_traj=None, seq="", filename="", sample_stride=1):

    if sample_stride > 1:
        pred_traj[0] = pred_traj[0][::sample_stride]
        pred_traj[1] = pred_traj[1][::sample_stride]
        if gt_traj is not None:
            updated_gt_traj = []
            updated_gt_traj.append(gt_traj[0][::sample_stride])
            updated_gt_traj.append(gt_traj[1][::sample_stride])
            gt_traj = updated_gt_traj

    pred_traj = make_traj(pred_traj)

    if gt_traj is not None:
        gt_traj = make_traj(gt_traj)

        if pred_traj.timestamps.shape[0] == gt_traj.timestamps.shape[0]:
            pred_traj.timestamps = gt_traj.timestamps
        else:
            print(pred_traj.timestamps.shape[0], gt_traj.timestamps.shape[0])

        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

    # ATE
    traj_ref = gt_traj
    traj_est = pred_traj

    ate_result = main_ape.ape(
        traj_ref,
        traj_est,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=True,
    )

    ate = ate_result.stats["rmse"]
    # print(ate_result.np_arrays['error_array'])
    # exit()

    # RPE rotation and translation
    delta_list = [1]
    rpe_rots, rpe_transs = [], []
    for delta in delta_list:
        rpe_rots_result = main_rpe.rpe(
            traj_ref,
            traj_est,
            est_name="traj",
            pose_relation=PoseRelation.rotation_angle_deg,
            align=True,
            correct_scale=True,
            delta=delta,
            delta_unit=Unit.frames,
            rel_delta_tol=0.01,
            all_pairs=True,
        )

        rot = rpe_rots_result.stats["rmse"]
        rpe_rots.append(rot)

    for delta in delta_list:
        rpe_transs_result = main_rpe.rpe(
            traj_ref,
            traj_est,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=True,
            correct_scale=True,
            delta=delta,
            delta_unit=Unit.frames,
            rel_delta_tol=0.01,
            all_pairs=True,
        )

        trans = rpe_transs_result.stats["rmse"]
        rpe_transs.append(trans)

    rpe_trans, rpe_rot = np.mean(rpe_transs), np.mean(rpe_rots)
    with open(filename, "w+") as f:
        f.write(f"Seq: {seq} \n\n")
        f.write(f"{ate_result}")
        f.write(f"{rpe_rots_result}")
        f.write(f"{rpe_transs_result}")

    print(f"Save results to {filename}")
    return ate, rpe_trans, rpe_rot


def eval_metrics_first_pose_align_last_pose(
    pred_traj, gt_traj=None, seq="", filename="", figpath="", sample_stride=1
):
    if sample_stride > 1:
        pred_traj[0] = pred_traj[0][::sample_stride]
        pred_traj[1] = pred_traj[1][::sample_stride]
        if gt_traj is not None:
            gt_traj = [gt_traj[0][::sample_stride], gt_traj[1][::sample_stride]]
    pred_traj = make_traj(pred_traj)
    if gt_traj is not None:
        gt_traj = make_traj(gt_traj)

        if pred_traj.timestamps.shape[0] == gt_traj.timestamps.shape[0]:
            pred_traj.timestamps = gt_traj.timestamps
        else:
            print(
                "Different number of poses:",
                pred_traj.timestamps.shape[0],
                gt_traj.timestamps.shape[0],
            )

        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

    if gt_traj is not None and pred_traj is not None:
        if len(gt_traj.poses_se3) > 0 and len(pred_traj.poses_se3) > 0:
            first_gt_pose = gt_traj.poses_se3[0]
            first_pred_pose = pred_traj.poses_se3[0]
            # T = (first_gt_pose) * inv(first_pred_pose)
            T = first_gt_pose @ np.linalg.inv(first_pred_pose)

            # Apply T to every predicted pose
            aligned_pred_poses = []
            for pose in pred_traj.poses_se3:
                aligned_pred_poses.append(T @ pose)
            aligned_pred_traj = PoseTrajectory3D(
                poses_se3=aligned_pred_poses,
                timestamps=np.array(pred_traj.timestamps),
                # optionally copy other fields if your make_traj object has them
            )
            pred_traj = aligned_pred_traj  # .poses_se3 = aligned_pred_poses
        plot_trajectory(
            pred_traj,
            gt_traj,
            title=seq,
            filename=figpath,
            align=False,
            correct_scale=False,
        )

    if gt_traj is not None and len(gt_traj.poses_se3) > 0:
        gt_traj = PoseTrajectory3D(
            poses_se3=[gt_traj.poses_se3[-1]], timestamps=[gt_traj.timestamps[-1]]
        )
    if pred_traj is not None and len(pred_traj.poses_se3) > 0:
        pred_traj = PoseTrajectory3D(
            poses_se3=[pred_traj.poses_se3[-1]], timestamps=[pred_traj.timestamps[-1]]
        )

    ate_result = main_ape.ape(
        gt_traj,
        pred_traj,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=False,  # <-- important
        correct_scale=False,  # <-- important
    )
    ate = ate_result.stats["rmse"]
    with open(filename, "w+") as f:
        f.write(f"Seq: {seq}\n\n")
        f.write(f"{ate_result}")

    print(f"Save results to {filename}")

    return ate


def best_plotmode(traj):
    _, i1, i2 = np.argsort(np.var(traj.positions_xyz, axis=0))
    plot_axes = "xyz"[i2] + "xyz"[i1]
    return getattr(plot.PlotMode, plot_axes)


def plot_trajectory(
    pred_traj, gt_traj=None, title="", filename="", align=True, correct_scale=True
):
    pred_traj = make_traj(pred_traj)

    if gt_traj is not None:
        gt_traj = make_traj(gt_traj)
        if pred_traj.timestamps.shape[0] == gt_traj.timestamps.shape[0]:
            pred_traj.timestamps = gt_traj.timestamps
        else:
            print("WARNING", pred_traj.timestamps.shape[0], gt_traj.timestamps.shape[0])

        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

        if align:
            pred_traj.align(gt_traj, correct_scale=correct_scale)

    plot_collection = plot.PlotCollection("PlotCol")
    fig = plt.figure(figsize=(8, 8))
    plot_mode = best_plotmode(gt_traj if (gt_traj is not None) else pred_traj)
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(title)
    if gt_traj is not None:
        plot.traj(ax, plot_mode, gt_traj, "--", "gray", "Ground Truth")
    plot.traj(ax, plot_mode, pred_traj, "-", "blue", "Predicted")
    plot_collection.add_figure("traj_error", fig)
    plot_collection.export(filename, confirm_overwrite=False)
    plt.close(fig=fig)
    print(f"Saved trajectory to {filename.replace('.png','')}_traj_error.png")


def save_trajectory_tum_format(traj, filename):
    traj = make_traj(traj)
    tostr = lambda a: " ".join(map(str, a))
    with Path(filename).open("w") as f:
        for i in range(traj.num_poses):
            f.write(
                f"{traj.timestamps[i]} {tostr(traj.positions_xyz[i])} {tostr(traj.orientations_quat_wxyz[i][[0,1,2,3]])}\n"
            )
    print(f"Saved trajectory to {filename}")


def extract_metrics(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    # Extract metrics using regex
    ate_match = re.search(
        r"APE w.r.t. translation part \(m\).*?rmse\s+([0-9.]+)", content, re.DOTALL
    )
    rpe_trans_match = re.search(
        r"RPE w.r.t. translation part \(m\).*?rmse\s+([0-9.]+)", content, re.DOTALL
    )
    rpe_rot_match = re.search(
        r"RPE w.r.t. rotation angle in degrees \(deg\).*?rmse\s+([0-9.]+)",
        content,
        re.DOTALL,
    )

    ate = float(ate_match.group(1)) if ate_match else 0.0
    rpe_trans = float(rpe_trans_match.group(1)) if rpe_trans_match else 0.0
    rpe_rot = float(rpe_rot_match.group(1)) if rpe_rot_match else 0.0

    return ate, rpe_trans, rpe_rot


def process_directory(directory):
    results = []
    for root, _, files in os.walk(directory):
        if files is not None:
            files = sorted(files)
        for file in files:
            if file.endswith("_metric.txt"):
                file_path = os.path.join(root, file)
                seq_name = file.replace("_eval_metric.txt", "")
                ate, rpe_trans, rpe_rot = extract_metrics(file_path)
                results.append((seq_name, ate, rpe_trans, rpe_rot))

    return results


def calculate_averages(results):
    total_ate = sum(r[1] for r in results)
    total_rpe_trans = sum(r[2] for r in results)
    total_rpe_rot = sum(r[3] for r in results)
    count = len(results)

    if count == 0:
        return 0.0, 0.0, 0.0

    avg_ate = total_ate / count
    avg_rpe_trans = total_rpe_trans / count
    avg_rpe_rot = total_rpe_rot / count

    return avg_ate, avg_rpe_trans, avg_rpe_rot
