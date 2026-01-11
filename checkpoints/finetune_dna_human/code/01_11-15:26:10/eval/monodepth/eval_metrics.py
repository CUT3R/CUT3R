import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from eval.monodepth.tools import depth_evaluation
import numpy as np
import json
from tqdm import tqdm
import glob
import cv2
from eval.monodepth.metadata import dataset_metadata
import argparse
from PIL import Image

TAG_FLOAT = 202021.25


def depth_read_sintel(filename):
    """Read depth data from file, return as numpy array."""
    f = open(filename, "rb")
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert (
        check == TAG_FLOAT
    ), " depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? ".format(
        TAG_FLOAT, check
    )
    width = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    size = width * height
    assert (
        width > 0 and height > 0 and size > 1 and size < 100000000
    ), " depth_read:: Wrong input size (width = {0}, height = {1}).".format(
        width, height
    )
    depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    return depth


def depth_read_bonn(filename):
    # loads depth map D from png file
    # and returns it as a numpy array
    depth_png = np.asarray(Image.open(filename))
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255
    depth = depth_png.astype(np.float64) / 5000.0
    depth[depth_png == 0] = -1.0
    return depth


def depth_read_kitti(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    img_pil = Image.open(filename)
    depth_png = np.array(img_pil, dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255

    depth = depth_png.astype(float) / 256.0
    depth[depth_png == 0] = -1.0
    return depth


def get_gt_depth(filename, dataset):
    if dataset == "sintel":
        return depth_read_sintel(filename)
    elif dataset == "bonn":
        return depth_read_bonn(filename)
    elif dataset == "kitti":
        return depth_read_kitti(filename)
    elif dataset == "nyu":
        return np.load(filename)
    else:
        raise NotImplementedError


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument(
        "--eval_dataset", type=str, default="nyu", choices=list(dataset_metadata.keys())
    )
    return parser


def main(args):
    if args.eval_dataset == "nyu":
        depth_pathes = glob.glob("data/nyu-v2/val/nyu_depths/*.npy")
        depth_pathes = sorted(depth_pathes)
        pred_pathes = glob.glob(
            f"{args.output_dir}/*.npy"
        )  # TODO: update the path to your prediction
        pred_pathes = sorted(pred_pathes)
    elif args.eval_dataset == "sintel":
        pred_pathes = glob.glob(
            f"{args.output_dir}/*/*.npy"
        )  # TODO: update the path to your prediction
        pred_pathes = sorted(pred_pathes)
        full = len(pred_pathes) > 643
        if full:
            depth_pathes = glob.glob(f"data/sintel/training/depth/*/*.dpt")
            depth_pathes = sorted(depth_pathes)
        else:
            seq_list = [
                "alley_2",
                "ambush_4",
                "ambush_5",
                "ambush_6",
                "cave_2",
                "cave_4",
                "market_2",
                "market_5",
                "market_6",
                "shaman_3",
                "sleeping_1",
                "sleeping_2",
                "temple_2",
                "temple_3",
            ]
            depth_pathes_folder = [
                f"data/sintel/training/depth/{seq}" for seq in seq_list
            ]
            depth_pathes = []
            for depth_pathes_folder_i in depth_pathes_folder:
                depth_pathes += glob.glob(depth_pathes_folder_i + "/*.dpt")
            depth_pathes = sorted(depth_pathes)
    elif args.eval_dataset == "bonn":
        seq_list = ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]
        img_pathes_folder = [
            f"data/bonn/rgbd_bonn_dataset/rgbd_bonn_{seq}/rgb_110/*.png"
            for seq in seq_list
        ]
        img_pathes = []
        for img_pathes_folder_i in img_pathes_folder:
            img_pathes += glob.glob(img_pathes_folder_i)
        img_pathes = sorted(img_pathes)
        depth_pathes_folder = [
            f"data/bonn/rgbd_bonn_dataset/rgbd_bonn_{seq}/depth_110/*.png"
            for seq in seq_list
        ]
        depth_pathes = []
        for depth_pathes_folder_i in depth_pathes_folder:
            depth_pathes += glob.glob(depth_pathes_folder_i)
        depth_pathes = sorted(depth_pathes)
        pred_pathes = glob.glob(
            f"{args.output_dir}/*/*.npy"
        )  # TODO: update the path to your prediction
        pred_pathes = sorted(pred_pathes)
    elif args.eval_dataset == "kitti":
        depth_pathes = glob.glob(
            "data/kitti/depth_selection/val_selection_cropped/groundtruth_depth_gathered/*/*.png"
        )
        depth_pathes = sorted(depth_pathes)
        pred_pathes = glob.glob(
            f"{args.output_dir}/*/*depth.npy"
        )  # TODO: update the path to your prediction
        pred_pathes = sorted(pred_pathes)
    else:
        raise NotImplementedError

    gathered_depth_metrics = []
    for idx in tqdm(range(len(depth_pathes))):
        pred_depth = np.load(pred_pathes[idx])
        gt_depth = get_gt_depth(depth_pathes[idx], args.eval_dataset)
        pred_depth = cv2.resize(
            pred_depth,
            (gt_depth.shape[1], gt_depth.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        if args.eval_dataset == "nyu":
            depth_results, error_map, depth_predict, depth_gt = depth_evaluation(
                pred_depth, gt_depth, max_depth=None, lr=1e-3
            )
        elif args.eval_dataset == "sintel":
            depth_results, error_map, depth_predict, depth_gt = depth_evaluation(
                pred_depth, gt_depth, max_depth=70, use_gpu=True, post_clip_max=70
            )
        elif args.eval_dataset == "bonn":
            depth_results, error_map, depth_predict, depth_gt = depth_evaluation(
                pred_depth, gt_depth, max_depth=70, use_gpu=True
            )
        elif args.eval_dataset == "kitti":
            depth_results, error_map, depth_predict, depth_gt = depth_evaluation(
                pred_depth, gt_depth, max_depth=None, use_gpu=True
            )
        gathered_depth_metrics.append(depth_results)

    depth_log_path = os.path.join(args.output_dir, "metric.json")
    average_metrics = {
        key: np.average(
            [metrics[key] for metrics in gathered_depth_metrics],
            weights=[metrics["valid_pixels"] for metrics in gathered_depth_metrics],
        )
        for key in gathered_depth_metrics[0].keys()
        if key != "valid_pixels"
    }
    print(f"{args.eval_dataset} - Average depth evaluation metrics:", average_metrics)
    with open(depth_log_path, "w") as f:
        f.write(json.dumps(average_metrics))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
