import os
import glob
from tqdm import tqdm

# Define the merged dataset metadata dictionary
dataset_metadata = {
    "davis": {
        "img_path": "data/davis/DAVIS/JPEGImages/480p",
        "mask_path": "data/davis/DAVIS/masked_images/480p",
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: None,
        "traj_format": None,
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: os.path.join(mask_path, seq),
        "skip_condition": None,
        "process_func": None,  # Not used in mono depth estimation
    },
    "kitti": {
        "img_path": "data/kitti/depth_selection/val_selection_cropped/image_gathered",  # Default path
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: None,
        "traj_format": None,
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_kitti(args, img_path),
    },
    "bonn": {
        "img_path": "data/bonn/rgbd_bonn_dataset",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(
            img_path, f"rgbd_bonn_{seq}", "rgb_110"
        ),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, f"rgbd_bonn_{seq}", "groundtruth_110.txt"
        ),
        "traj_format": "tum",
        "seq_list": ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"],
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_bonn(args, img_path),
    },
    "nyu": {
        "img_path": "data/nyu-v2/val/nyu_images",
        "mask_path": None,
        "process_func": lambda args, img_path: process_nyu(args, img_path),
    },
    "scannet": {
        "img_path": "data/scannetv2",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_90.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,  # lambda save_dir, seq: os.path.exists(os.path.join(save_dir, seq)),
        "process_func": lambda args, img_path: process_scannet(args, img_path),
    },
    "scannet-257": {
        "img_path": "data/scannetv2_3_257",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_90.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,  # lambda save_dir, seq: os.path.exists(os.path.join(save_dir, seq)),
        "process_func": lambda args, img_path: process_scannet(args, img_path),
    },
    "scannet-129": {
        "img_path": "data/scannetv2_3_129",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_90.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,  # lambda save_dir, seq: os.path.exists(os.path.join(save_dir, seq)),
        "process_func": lambda args, img_path: process_scannet(args, img_path),
    },
    "scannet-65": {
        "img_path": "data/scannetv2_3_65",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_90.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,  # lambda save_dir, seq: os.path.exists(os.path.join(save_dir, seq)),
        "process_func": lambda args, img_path: process_scannet(args, img_path),
    },
    "scannet-33": {
        "img_path": "data/scannetv2_3_33",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_90.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,  # lambda save_dir, seq: os.path.exists(os.path.join(save_dir, seq)),
        "process_func": lambda args, img_path: process_scannet(args, img_path),
    },
    "tum": {
        "img_path": "data/tum",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "rgb_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "groundtruth_90.txt"
        ),
        "traj_format": "tum",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "sintel": {
        "img_path": "data/sintel/training/final",
        "anno_path": "data/sintel/training/camdata_left",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(anno_path, seq),
        "traj_format": None,
        "seq_list": [
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
        ],
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_sintel(args, img_path),
    },
}


# Define processing functions for each dataset
def process_kitti(args, img_path):
    for dir in tqdm(sorted(glob.glob(f"{img_path}/*"))):
        filelist = sorted(glob.glob(f"{dir}/*.png"))
        save_dir = f"{args.output_dir}/{os.path.basename(dir)}"
        yield filelist, save_dir


def process_bonn(args, img_path):
    if args.full_seq:
        for dir in tqdm(sorted(glob.glob(f"{img_path}/*/"))):
            filelist = sorted(glob.glob(f"{dir}/rgb/*.png"))
            save_dir = f"{args.output_dir}/{os.path.basename(os.path.dirname(dir))}"
            yield filelist, save_dir
    else:
        seq_list = (
            ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]
            if args.seq_list is None
            else args.seq_list
        )
        for seq in tqdm(seq_list):
            filelist = sorted(glob.glob(f"{img_path}/rgbd_bonn_{seq}/rgb_110/*.png"))
            save_dir = f"{args.output_dir}/{seq}"
            yield filelist, save_dir


def process_nyu(args, img_path):
    filelist = sorted(glob.glob(f"{img_path}/*.png"))
    save_dir = f"{args.output_dir}"
    yield filelist, save_dir


def process_scannet(args, img_path):
    seq_list = sorted(glob.glob(f"{img_path}/*"))
    for seq in tqdm(seq_list):
        filelist = sorted(glob.glob(f"{seq}/color_90/*.jpg"))
        save_dir = f"{args.output_dir}/{os.path.basename(seq)}"
        yield filelist, save_dir


def process_sintel(args, img_path):
    if args.full_seq:
        for dir in tqdm(sorted(glob.glob(f"{img_path}/*/"))):
            filelist = sorted(glob.glob(f"{dir}/*.png"))
            save_dir = f"{args.output_dir}/{os.path.basename(os.path.dirname(dir))}"
            yield filelist, save_dir
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
        for seq in tqdm(seq_list):
            filelist = sorted(glob.glob(f"{img_path}/{seq}/*.png"))
            save_dir = f"{args.output_dir}/{seq}"
            yield filelist, save_dir
