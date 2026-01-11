#!/usr/bin/env python3
"""
Preprocessing Script for Spring Dataset

This script:
  - Recursively processes each sequence in a given 'root_dir' for the Spring dataset.
  - Reads RGB, disparity, optical flow files, and camera intrinsics/extrinsics.
  - Converts disparity to depth, rescales images/flows, and writes processed results
    (RGB, Depth, Cam intrinsics/poses, Forward Flow, Backward Flow) to 'out_dir'.

Usage:
    python preprocess_spring.py \
        --root_dir /path/to/spring/train \
        --out_dir /path/to/processed_spring \
        --baseline 0.065 \
        --output_size 960 540

"""

import os
import argparse
import numpy as np
import cv2
from PIL import Image
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Custom modules (adapt these imports to your actual module locations)
import flow_IO
import src.dust3r.datasets.utils.cropping as cropping


def rescale_flow(flow, size):
    """
    Resize an optical flow field to a new resolution and scale its vectors accordingly.

    Args:
        flow (np.ndarray): Flow array of shape [H, W, 2].
        size (tuple): Desired (width, height) for the resized flow.

    Returns:
        np.ndarray: Resized and scaled flow array.
    """
    h, w = flow.shape[:2]
    new_w, new_h = size

    # Resize the flow map
    flow_resized = cv2.resize(
        flow.astype("float32"), (new_w, new_h), interpolation=cv2.INTER_LINEAR
    )

    # Scale the flow vectors to match the new resolution
    flow_resized[..., 0] *= new_w / w
    flow_resized[..., 1] *= new_h / h

    return flow_resized


def get_depth(disparity, fx_baseline):
    """
    Convert disparity to depth using baseline * focal_length / disparity.

    Args:
        disparity (np.ndarray): Disparity map (same resolution as the RGB).
        fx_baseline (float): Product of the focal length (fx) and baseline.

    Returns:
        np.ndarray: Depth map.
    """
    # Avoid divide-by-zero
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid_mask = disparity != 0
    depth[valid_mask] = fx_baseline / disparity[valid_mask]
    return depth


def process_sequence(seq, root_dir, out_dir, baseline, output_size):
    """
    Process a single sequence from the Spring dataset:
      - Reads RGB frames, disparity maps, forward/backward optical flow, intrinsics, extrinsics.
      - Converts disparity to depth.
      - Rescales images, depth, and flow to the specified 'output_size'.
      - Saves the processed data to the output directory.

    Args:
        seq (str): Name of the sequence (subdirectory).
        root_dir (str): Root directory containing the Spring dataset sequences.
        out_dir (str): Output directory to store processed files.
        baseline (float): Stereo baseline for disparity-to-depth conversion (SPRING_BASELINE).
        output_size (tuple): (width, height) for output images and flows.

    Returns:
        None or str:
            - Returns None if processing is successful.
            - Returns an error message (str) if an error occurs.
    """
    seq_dir = os.path.join(root_dir, seq)
    img_dir = os.path.join(seq_dir, "frame_left")
    disp1_dir = os.path.join(seq_dir, "disp1_left")
    fflow_dir = os.path.join(seq_dir, "flow_FW_left")
    bflow_dir = os.path.join(seq_dir, "flow_BW_left")
    intrinsics_path = os.path.join(seq_dir, "cam_data", "intrinsics.txt")
    extrinsics_path = os.path.join(seq_dir, "cam_data", "extrinsics.txt")

    try:
        # Check required files/folders
        for path in (
            img_dir,
            disp1_dir,
            fflow_dir,
            bflow_dir,
            intrinsics_path,
            extrinsics_path,
        ):
            if not os.path.exists(path):
                return f"Missing required path: {path}"

        # Prepare output directories
        out_img_dir = os.path.join(out_dir, seq, "rgb")
        out_depth_dir = os.path.join(out_dir, seq, "depth")
        out_cam_dir = os.path.join(out_dir, seq, "cam")
        out_fflow_dir = os.path.join(out_dir, seq, "flow_forward")
        out_bflow_dir = os.path.join(out_dir, seq, "flow_backward")
        for d in [
            out_img_dir,
            out_depth_dir,
            out_cam_dir,
            out_fflow_dir,
            out_bflow_dir,
        ]:
            os.makedirs(d, exist_ok=True)

        # Read camera data
        all_intrinsics = np.loadtxt(intrinsics_path)
        all_extrinsics = np.loadtxt(extrinsics_path)

        # Collect filenames
        rgbs = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
        disps = sorted([f for f in os.listdir(disp1_dir) if f.endswith(".dsp5")])
        fflows = sorted([f for f in os.listdir(fflow_dir) if f.endswith(".flo5")])
        bflows = sorted([f for f in os.listdir(bflow_dir) if f.endswith(".flo5")])

        # Basic consistency check
        if not (len(all_intrinsics) == len(all_extrinsics) == len(rgbs) == len(disps)):
            return (
                f"Inconsistent lengths in {seq}: "
                f"Intrinsics {len(all_intrinsics)}, "
                f"Extrinsics {len(all_extrinsics)}, "
                f"RGBs {len(rgbs)}, "
                f"Disparities {len(disps)}"
            )
        # Note: fflows+1 == len(all_intrinsics), bflows+1 == len(all_intrinsics)

        # Check if already processed
        if len(os.listdir(out_img_dir)) == len(rgbs):
            return None  # Already done, skip

        # Process each frame
        for i in tqdm(
            range(len(all_intrinsics)), desc=f"Processing {seq}", leave=False
        ):
            frame_num = i + 1  # frames appear as 1-based in filenames
            img_path = os.path.join(img_dir, f"frame_left_{frame_num:04d}.png")
            disp1_path = os.path.join(disp1_dir, f"disp1_left_{frame_num:04d}.dsp5")
            fflow_path = None
            bflow_path = None

            if i < len(all_intrinsics) - 1:
                fflow_path = os.path.join(
                    fflow_dir, f"flow_FW_left_{frame_num:04d}.flo5"
                )
            if i > 0:
                bflow_path = os.path.join(
                    bflow_dir, f"flow_BW_left_{frame_num:04d}.flo5"
                )

            # Load image
            image = Image.open(img_path).convert("RGB")

            # Build the intrinsics matrix
            K = np.eye(3, dtype=np.float32)
            K[0, 0] = all_intrinsics[i][0]  # fx
            K[1, 1] = all_intrinsics[i][1]  # fy
            K[0, 2] = all_intrinsics[i][2]  # cx
            K[1, 2] = all_intrinsics[i][3]  # cy

            # Build the pose
            cam_ext = all_extrinsics[i].reshape(4, 4)
            pose = np.linalg.inv(cam_ext).astype(np.float32)
            if np.any(np.isinf(pose)) or np.any(np.isnan(pose)):
                return f"Invalid pose for frame {i} in {seq}"

            # Load disparity
            disp1 = flow_IO.readDispFile(disp1_path)
            # Subsample by 2
            disp1 = disp1[::2, ::2]

            # Convert disparity to depth
            fx_baseline = all_intrinsics[i][0] * baseline  # fx * baseline
            depth = get_depth(disp1, fx_baseline)
            depth[np.isinf(depth)] = 0.0
            depth[np.isnan(depth)] = 0.0

            # Load optical flows if available
            fflow = None
            bflow = None
            if fflow_path and os.path.exists(fflow_path):
                fflow = flow_IO.readFlowFile(fflow_path)
                fflow = fflow[::2, ::2]
            if bflow_path and os.path.exists(bflow_path):
                bflow = flow_IO.readFlowFile(bflow_path)
                bflow = bflow[::2, ::2]

            # Rescale image, depth, and intrinsics
            image, depth, K_scaled = cropping.rescale_image_depthmap(
                image, depth, K, output_size
            )
            W_new, H_new = image.size  # after rescale_image_depthmap

            # Rescale forward/backward flow
            if fflow is not None:
                fflow = rescale_flow(fflow, (W_new, H_new))
            if bflow is not None:
                bflow = rescale_flow(bflow, (W_new, H_new))

            # Save output
            out_index_str = f"{i:04d}"
            out_img_path = os.path.join(out_img_dir, out_index_str + ".png")
            image.save(out_img_path)

            out_depth_path = os.path.join(out_depth_dir, out_index_str + ".npy")
            np.save(out_depth_path, depth)

            out_cam_path = os.path.join(out_cam_dir, out_index_str + ".npz")
            np.savez(out_cam_path, intrinsics=K_scaled, pose=pose)

            if fflow is not None:
                out_fflow_path = os.path.join(out_fflow_dir, out_index_str + ".npy")
                np.save(out_fflow_path, fflow)
            if bflow is not None:
                out_bflow_path = os.path.join(out_bflow_dir, out_index_str + ".npy")
                np.save(out_bflow_path, bflow)

    except Exception as e:
        return f"Error processing sequence {seq}: {e}"

    return None  # success


def main():
    parser = argparse.ArgumentParser(description="Preprocess Spring dataset.")
    parser.add_argument(
        "--root_dir",
        required=True,
        help="Path to the root directory containing Spring dataset sequences.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Path to the output directory where processed files will be saved.",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=0.065,
        help="Stereo baseline for disparity-to-depth conversion (default: 0.065).",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        nargs=2,
        default=[960, 540],
        help="Output image size (width height) for rescaling.",
    )
    args = parser.parse_args()

    # Gather sequences
    if not os.path.isdir(args.root_dir):
        raise ValueError(f"Root directory not found: {args.root_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    seqs = sorted(
        [
            d
            for d in os.listdir(args.root_dir)
            if os.path.isdir(os.path.join(args.root_dir, d))
        ]
    )
    if not seqs:
        raise ValueError(f"No valid sequence folders found in {args.root_dir}")

    # Process each sequence in parallel
    with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
        future_to_seq = {
            executor.submit(
                process_sequence,
                seq,
                args.root_dir,
                args.out_dir,
                args.baseline,
                args.output_size,
            ): seq
            for seq in seqs
        }
        for future in tqdm(
            as_completed(future_to_seq),
            total=len(future_to_seq),
            desc="Processing all sequences",
        ):
            seq = future_to_seq[future]
            error = future.result()
            if error:
                print(f"Sequence '{seq}' failed: {error}")


if __name__ == "__main__":
    main()
