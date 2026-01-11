import subprocess
import os
import argparse
import glob


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mapfree_dir",
        default="mapfree/train/",
    )
    parser.add_argument(
        "--colmap_dir",
        default="mapfree/colmap",
    )
    parser.add_argument(
        "--output_dir",
        default="processed_mapfree",
    )
    return parser


def run_patch_match_stereo(root_colmap_dir, root_img_dir):
    scene_names = sorted(os.listdir(root_colmap_dir))
    sub_dir_names = ["seq0", "seq1"]
    for scene_name in scene_names:
        scene_dir = os.path.join(root_colmap_dir, scene_name)
        img_dir = os.path.join(root_img_dir, scene_name)
        for i, sub in enumerate(sub_dir_names):
            sub_dir = os.path.join(scene_dir, sub)
            out_dir = os.path.join(scene_dir, f"dense{i}")
            if not os.path.exists(sub_dir):
                continue
            if os.path.exists(out_dir) and os.path.exists(
                os.path.join(out_dir, f"stereo/depth_maps/{sub}")
            ):
                if len(
                    glob.glob(
                        os.path.join(out_dir, f"stereo/depth_maps/{sub}/*geometric.bin")
                    )
                ) == len(glob.glob(os.path.join(img_dir, sub, "*.jpg"))):
                    print(f"depth maps already computed, skip {sub_dir}")
                    continue

            print(sub_dir)
            cmd = f"colmap image_undistorter \
                    --image_path {img_dir} \
                    --input_path {sub_dir} \
                    --output_path {out_dir} \
                    --output_type COLMAP;"

            subprocess.call(cmd, shell=True)
            cmd = f"rm -rf {out_dir}/images/seq{i}; rm -rf {out_dir}/sparse;"
            cmd += f"cp -r {sub_dir} {out_dir}/sparse;"
            cmd += f"cp -r {img_dir}/{sub} {out_dir}/images;"
            subprocess.call(cmd, shell=True)

            # we comment this because we have released the mvs results, but feel free to re-run the mvs

            # cmd = f"colmap patch_match_stereo \
            #         --workspace_path {out_dir} \
            #         --workspace_format COLMAP \
            #         --PatchMatchStereo.cache_size 512 \
            #         --PatchMatchStereo.geom_consistency true"
            # subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    root_colmap_dir = args.colmap_dir
    root_img_dir = args.mapfree_dir

    # run patch match stereo
    run_patch_match_stereo(root_colmap_dir, root_img_dir)
