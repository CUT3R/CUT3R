#!/bin/bash
set -e

workdir='.'
model_name='ours'
ckpt_name='cut3r_512_dpt_4_64'
model_weights="${workdir}/src/${ckpt_name}.pth"
datasets=('sintel' 'bonn' 'kitti' 'nyu')

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/monodepth/${data}_${model_name}"
    echo "$output_dir"
    python eval/monodepth/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data"
done

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/monodepth/${data}_${model_name}"
    python eval/monodepth/eval_metrics.py \
        --output_dir "$output_dir" \
        --eval_dataset "$data"
done

