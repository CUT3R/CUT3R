#!/bin/bash

set -e

workdir='.'
model_name='ours'
ckpt_name='cut3r_512_dpt_4_64'
model_weights="${workdir}/src/${ckpt_name}.pth"

output_dir="${workdir}/eval_results/mv_recon/${model_name}_${ckpt_name}"
echo "$output_dir"
accelerate launch --num_processes 8 --main_process_port 29501 eval/mv_recon/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --model_name "$model_name"