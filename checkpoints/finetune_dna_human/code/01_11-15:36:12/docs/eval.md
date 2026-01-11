# Evaluation Scripts

## Monodepth

```bash
bash eval/monodepth/run.sh
```
Results will be saved in `eval_results/monodepth/${data}_${model_name}/metric.json`.

### Video Depth

```bash
bash eval/video_depth/run.sh # You may need to change [--num_processes] to the number of your gpus
```
Results will be saved in `eval_results/video_depth/${data}_${model_name}/result_scale.json`.

### Camera Pose Estimation

```bash
bash eval/relpose/run.sh # You may need to change [--num_processes] to the number of your gpus
```
Results will be saved in `eval_results/relpose/${data}_${model_name}/_error_log.txt`.

### Multi-view Reconstruction

```bash
bash eval/mv_recon/run.sh # You may need to change [--num_processes] to the number of your gpus
```

Results will be saved in `eval_results/mv_recon/${model_name}_${ckpt_name}/logs_all.txt`.

