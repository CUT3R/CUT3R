# Training

Please note that this is an academic project, and due to resource constraints, we trained our model iteratively while exploring different configurations. As a result, releasing the complete training procedure is challenging. However, if you wish to train the model from scratch, we provide a set of configurations below that we believe are representative. For fine-tuning, we recommend starting with the scripts available [here](#fine-tuning). There are many design choices to consider, particularly under varying computational constraints, and we look forward to seeing the community explore these possibilities further.

## Training Configurations

You could refer to the following commands as a starting point if you would like to train from scratch.

```
# Remember to replace the dataset path to your own path
# the script has been tested on a 8xA100(80G) machine

cd src/

# stage 1, train 224+linear model on static datasets
NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py  --config-name stage1

# stage 2, finetune 224+linear model on all datasets
NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py  --config-name stage2

# stage 3, train 512+dpt model on all datasets
NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py  --config-name stage3

# stage 4, train 512+dpt model on long sequences (32 views)
NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py  --config-name stage4

# Finally, finetune 512+dpt model on 4-64 views
NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py  --config-name dpt_512_vary_4_64

```

## Fine-tuning

To fine-tune the released checkpoints, you can use the two provided config files as a starting point. Note that these configs correspond to the final stage of training, where the goal is to train the model to handle <strong>long sequences</strong>. Therefore, in these configs, the encoders are frozen, and single-view datasets are removed. You may adjust the configurations as needed to suit your requirements.

```
# Remember to replace the dataset path to your own path
# the script has been tested on a 8xA100(80G) machine

cd src/

# finetune 512 checkpoint
CUDA_LAUNCH_BLOCKING=1 NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py  --config-name dpt_512_vary_4_64

# finetune 224 checkpoint
CUDA_LAUNCH_BLOCKING=1 NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py  --config-name linear_224_fixed_16
```
