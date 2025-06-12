#!/usr/bin/env bash

if [ "$DEBUG" = "1" ]; then
  set -x
fi

# 使用层次化扩散模型的训练脚本

# Get the number of GPUs from CUDA_VISIBLE_DEVICES
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NGPUS=$(nvidia-smi --list-gpus | wc -l)
else
    NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr -cd ',' | wc -c)
    NGPUS=$((NGPUS + 1))
fi

PORT=${1:-29500}
shift 1

echo "Training HierarchicalDiffusionPaCo with $NGPUS GPUs"

# 强制使用层次化扩散模型配置
if [ $NGPUS -gt 1 ]; then
    # 分布式训练
    torchrun --master_port=${PORT} --nproc_per_node=${NGPUS} train.py \
        distributed=true \
        model=hierarchical_diffusion_paco \
        "$@"
else
    # 单GPU训练
    export LOCAL_RANK=0
    python train.py \
        model=hierarchical_diffusion_paco \
        "$@"
fi

# 使用方法:
# CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_hierarchical_diffusion.sh
# CUDA_VISIBLE_DEVICES=0 ./scripts/train_hierarchical_diffusion.sh exp_name=my_hierarchical_exp