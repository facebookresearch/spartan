#!/bin/bash

# ViT-B/16 training with unstructured sparsity

# Terminate on error
set -e

SCRIPT_DIR=$(realpath $(dirname "$0"))
ROOT_DIR=$(dirname $(dirname ${SCRIPT_DIR}))
echo "Spartan root dir: ${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"

TRAIN_DATASET="/tmp/imagenet/train"  # Replace with own path
VAL_DATASET="/tmp/imagenet/val"
# TRAIN_DATASET="/tmp/imagenet/train/shard_{00000000..00001281}.tar"  # For sharded loading with WebDataset

# Training artifacts will be output here
TRAIN_DIR=/tmp/imagenet-vitb16
mkdir -p ${TRAIN_DIR}

torchrun --nproc_per_node=8 -m experiments.train \
  arch=vit_b_16 \
  output_dir=${TRAIN_DIR} \
  num_epochs=100 \
  optimizer=adamw \
  lr=0.003 \
  lr_min=0 \
  lr_warmup_epochs=10 \
  clip_grad_norm=1 \
  weight_decay=0.3 \
  bias_weight_decay=0 \
  norm_weight_decay=0 \
  data.train_dataset_path=${TRAIN_DATASET} \
  data.valid_dataset_path=${VAL_DATASET} \
  data.batch_size_per_device=256 \
  data.mixup_alpha=0.2 \
  data.cutmix_alpha=1.0 \
  data.auto_augment=ra \
  sparsifier.sparsity_mode=spartan \
  sparsifier.sparsity_dist=global \
  sparsifier.cost_frac_init=1.0 \
  sparsifier.cost_frac_target=0.1 \
  sparsifier.finetune_frac=0.2 \
  sparsifier.warmup_frac=0.2 \
  sparsifier.sinkhorn_beta_target=20

python -m experiments.eval train_dir=${TRAIN_DIR}
