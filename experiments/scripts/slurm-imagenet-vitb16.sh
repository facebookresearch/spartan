#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --job-name=imagenet-vitb16
#SBATCH --cpus-per-task=12
#SBATCH --output=/tmp/imagenet-vitb16/out.%j.log
#SBATCH --error=/tmp/imagenet-vitb16/err.%j.log

# ViT-B/16 training with unstructured sparsity

# Terminate on error
set -e

# Find a free port
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export WORLD_SIZE=${SLURM_NTASKS}
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
echo "SLURM JOB ID = ${SLURM_JOB_ID}"
echo "WORLD SIZE = ${WORLD_SIZE}"
echo "MASTER PORT = ${MASTER_PORT}"
echo "NODELIST = ${SLURM_NODELIST}"
echo "MASTER_ADDR = ${MASTER_ADDR}"

SCRIPT_DIR=$(realpath $(dirname "$0"))
ROOT_DIR=$(dirname $(dirname ${SCRIPT_DIR}))
echo "Spartan root dir: ${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"

TRAIN_DATASET="/tmp/imagenet/train"  # Replace with own path
VAL_DATASET="/tmp/imagenet/val"
# TRAIN_DATASET="/tmp/imagenet/train/shard_{00000000..00001281}.tar"  # For sharded loading with WebDataset

# Training artifacts will be output here
TRAIN_DIR=/tmp/imagenet-vitb16/${SLURM_JOB_ID}
mkdir -p ${TRAIN_DIR}

# Activate conda env
source ~/miniconda/etc/profile.d/conda.sh
conda activate spartan

srun --unbuffered --output ${TRAIN_DIR}/%t.out --error ${TRAIN_DIR}/%t.err \
python -m experiments.train \
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
