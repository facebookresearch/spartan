# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer based on https://github.com/pytorch/vision/blob/main/references/classification/train.py"""

from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import torchvision
import os
import time
import math
import json
import datetime
import warnings
from omegaconf import OmegaConf
from pathlib import Path
import experiments.utils as utils
from experiments.data import DataConfig, get_train_loader
from spartan.sparsifier import Sparsifier, SparsifierConfig
from spartan.utils import get_logger

logger = get_logger(__name__)


@dataclass
class TrainConfig:
    # Model library (torchvision or timm)
    model_lib: str = "torchvision"
    # Model architecture
    arch: str = "resnet50"
    # Output directory for logs and checkpoints
    output_dir: str = os.getcwd()
    # Checkpoint for resuming training
    resume: Optional[str] = None
    # Checkpoint after every checkpoint_freq epochs
    checkpoint_freq: int = 2
    # Device to use for training (cpu or cuda)
    device: str = "cuda"
    # Training epochs
    num_epochs: int = 100
    # Optimizer (sgd or adamw)
    optimizer: str = "sgd"
    # Loss function (ce or bce)
    loss_fn: str = "ce"
    # Label smoothing parameter
    label_smoothing: float = 0.1
    # Learning rate schedule (currently only cosine)
    lr_schedule: str = "cosine"
    # Base learning rate
    lr: float = 1.0
    # Minimum learning rate
    lr_min: float = 1e-3
    # Number of epochs for linear LR warmup
    lr_warmup_epochs: int = 5
    # SGD Nesterov momentum setting
    momentum: float = 0.9
    # Base weight decay setting
    weight_decay: float = 1e-4
    # Weight decay to apply to bias terms (if specified)
    bias_weight_decay: Optional[float] = None
    # Weight decay to apply to normalization weights (if specified)
    norm_weight_decay: Optional[float] = None
    # Update mask after every `make_update_steps` minibatches
    mask_update_steps: int = 1
    # Clip gradient norm to value (if specified)
    clip_grad_norm: Optional[float] = None
    # Whether to force CUDA backend to run in deterministic mode
    use_deterministic_algorithms: bool = False
    # Whether to disable mixed precision training
    disable_amp: bool = False
    # Stochastic depth parameter (used e.g. in timm ViT implementation)
    drop_path_rate: float = 0.0

    # Training data configuration
    data: DataConfig = DataConfig()

    # Sparsification configuration
    sparsifier: SparsifierConfig = SparsifierConfig()

    # Distributed training config
    distributed: utils.DistributedConfig = utils.DistributedConfig()

    # Log file name
    log_file: str = "log.txt"
    # Output progress every `print_freq` batches
    print_freq: int = 10
    # Verbosity
    verbose: bool = True


def get_optimizer(config: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    norm_params = []
    bias_params = []
    remaining_params = []
    norm_classes = (
        torch.nn.modules.batchnorm._BatchNorm,
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.modules.instancenorm._InstanceNorm,
        torch.nn.LocalResponseNorm,
    )

    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if isinstance(module, norm_classes):
                norm_params.append(param)
            elif name == "bias":
                bias_params.append(param)
            else:
                remaining_params.append(param)

    parameters = [
        {
            "params": norm_params,
            "weight_decay": config.weight_decay
            if config.norm_weight_decay is None
            else config.norm_weight_decay,
        },
        {
            "params": bias_params,
            "weight_decay": config.weight_decay
            if config.bias_weight_decay is None
            else config.bias_weight_decay,
        },
        {"params": remaining_params, "weight_decay": config.weight_decay},
    ]

    optimizer_name = config.optimizer.lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Invalid optimizer: {config.optimizer}")

    return optimizer


def get_lr_scheduler(
    config: TrainConfig, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    if config.lr_schedule == "cosine":
        scheduler = utils.WarmupCosineLrScheduler(
            optimizer, config.num_epochs, config.lr_warmup_epochs, config.lr_min
        )
    else:
        raise ValueError(f"Invalid LR schedule: {config.lr_schedule}")
    return scheduler


def save_checkpoint(
    config: TrainConfig,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    sparsifier: Optional[Sparsifier],
    scaler: Optional[torch.cuda.amp.grad_scaler.GradScaler],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "sparsifier": sparsifier.state_dict() if sparsifier else None,
        "scaler": scaler.state_dict() if scaler else None,
    }
    checkpoint_path = Path(config.output_dir) / "checkpoint.pt"
    utils.save_on_master(checkpoint, checkpoint_path)
    if config.verbose:
        logger.info(f"Saved checkpoint {checkpoint_path}")


def setup() -> TrainConfig:
    default_config: TrainConfig = OmegaConf.structured(TrainConfig)
    cli_args = OmegaConf.from_cli()
    config: TrainConfig = OmegaConf.merge(default_config, cli_args)
    utils.init_distributed_mode(config.distributed)
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.backends.cuda.matmul.allow_tf32 = True

    if config.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    if (config.sparsifier.sparsity_mode == "spartan") and (config.mask_update_steps != 1):
        raise RuntimeError("Spartan is only compatible with mask_update_steps = 1")

    if utils.is_main_process():
        logger.info(f"CLI args:\n{OmegaConf.to_yaml(cli_args)}")
        logger.info(f"Training config:\n{OmegaConf.to_yaml(config)}")
        with open(Path(config.output_dir) / "train_config.yaml", "w") as f:
            OmegaConf.save(config, f)
    return config


def main():
    config = setup()
    train_start = time.time()

    # Load dataset
    train_loader, dataset_constants = get_train_loader(config.data, config.distributed)

    # Create model
    if config.model_lib == "torchvision":
        if config.drop_path_rate != 0.0:
            warnings.warn("drop_path_rate option only supported by timm model library")
        model = getattr(torchvision.models, config.arch)(num_classes=dataset_constants.num_classes)
    elif config.model_lib == "timm":
        import timm

        model = timm.create_model(
            config.arch,
            num_classes=dataset_constants.num_classes,
            drop_path_rate=config.drop_path_rate,
        )
    else:
        raise ValueError(f"Unsupported model library: {config.model_lib}")

    # Create sparsifier. This converts modules inplace to their masked counterparts.
    sparsifier: Optional[Sparsifier] = None
    if config.sparsifier.sparsity_mode is not None:
        sparsifier = Sparsifier(
            model,
            config.sparsifier,
            total_iters=config.num_epochs * len(train_loader),
            verbose=config.verbose,
        )
    model.to(device=config.device)
    model_without_ddp = model
    if config.distributed.is_distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[config.distributed.gpu])
        model_without_ddp = model.module

    # Create train classes
    if config.loss_fn == "ce":
        loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    elif config.loss_fn == "bce":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Invalid loss function: {config.loss_fn}")
    scaler = None if config.disable_amp else torch.cuda.amp.GradScaler()
    optimizer = get_optimizer(config, model)
    lr_scheduler = get_lr_scheduler(config, optimizer)
    start_epoch = 0

    # Load from checkpoint
    if config.resume is not None:
        checkpoint = torch.load(config.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        if sparsifier:
            sparsifier.load_state_dict(checkpoint["sparsifier"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        epoch_start = time.time()

        # Ensure that dataset is shuffled each epoch
        train_sampler = train_loader.sampler if hasattr(train_loader, "sampler") else None
        if train_sampler is None:
            os.environ["WDS_EPOCH"] = str(epoch)  # used by webdataset sampler
        elif config.distributed.is_distributed and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.2e}"))
        metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value:.1f}"))
        if config.sparsifier.sparsity_mode is not None:
            metric_logger.add_meter("cost", utils.SmoothedValue(window_size=1, fmt="{value:.3f}"))
        if config.sparsifier.sparsity_mode == "spartan":
            metric_logger.add_meter(
                "sinkhorn_beta", utils.SmoothedValue(window_size=1, fmt="{value:.3f}")
            )
            metric_logger.add_meter(
                "sinkhorn_iters", utils.SmoothedValue(window_size=1, fmt="{value:.3f}")
            )

        model.train()
        header = f"Epoch: [{epoch}]"
        for batch_idx, (inputs, labels) in enumerate(
            metric_logger.log_every(train_loader, config.print_freq, header)
        ):
            batch_start = time.time()
            with torch.cuda.amp.autocast(enabled=(not config.disable_amp)):
                inputs = inputs.to(device=config.device, non_blocking=True)
                labels = labels.to(device=config.device, non_blocking=True)
                if config.loss_fn == "bce":
                    labels = labels.gt(0.0).type(labels.dtype)
                logits = model(inputs)
                loss: torch.Tensor = loss_fn(logits, labels)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                raise RuntimeError("Loss is {}, stopping training".format(loss_value))

            # Measure accuracy and record loss
            acc1, acc5 = utils.accuracy(logits, labels, topk=(1, 5))
            batch_size = inputs.shape[0]
            metric_logger.update(loss=loss_value, lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters["acc1"].update(acc1, n=batch_size)
            metric_logger.meters["acc5"].update(acc5, n=batch_size)
            metric_logger.meters["img/s"].update(batch_size / (time.time() - batch_start))

            # Model update
            optimizer.zero_grad(set_to_none=True)
            if config.disable_amp:
                loss.backward()
                if config.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                if config.clip_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()

            # Update sparse masks
            if sparsifier and (batch_idx % config.mask_update_steps == 0):
                step_res = sparsifier.step()
                metric_logger.update(cost=step_res.cost_frac)
                if config.sparsifier.sparsity_mode == "spartan":
                    metric_logger.update(sinkhorn_beta=step_res.sinkhorn_beta)
                    metric_logger.update(sinkhorn_iters=step_res.sinkhorn_iters)

        lr_scheduler.step()

        # Checkpoint
        if (epoch % config.checkpoint_freq == 0) or (epoch == config.num_epochs - 1):
            save_checkpoint(
                config,
                epoch,
                model_without_ddp,
                optimizer,
                lr_scheduler,
                sparsifier,
                scaler,
            )

        # Write logs
        if sparsifier and epoch == 0:
            logger.info(f"Total cost = {sparsifier.total_cost()}")
        if utils.is_main_process():
            epoch_log = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "epoch": epoch,
                "epoch_time": time.time() - epoch_start,
                **{k: meter.global_avg for k, meter in metric_logger.meters.items()},
            }
            with (Path(config.output_dir) / config.log_file).open("a") as f:
                f.write(json.dumps(epoch_log) + "\n")

    # Converts masked modules to base modules with sparse parameters
    if sparsifier:
        sparsifier.finalize()
    utils.save_on_master(model_without_ddp.state_dict(), Path(config.output_dir) / "model.pt")
    total_time = time.time() - train_start
    logger.info(f"Training finished in {total_time:.1f}s")


if __name__ == "__main__":
    main()
