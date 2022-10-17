# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
import time
from omegaconf import OmegaConf, MISSING
from .train import TrainConfig
from .data import DataConfig, get_validation_loader
import experiments.utils as utils
from spartan.utils import get_logger


logger = get_logger(__name__)


@dataclass
class EvalConfig:
    data: DataConfig = DataConfig()
    train_dir: str = MISSING
    output_file: Optional[str] = None
    print_freq: int = 10
    device: str = "cuda"
    verbose: bool = False


def compute_model_sparsity_fraction(model: nn.Module) -> float:
    total_zeros = sum([(p == 0).sum().item() for p in model.parameters()])
    total_params = sum([p.numel() for p in model.parameters()])
    return total_zeros / total_params


def setup() -> Tuple[EvalConfig, TrainConfig]:
    default_config = OmegaConf.structured(EvalConfig)
    eval_config: EvalConfig = OmegaConf.merge(default_config, OmegaConf.from_cli())
    train_config: TrainConfig = OmegaConf.load(Path(eval_config.train_dir) / "train_config.yaml")
    logger.info(f"Train config:\n{OmegaConf.to_yaml(train_config)}")
    logger.info(f"Eval config:\n{OmegaConf.to_yaml(eval_config)}")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return eval_config, train_config


def main():
    eval_config, train_config = setup()
    eval_start = time.time()

    # Load data
    eval_loader, dataset_constants = get_validation_loader(eval_config.data)

    # Create model
    if train_config.model_lib == "torchvision":
        model: nn.Module = getattr(models, train_config.arch)(
            num_classes=dataset_constants.num_classes
        )
    elif train_config.model_lib == "timm":
        import timm

        model: nn.Module = timm.create_model(
            train_config.arch, num_classes=dataset_constants.num_classes
        )
    else:
        raise ValueError(f"Unsupported model library: {train_config.model_lib}")

    # Load parameters
    state_dict = torch.load(Path(eval_config.train_dir) / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval().to(device=eval_config.device)

    # Run evaluation
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Eval: "
    for inputs, labels in metric_logger.log_every(eval_loader, eval_config.print_freq, header):
        with torch.inference_mode():
            inputs = inputs.to(eval_config.device, non_blocking=True)
            labels = labels.to(eval_config.device, non_blocking=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)

            # measure accuracy and record loss
            acc1, acc5 = utils.accuracy(logits, labels, topk=(1, 5))
            batch_size = inputs.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1, n=batch_size)
            metric_logger.meters["acc5"].update(acc5, n=batch_size)

    if train_config.sparsifier.sparsity_mode is not None:
        print(f"sparsity % = {compute_model_sparsity_fraction(model):.4f}")

    loss_avg = metric_logger.meters["loss"].global_avg
    acc1_avg = metric_logger.meters["acc1"].global_avg
    acc5_avg = metric_logger.meters["acc5"].global_avg
    print(f"loss: {loss_avg:.4f}")
    print(f"acc@1: {acc1_avg:.4f}")
    print(f"acc@5: {acc5_avg:.4f}")
    if eval_config.output_file:
        torch.save(
            {
                "loss": loss_avg,
                "top1": acc1_avg,
                "top5": acc5_avg,
            },
            os.path.join(eval_config.train_dir, eval_config.output_file),
        )

    total_time = time.time() - eval_start
    logger.info(f"Eval finished in {total_time:.1f}s")


if __name__ == "__main__":
    main()
