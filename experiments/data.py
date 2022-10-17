# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple
from braceexpand import braceexpand
from dataclasses import dataclass
import webdataset as wds
import math
import torch
import torch.distributed as dist
from torch.utils.data.dataloader import default_convert, default_collate
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from experiments.transforms import RandomCutmix, RandomMixup


@dataclass
class DataConfig:
    dataset: str = "imagenet"
    train_dataset_path: str = (
        "/datasets01/imagenet/imagenet_shared_1000/train/shard_{00000000..00001281}.tar"
    )
    valid_dataset_path: str = "/datasets01/imagenet_full_size/061417/val"
    # Number of DataLoader workers per device
    workers_per_device: int = 10
    # Batch size per device
    batch_size_per_device: int = 512
    # Autoaugment mode (ra, ta_wide, augmix, or a transforms.autoaugment.AutoAugmentPolicy)
    auto_augment: Optional[str] = None
    # Image resize dimension
    resize_dim: int = 256
    # Image crop dimension
    crop_dim: int = 224
    # Image interpolation mode
    interpolation: str = "bicubic"
    # Horizontal flip probability
    hflip_prob: float = 0.5
    # Random erase parameter
    random_erase: float = 0.0
    # Mixup alpha parameter
    mixup_alpha: float = 0.0
    # CutMix alpha parameter
    cutmix_alpha: float = 0.0
    # Whether to use repeated augmentation
    repeated_aug: bool = False
    # Number of repetitions for repeated augmentation
    repeated_aug_reps: int = 3
    # Image shuffle buffer size for WebDataset data loading
    webdataset_shuffle_buffer: int = 10000


@dataclass
class ImageDatasetConstants:
    dataset_size: int
    num_classes: int
    channel_means: Tuple[float]
    channel_stds: Tuple[float]


@dataclass
class ImageNetConstants(ImageDatasetConstants):
    dataset_size: int = 1281167
    num_classes: int = 1000
    channel_means: Tuple[float] = (0.485, 0.456, 0.406)
    channel_stds: Tuple[float] = (0.229, 0.224, 0.225)


def get_dataset_constants(dataset: str):
    constants_dict = {
        "imagenet": ImageNetConstants,
    }

    if dataset.lower() not in constants_dict:
        raise ValueError(f"Invalid dataset name: {dataset}")
    return constants_dict[dataset.lower()]()


def get_train_transforms(config: DataConfig):
    constants = get_dataset_constants(config.dataset)
    interpolation = InterpolationMode(config.interpolation)
    trans = [transforms.RandomResizedCrop(config.crop_dim, interpolation=interpolation)]
    if config.hflip_prob > 0:
        trans.append(transforms.RandomHorizontalFlip(config.hflip_prob))
    if config.auto_augment is not None:
        if config.auto_augment == "ra":
            trans.append(transforms.autoaugment.RandAugment(interpolation=interpolation))
        elif config.auto_augment == "ta_wide":
            trans.append(transforms.autoaugment.TrivialAugmentWide(interpolation=interpolation))
        elif config.auto_augment == "augmix":
            trans.append(transforms.autoaugment.AugMix(interpolation=interpolation))
        else:
            aa_policy = transforms.autoaugment.AutoAugmentPolicy(config.auto_augment)
            trans.append(
                transforms.autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation)
            )
    trans.extend(
        [
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=constants.channel_means, std=constants.channel_stds),
        ]
    )

    if config.random_erase > 0:
        trans.append(transforms.RandomErasing(p=config.random_erase))
    return transforms.Compose(trans)


def get_train_loader(config: DataConfig, distributed: bool = False):
    constants = get_dataset_constants(config.dataset)
    dataset_size = constants.dataset_size
    train_tf = get_train_transforms(config)

    # assume that we're given a list of webdataset shards if the dataset path is expandable
    use_webdataset = len(list(braceexpand(config.train_dataset_path))) > 1
    if config.repeated_aug:
        if use_webdataset:
            raise RuntimeError("RA sampling is incompatible with webdataset dataloading")
        if not distributed:
            raise RuntimeError("RA sampling requires distributed dataloading")

    if use_webdataset:
        shardlist = wds.PytorchShardList(config.train_dataset_path, epoch_shuffle=True)
        dataset = (
            wds.WebDataset(shardlist)
            .shuffle(config.webdataset_shuffle_buffer)
            .decode("pil")
            .to_tuple("ppm;jpg;jpeg;png", "cls")
            .map_tuple(train_tf, wds.iterators.identity)
            .batched(config.batch_size_per_device, partial=False)
        )
    else:
        dataset = torchvision.datasets.ImageFolder(config.train_dataset_path, train_tf)

    collate_fn = None
    mixup_transforms = []
    if config.mixup_alpha > 0.0:
        mixup_transforms.append(RandomMixup(constants.num_classes, p=1.0, alpha=config.mixup_alpha))
    if config.cutmix_alpha > 0.0:
        mixup_transforms.append(
            RandomCutmix(constants.num_classes, p=1.0, alpha=config.cutmix_alpha)
        )
    if mixup_transforms:
        mixupcutmix = transforms.RandomChoice(mixup_transforms)
        if use_webdataset:
            collate_fn = lambda batch: mixupcutmix(*default_convert(batch))
        else:
            collate_fn = lambda batch: mixupcutmix(*default_collate(batch))

    if use_webdataset:
        train_sampler = None
        train_loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,  # handled by dataset
            num_workers=config.workers_per_device,
            pin_memory=True,
            collate_fn=collate_fn,
        ).ddp_equalize(dataset_size // config.batch_size_per_device)
        if not hasattr(type(train_loader), "__len__"):
            setattr(type(train_loader), "__len__", lambda x: x.length)
    else:
        if distributed:
            if config.repeated_aug:
                train_sampler = RASampler(
                    dataset, shuffle=True, repetitions=config.repeated_aug_reps
                )
            else:
                train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size_per_device,
            sampler=train_sampler,
            num_workers=config.workers_per_device,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collate_fn,
        )

    return train_loader, constants


def get_validation_loader(config: DataConfig):
    constants = get_dataset_constants(config.dataset)
    interpolation = InterpolationMode(config.interpolation)
    eval_tf = transforms.Compose(
        [
            transforms.Resize(config.resize_dim, interpolation=interpolation),
            transforms.CenterCrop(config.crop_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=constants.channel_means, std=constants.channel_stds),
        ]
    )

    dataset = torchvision.datasets.ImageFolder(config.valid_dataset_path, eval_tf)
    valid_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size_per_device,
        shuffle=False,
        num_workers=config.workers_per_device,
        drop_last=False,
        pin_memory=True,
    )

    return valid_loader, constants


class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU).
    Heavily based on 'torch.utils.data.DistributedSampler'.
    This is copied from the DeiT Repo:
    https://github.com/facebookresearch/deit/blob/main/samplers.py
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, repetitions=3):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * float(repetitions) / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(
            math.floor(len(self.dataset) // 256 * 256 / self.num_replicas)
        )
        self.shuffle = shuffle
        self.seed = seed
        self.repetitions = repetitions

    def __iter__(self):
        if self.shuffle:
            # Deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(self.repetitions)]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[: self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
