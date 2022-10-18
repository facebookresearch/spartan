# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence, Tuple, Union
import itertools
import logging
import numpy as np
import sys
from torch import Tensor


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s %(process)d %(name)s:%(lineno)d %(levelname)s %(message)s"
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def normalize_block_dims(
    block_dims: Union[int, Sequence[int]], tensor: Tensor, strict: bool = False
) -> Tuple:
    """Validates that tensor dimensions are divisible by the block dimensions.
    Replaces block dimensions set to -1 with the size of the corresponding tensor dimension."""
    if type(block_dims) is int:
        block_dims = [block_dims] * tensor.ndim

    if len(block_dims) != tensor.ndim:
        raise ValueError("Number of block dimensions must equal the number of input dimensions")

    block_dims = tuple(
        map(lambda t: tensor.shape[t[0]] if (t[1] == -1) else t[1], enumerate(block_dims))
    )

    if np.any([k <= 0 for k in block_dims]):
        raise ValueError("Block dimensions must be positive or -1")

    if np.any([d % k != 0 for d, k in zip(tensor.shape, block_dims)]):
        if strict:
            raise ValueError("Tensor dimensions must be divisible by block dimensions")
        else:
            print(
                f"Falling back to unstructured sparsity because tensor dimensions {tensor.shape} "
                f"are not divisible by block dimensions {block_dims}"
            )
            block_dims = tuple([1] * len(block_dims))

    return block_dims


def tensor_to_blocks(input: Tensor, block_dims: Union[int, Sequence[int]]) -> Tensor:
    """Converts the input tensor to a collection of blocks with the given block dimensions."""
    outer_dims = [d // k for d, k in zip(input.shape, block_dims)]
    dims = itertools.chain.from_iterable(zip(outer_dims, block_dims))
    order = itertools.chain(range(0, 2 * input.ndim, 2), range(1, 2 * input.ndim + 1, 2))
    blocks = input.view(*dims).permute(*order)
    return blocks


def blocks_to_tensor(blocks: Tensor) -> Tensor:
    """Converts a collection of blocks to a flattened tensor. Inverse of `tensor_to_blocks()`."""
    n = blocks.ndim // 2
    order = itertools.chain.from_iterable(zip(range(n), range(n, 2 * n)))
    dims = [blocks.shape[i] * blocks.shape[i + n] for i in range(n)]
    output = blocks.permute(*order).view(*dims)
    return output


def compute_block_values(
    values: Tensor, block_dims: Union[int, Sequence[int]], reduction: str = "sum"
) -> Tensor:
    """Computes a reduction over each block of a tensor."""
    block_dims = normalize_block_dims(block_dims, values)
    if block_dims == tuple(1 for _ in range(values.ndim)):
        return values
    blocks = tensor_to_blocks(values, block_dims)
    blocks = blocks.reshape(*blocks.shape[: values.ndim], -1)
    if reduction == "sum":
        block_values = blocks.sum(-1)
    elif reduction == "mean":
        block_values = blocks.mean(-1)
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")
    return block_values


def mask_blocks(tensor: Tensor, block_mask: Tensor) -> Tensor:
    """Masks the tensor with the given block mask values."""
    block_dims = [t // b for t, b in zip(tensor.shape, block_mask.shape)]
    blocks = tensor_to_blocks(tensor, block_dims)
    remaining_dims = [1] * tensor.ndim
    masked_blocks = blocks * block_mask.view(*blocks.shape[: tensor.ndim], *remaining_dims)
    masked_mat = blocks_to_tensor(masked_blocks)
    return masked_mat


def mask_tensor(tensor: Tensor, mask: Tensor) -> Tensor:
    """Masks the tensor with the given mask. The mask tensor can either be a
    regular mask of the same shape as `tensor` or a block mask."""
    if tensor.shape != mask.shape:
        return mask_blocks(tensor, mask)
    return tensor * mask
