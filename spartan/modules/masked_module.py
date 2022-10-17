# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Sequence
import torch
from torch import Tensor
from torch.cuda.amp import custom_fwd, custom_bwd
from torch.autograd.function import once_differentiable
import torch.nn as nn

from ..utils import mask_tensor


_valid_mask_modes = (
    # sparse forward, dense backward
    "dual_averaging",
    # sparse forward, soft top-k backward
    "spartan",
    # sparse forward, sparse backward
    "standard",
)


class DualAveragingMask(torch.autograd.Function):
    """A custom masking operator used for implementing dual averaging updates,
    i.e., 'straight-through' updates where the mask is applied in the forward
    pass, but is treated as an identity function in the backward pass.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, params, mask):
        return mask_tensor(params, mask)

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output, None


class MaskedModule(nn.Module):
    """Base class for masked modules used for sparse training."""

    def __init__(self):
        super().__init__()
        # A mask tensor with entries in [0, 1] that is applied to the module parameters
        self.register_buffer("mask", None)
        # A scalar threshold used to convert a soft mask to a hard mask
        self.register_buffer("hard_mask_threshold", None)
        self.mask_mode = "spartan"

    @classmethod
    def from_dense(
        cls,
        module: nn.Module,
        block_dims: Optional[Sequence[int]] = None,
        mask_mode: Optional[str] = None,
    ) -> "MaskedModule":
        """Instantiate a MaskedModule, copying parameters from the base module.
        Optionally specify a sequence of block dimensions for block sparse masking."""
        raise NotImplementedError

    def to_dense(self) -> nn.Module:
        """Returns an instance of the base module with sparse parameters."""
        raise NotImplementedError

    def set_mask_mode(self, mode: str) -> None:
        if mode not in _valid_mask_modes:
            raise ValueError(f"Invalid mask mode: {mode}")
        self.mask_mode = mode
        return self

    def mask_values(self, value_fn: Callable[[nn.Module, Tensor], Tensor]) -> Tensor:
        """Returns a tensor containing the value of each mask entry.
        Entries with higher values are preferentially retained."""
        raise NotImplementedError

    def mask_costs(self, cost_fn: Callable[[nn.Module, Tensor], Tensor]) -> Tensor:
        """Returns a tensor containing the cost of each mask entry.
        Entries with higher costs are preferentially pruned."""
        raise NotImplementedError

    def num_masked_parameters(self) -> int:
        raise NotImplementedError
