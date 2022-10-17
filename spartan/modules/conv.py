# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Sequence
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from .masked_module import MaskedModule, DualAveragingMask
from ..utils import mask_tensor, normalize_block_dims, compute_block_values


class MaskedConv2d(nn.Conv2d, MaskedModule):
    """A Conv2d module that maintains a `mask` tensor alongside the `weight`
    parameter for sparse training.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        block_dims: Sequence[int] = (1, 1, 1, 1),
        mask_mode: str = "spartan",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

        # keep track of the output shape for FLOP counting
        self.output_shape = None

        if len(block_dims) == 2:
            block_dims = (*block_dims, 1, 1)
        self.block_dims = normalize_block_dims(block_dims, self.weight)
        if np.any([w % b != 0 for w, b in zip(self.weight.shape, self.block_dims)]):
            raise ValueError("Weight matrix dimensions must be divisible by block dimensions")

        self.set_mask_mode(mask_mode)
        self.mask_dims = tuple([w // b for w, b in zip(self.weight.shape, self.block_dims)])
        self.register_buffer("mask", torch.ones(self.mask_dims, device=device, dtype=dtype))
        self.register_buffer("hard_mask_threshold", torch.tensor(0.0, device=device, dtype=dtype))

    @classmethod
    def from_dense(
        cls,
        module: nn.Conv2d,
        block_dims: Optional[Sequence[int]] = None,
        mask_mode: Optional[str] = None,
    ) -> "MaskedConv2d":
        kwargs = {
            "in_channels": module.in_channels,
            "out_channels": module.out_channels,
            "kernel_size": module.kernel_size,
            "stride": module.stride,
            "padding": module.padding,
            "dilation": module.dilation,
            "groups": module.groups,
            "bias": module.bias is not None,
            "padding_mode": module.padding_mode,
            "device": module.weight.device,
            "dtype": module.weight.dtype,
        }
        if block_dims is not None:
            kwargs["block_dims"] = block_dims
        if mask_mode is not None:
            kwargs["mask_mode"] = mask_mode
        m_new = cls(**kwargs)
        m_new.weight.data = module.weight
        if m_new.bias is not None:
            m_new.bias.data = module.bias
        return m_new

    def to_dense(self) -> nn.Conv2d:
        module = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias is not None,
            self.padding_mode,
            self.weight.device,
            self.weight.dtype,
        )
        module.weight.data = self.masked_weight
        if self.bias is not None:
            module.bias.data = self.bias

        # used for flop computations
        module.output_shape = self.output_shape
        return module

    def mask_values(self, value_fn: Callable[[nn.Module, Tensor], Tensor]) -> Tensor:
        vals = compute_block_values(value_fn(self, self.weight), self.block_dims)
        return vals

    def mask_costs(self, cost_fn: Callable[[nn.Module, Tensor], Tensor]) -> Tensor:
        cost = torch.tensor(
            np.prod(self.block_dims), device=self.weight.device, dtype=self.weight.dtype
        )
        cost = cost_fn(self, cost)
        costs = cost.expand(self.mask_dims)
        return costs

    def num_masked_parameters(self) -> int:
        return self.weight.numel()

    @property
    def masked_weight(self) -> Tensor:
        if self.mask_mode == "dual_averaging":
            return DualAveragingMask.apply(self.weight, self.mask)
        elif self.mask_mode == "spartan":
            return DualAveragingMask.apply(
                mask_tensor(self.weight, self.mask), self.mask >= self.hard_mask_threshold
            )

        # mask_mode == "standard"
        return mask_tensor(self.weight, self.mask)

    def forward(self, input: Tensor):
        output = self._conv_forward(input, self.masked_weight, self.bias)
        if self.output_shape is None:
            self.output_shape = output.shape
        return output
