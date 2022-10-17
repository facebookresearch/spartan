# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .masked_module import MaskedModule, DualAveragingMask
from .linear import MaskedLinear
from ..utils import mask_tensor, normalize_block_dims, compute_block_values


class MaskedMultiheadAttention(nn.MultiheadAttention, MaskedModule):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        block_dims: Sequence[int] = (1, 1),
        mask_mode: str = "spartan",
        device=None,
        dtype=None,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            kdim,
            vdim,
            batch_first,
            device,
            dtype,
        )
        self.out_proj = MaskedLinear(
            self.embed_dim, self.embed_dim, bias, block_dims, mask_mode, device, dtype
        )
        if self._qkv_same_embed_dim:
            self.in_proj_block_dims = normalize_block_dims(block_dims, self.in_proj_weight)
            self.q_proj_block_dims = None
            self.k_proj_block_dims = None
            self.v_proj_block_dims = None

            if np.any(
                [w % b != 0 for w, b in zip(self.in_proj_weight.shape, self.in_proj_block_dims)]
            ):
                raise ValueError("in_proj_weight dimensions must be divisible by block dimensions")

            self.in_proj_mask_dims = tuple(
                [w // b for w, b in zip(self.in_proj_weight.shape, self.in_proj_block_dims)]
            )
            self.q_proj_mask_dims = None
            self.k_proj_mask_dims = None
            self.v_proj_mask_dims = None
            self.mask_dims = self.in_proj_mask_dims
        else:
            self.q_proj_block_dims = normalize_block_dims(block_dims, self.q_proj_weight)
            self.k_proj_block_dims = normalize_block_dims(block_dims, self.k_proj_weight)
            self.v_proj_block_dims = normalize_block_dims(block_dims, self.v_proj_weight)
            self.in_proj_block_dims = None

            if np.any(
                [w % b != 0 for w, b in zip(self.q_proj_weight.shape, self.q_proj_block_dims)]
            ):
                raise ValueError("q_proj_weight dimensions must be divisible by block dimensions")
            if np.any(
                [w % b != 0 for w, b in zip(self.k_proj_weight.shape, self.k_proj_block_dims)]
            ):
                raise ValueError("k_proj_weight dimensions must be divisible by block dimensions")
            if np.any(
                [w % b != 0 for w, b in zip(self.v_proj_weight.shape, self.v_proj_block_dims)]
            ):
                raise ValueError("v_proj_weight dimensions must be divisible by block dimensions")

            self.q_proj_mask_dims = tuple(
                [w // b for w, b in zip(self.q_proj_weight.shape, self.q_proj_block_dims)]
            )
            self.k_proj_mask_dims = tuple(
                [w // b for w, b in zip(self.k_proj_weight.shape, self.k_proj_block_dims)]
            )
            self.v_proj_mask_dims = tuple(
                [w // b for w, b in zip(self.v_proj_weight.shape, self.v_proj_block_dims)]
            )
            self.in_proj_mask_dims = None
            self.mask_dims = (
                np.prod(self.q_proj_mask_dims)
                + np.prod(self.k_proj_mask_dims)
                + np.prod(self.v_proj_mask_dims),
            )

        # keep track of the output shape for FLOP counting
        self.output_shape = None
        self.set_mask_mode(mask_mode)
        self.register_buffer(
            "mask",
            torch.ones(
                self.mask_dims, device=self.out_proj.weight.device, dtype=self.out_proj.weight.dtype
            ),
        )
        self.register_buffer("hard_mask_threshold", torch.tensor(0.0, device=device, dtype=dtype))

    @classmethod
    def from_dense(
        cls,
        module: nn.MultiheadAttention,
        block_dims: Optional[Sequence[int]] = None,
        mask_mode: Optional[str] = None,
    ) -> "MaskedMultiheadAttention":
        kwargs = {
            "embed_dim": module.embed_dim,
            "num_heads": module.num_heads,
            "dropout": module.dropout,
            "bias": module.in_proj_bias is not None,
            "add_bias_kv": module.bias_k is not None,
            "add_zero_attn": module.add_zero_attn,
            "kdim": module.kdim,
            "vdim": module.vdim,
            "batch_first": module.batch_first,
            "device": module.out_proj.weight.device,
            "dtype": module.out_proj.weight.dtype,
        }
        if block_dims is not None:
            kwargs["block_dims"] = block_dims
        if mask_mode is not None:
            kwargs["mask_mode"] = mask_mode
        m_new = cls(**kwargs)
        if m_new.bias_k is not None:
            m_new.bias_k.data = module.bias_k
            m_new.bias_v.data = module.bias_v
        m_new.out_proj.weight.data = module.out_proj.weight
        if m_new.out_proj.bias is not None:
            m_new.out_proj.bias.data = module.out_proj.bias

        if m_new._qkv_same_embed_dim:
            m_new.in_proj_weight.data = module.in_proj_weight
        else:
            m_new.q_proj_weight.data = module.q_proj_weight
            m_new.k_proj_weight.data = module.k_proj_weight
            m_new.v_proj_weight.data = module.v_proj_weight

        if m_new.in_proj_bias is not None:
            m_new.in_proj_bias.data = module.in_proj_bias

        if not module.training:
            m_new.eval()
        return m_new

    def to_dense(self) -> nn.MultiheadAttention:
        module = nn.MultiheadAttention(
            self.embed_dim,
            self.num_heads,
            self.dropout,
            self.in_proj_bias is not None,
            self.bias_k is not None,
            self.add_zero_attn,
            self.kdim,
            self.vdim,
            self.batch_first,
            self.out_proj.weight.device,
            self.out_proj.weight.dtype,
        )
        if module.bias_k is not None:
            module.bias_k.data = self.bias_k
            module.bias_v.data = self.bias_v
        module.out_proj = self.out_proj.to_dense()
        if module._qkv_same_embed_dim:
            module.in_proj_weight.data = self.masked_in_proj_weight
        else:
            module.q_proj_weight.data = self.masked_q_proj_weight
            module.k_proj_weight.data = self.masked_k_proj_weight
            module.v_proj_weight.data = self.masked_v_proj_weight

        if module.in_proj_bias is not None:
            module.in_proj_bias.data = self.in_proj_bias

        if not self.training:
            module.eval()

        # used for flop computations
        module.output_shape = self.output_shape
        return module

    def mask_values(self, value_fn: Callable[[nn.Module, Tensor], Tensor]) -> Tensor:
        if self._qkv_same_embed_dim:
            vals = compute_block_values(
                value_fn(self, self.in_proj_weight), self.in_proj_block_dims
            )
        else:
            q_vals = compute_block_values(
                value_fn(self, self.q_proj_weight), self.q_proj_block_dims
            )
            k_vals = compute_block_values(
                value_fn(self, self.k_proj_weight), self.k_proj_block_dims
            )
            v_vals = compute_block_values(
                value_fn(self, self.v_proj_weight), self.v_proj_block_dims
            )
            vals = torch.concat([q_vals.view(-1), k_vals.view(-1), v_vals.view(-1)])
        return vals

    def mask_costs(self, cost_fn: Callable[[nn.Module, Tensor], Tensor]) -> Tensor:
        factory_kwargs = {
            "device": self.out_proj.weight.device,
            "dtype": self.out_proj.weight.dtype,
        }
        if self._qkv_same_embed_dim:
            cost = torch.tensor(np.prod(self.in_proj_block_dims), **factory_kwargs)
            costs = cost_fn(self, cost).expand(self.mask_dims)
        else:
            q_cost = torch.tensor(np.prod(self.q_proj_block_dims), **factory_kwargs)
            q_costs = cost_fn(self, q_cost).expand(self.q_proj_mask_dims)
            k_cost = torch.tensor(np.prod(self.k_proj_block_dims), **factory_kwargs)
            k_costs = cost_fn(self, k_cost).expand(self.k_proj_mask_dims)
            v_cost = torch.tensor(np.prod(self.v_proj_block_dims), **factory_kwargs)
            v_costs = cost_fn(self, v_cost).expand(self.v_proj_mask_dims)
            costs = torch.concat([q_costs.view(-1), k_costs.view(-1), v_costs.view(-1)])
        return costs

    @property
    def in_proj_mask(self) -> Optional[Tensor]:
        if self.in_proj_weight is None:
            return None
        return self.mask

    @property
    def q_proj_mask(self) -> Optional[Tensor]:
        if self.q_proj_weight is None:
            return None
        hi = np.prod(self.q_proj_mask_dims)
        mask = self.mask[:hi].view(self.q_proj_mask_dims)
        return mask

    @property
    def k_proj_mask(self) -> Optional[Tensor]:
        if self.k_proj_weight is None:
            return None
        lo = np.prod(self.q_proj_mask_dims)
        hi = lo + np.prod(self.k_proj_mask_dims)
        mask = self.mask[lo:hi].view(self.k_proj_mask_dims)
        return mask

    @property
    def v_proj_mask(self) -> Optional[Tensor]:
        if self.v_proj_weight is None:
            return None
        lo = -np.prod(self.v_proj_mask_dims)
        mask = self.mask[lo:].view(self.v_proj_mask_dims)
        return mask

    @property
    def masked_in_proj_weight(self) -> Optional[Tensor]:
        if self.in_proj_weight is None:
            return None
        if self.mask_mode == "dual_averaging":
            return DualAveragingMask.apply(self.in_proj_weight, self.mask)
        if self.mask_mode == "spartan":
            return DualAveragingMask.apply(
                mask_tensor(self.in_proj_weight, self.mask), self.mask >= self.hard_mask_threshold
            )

        # mask_mode == "standard"
        return mask_tensor(self.in_proj_weight, self.mask)

    @property
    def masked_q_proj_weight(self) -> Optional[Tensor]:
        if self.q_proj_weight is None:
            return None
        if self.mask_mode == "dual_averaging":
            return DualAveragingMask.apply(self.q_proj_weight, self.q_proj_mask)
        if self.mask_mode == "spartan":
            return DualAveragingMask.apply(
                mask_tensor(self.q_proj_weight, self.q_proj_mask),
                self.q_proj_mask >= self.hard_mask_threshold,
            )
        return mask_tensor(self.q_proj_weight, self.q_proj_mask)

    @property
    def masked_k_proj_weight(self) -> Optional[Tensor]:
        if self.k_proj_weight is None:
            return None
        if self.mask_mode == "dual_averaging":
            return DualAveragingMask.apply(self.k_proj_weight, self.k_proj_mask)
        if self.mask_mode == "spartan":
            return DualAveragingMask.apply(
                mask_tensor(self.k_proj_weight, self.k_proj_mask),
                self.k_proj_mask >= self.hard_mask_threshold,
            )
        return mask_tensor(self.k_proj_weight, self.k_proj_mask)

    @property
    def masked_v_proj_weight(self) -> Optional[Tensor]:
        if self.v_proj_weight is None:
            return None
        if self.mask_mode == "dual_averaging":
            return DualAveragingMask.apply(self.v_proj_weight, self.v_proj_mask)
        if self.mask_mode == "spartan":
            return DualAveragingMask.apply(
                mask_tensor(self.v_proj_weight, self.v_proj_mask),
                self.v_proj_mask >= self.hard_mask_threshold,
            )
        return mask_tensor(self.v_proj_weight, self.v_proj_mask)

    def num_masked_parameters(self) -> int:
        if self._qkv_same_embed_dim:
            return self.in_proj_weight.numel()
        else:
            return (
                self.q_proj_weight.numel() + self.k_proj_weight.numel() + self.v_proj_weight.numel()
            )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        out_proj_weight = (
            self.out_proj.masked_weight
            if hasattr(self.out_proj, "masked_weight")
            else self.out_proj.weight
        )
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.masked_in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                out_proj_weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.masked_q_proj_weight,
                k_proj_weight=self.masked_k_proj_weight,
                v_proj_weight=self.masked_v_proj_weight,
                average_attn_weights=average_attn_weights,
            )
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.masked_in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                out_proj_weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
            )

        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(1, 0)
            self.output_shape = attn_output.shape
        return attn_output, attn_output_weights
