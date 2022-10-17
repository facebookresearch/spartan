# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from collections import namedtuple
from pydoc import locate
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

from . import convert
from .modules import MaskedModule, MaskedConv2d
from .utils import get_logger


logger = get_logger(__name__)


@dataclass
class SparsifierConfig:
    """Configuration for the Sparsifier class."""

    # Sparse mask mode: {"standard", "dual_averaging", "spartan"}
    sparsity_mode: str = "spartan"
    # Sparsity distribution: {"global", "uniform", "erk"}
    sparsity_dist: str = "global"

    # Warmup period as a fraction of total iterations
    warmup_frac: float = 0.2
    # Finetune period as a fraction of total iterations
    finetune_frac: float = 0.2
    # Annealing method for sparsity and beta values: {"linear", "cosine"}
    annealing_mode: str = "linear"
    # Cost model: {"sparsity", "flops"}
    cost_model: str = "sparsity"
    # Initial parameter cost fraction
    cost_frac_init: float = 1.0
    # Final parameter cost fraction
    cost_frac_target: float = 0.1
    # Increase sparsity of masked layers so that overall model sparsity equals target
    adjust_cost_frac: bool = True
    # Block structured sparsity dimensions
    block_dims: Tuple[int, int] = (1, 1)

    # Initial Sinkhorn entropic regularization parameter
    sinkhorn_beta_init: float = 1.0
    # Final Sinkhorn entropic regularization parameter
    sinkhorn_beta_target: float = 20.0
    # Maximum number of Sinkhorn iterations
    sinkhorn_max_iters: int = 100
    # Stopping tolerance for relative change in objective value
    sinkhorn_tol: float = 0.01

    # Types of layers to sparsify
    layer_types: List[str] = field(
        default_factory=lambda: [
            "torch.nn.Linear",
            "torch.nn.modules.linear.NonDynamicallyQuantizableLinear",
            "torch.nn.Conv2d",
            "torch.nn.MultiheadAttention",
        ]
    )
    # Prefixes of module names to sparsify
    module_prefixes: List[str] = field(default_factory=lambda: [""])


class Sparsifier(object):
    r"""Performs iterative parameter sparsification using the Spartan
    algorithm.

    Reference:
        Tai et al, 2022. Spartan: Differentiable Sparsity via Regularized
        Transportation. https://arxiv.org/abs/2205.14107

    Args:
        model: The model to be sparsified. The model is modified inplace to
            replace the layers to be sparsified with MaskedModule instances
        config: Sparsifier configuration
        total_iters: Total number of training iterations
        custom_dense_conversion_fns: Custom dense -> MaskedModule conversion functions
        custom_masked_conversion_fns: Custom MaskedModule -> dense conversion functions
        verbose: Whether to verbosely log progress

    Usage:
        > sparsifier = Sparsifier(model, config, total_iters)
        > for _ in range(train_iters):
        >     loss = ...
        >     loss.backward()
        >     optimizer.step()
        >     sparsifier.step()
        > sparsifier.finalize()
    """

    # Return types
    SparsityParams = namedtuple("SparsityParams", ["cost_frac", "sinkhorn_beta"])
    SetMaskResult = namedtuple("SetMaskResult", ["sinkhorn_iters"])
    StepResult = namedtuple(
        "StepResult", ["step_count", "cost_frac", "sinkhorn_iters", "sinkhorn_beta"]
    )

    def __init__(
        self,
        model: nn.Module,
        config: SparsifierConfig,
        total_iters: int,
        custom_dense_conversion_fns: Optional[Mapping[Type, Callable[..., MaskedModule]]] = None,
        custom_masked_conversion_fns: Optional[Mapping[Type, Callable[..., nn.Module]]] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.config = config
        self.total_iters = total_iters
        self.custom_dense_conversion_fns = custom_dense_conversion_fns
        self.custom_masked_conversion_fns = custom_masked_conversion_fns
        self.verbose = verbose
        self._value_fn = lambda module, weight: torch.abs(weight)
        self._cost_fn = _flops_cost_fn if config.cost_model == "flops" else _identity_cost_fn
        self._step_count = 0
        self._frozen = False
        self._finalized = False

        # Convert modules to masked versions
        conversion_fns = {}
        for typename in config.layer_types:
            t = locate(typename)
            conversion_fns[t] = convert.default_dense_conversion_fns[t]
        if self.custom_dense_conversion_fns is not None:
            conversion_fns.update(self.custom_dense_conversion_fns)

        convert.convert_to_masked_modules(
            self.model,
            conversion_fns=conversion_fns,
            module_prefixes=config.module_prefixes,
            block_dims=config.block_dims,
            mask_mode=config.sparsity_mode,
            inplace=True,
        )

        # Register masked modules
        self._named_masked_modules: Dict[str, MaskedModule] = {}
        for k, m in model.named_modules():
            if isinstance(m, MaskedModule):
                self._named_masked_modules[k] = m

        if self.verbose:
            for k, m in self._named_masked_modules.items():
                logger.info(
                    f"Sparsifying module {k} (type: {type(m).__name__}, "
                    f"num masked parameters: {m.num_masked_parameters()})"
                )

        if len(self._named_masked_modules) == 0:
            raise RuntimeError("Model does not contain any modules to be sparsified")

        if config.adjust_cost_frac and config.cost_model == "sparsity":
            adjusted_cost_frac_target = _adjust_sparsity_fraction(
                config.cost_frac_target, self.model
            )
            if adjusted_cost_frac_target <= 0:
                raise RuntimeError(
                    f"Cannot achieve target cost fraction of {config.cost_frac_target} "
                    f"by sparsifying the specified layers "
                    f"(adjusted cost fraction is {adjusted_cost_frac_target:.4f})"
                )
            logger.info(
                f"Adjusted target cost fraction: {adjusted_cost_frac_target:.4f} "
                f"(overall cost target: {config.cost_frac_target})"
            )

    @property
    def warmup_iters(self) -> int:
        return int(self.config.warmup_frac * self.total_iters)

    @property
    def finetune_iters(self) -> int:
        return int(self.config.finetune_frac * self.total_iters)

    @property
    def finetune_start(self) -> int:
        return self.total_iters - self.finetune_iters

    @property
    def sinkhorn_scaled_beta_target(self) -> float:
        """Target Sinkhorn beta parameter scaled by the sqrt of the number of
        block elements. This scaling compensates for the reduction in the
        variance of block values due to averaging over the entries of the
        block.
        """
        scale = np.sqrt(self.config.block_dims[0] * self.config.block_dims[1])
        return scale * self.config.sinkhorn_beta_target

    def sparsity_params(self) -> SparsityParams:
        """Compute the current sparsity parameters according to the provided
        schedule."""
        cost_frac = _evaluate_schedule(
            self.config.annealing_mode,
            self._step_count,
            start=self.config.cost_frac_init,
            end=self.config.cost_frac_target,
            start_iter=0,
            end_iter=self.warmup_iters,
        )

        sinkhorn_beta = _evaluate_schedule(
            self.config.annealing_mode,
            self._step_count,
            start=self.config.sinkhorn_beta_init,
            end=self.sinkhorn_scaled_beta_target,
            start_iter=0,
            end_iter=self.finetune_start,
        )

        return self.SparsityParams(cost_frac, sinkhorn_beta)

    def step(self) -> StepResult:
        """Updates model masks and sparsifier state. Must be called once after
        each call to model.backward() since masks are differentiable and depend
        on the model's parameters.
        """
        if self._finalized:
            raise RuntimeError("Attempted to call step when sparsifier is already finalized")

        self._step_count += 1
        sparsity_params = self.sparsity_params()
        cost_frac = sparsity_params.cost_frac
        sinkhorn_beta = sparsity_params.sinkhorn_beta

        # Adjust target sparsity fraction to compensate for unmasked modules
        if self.config.cost_model == "sparsity" and self.config.adjust_cost_frac:
            cost_frac = _adjust_sparsity_fraction(cost_frac, self.model)
            # adjust_flop_fraction is not implemented

        # Set masks
        sinkhorn_iters = 0
        if self._step_count <= self.finetune_start:
            if self.config.sparsity_dist == "global":
                set_mask_result = self._set_masks(cost_frac, sinkhorn_beta)
                sinkhorn_iters = set_mask_result.sinkhorn_iters
            elif self.config.sparsity_dist == "uniform":
                for k in self._named_masked_modules.keys():
                    set_mask_result = self._set_masks(cost_frac, sinkhorn_beta, [k])
                    sinkhorn_iters = max(sinkhorn_iters, set_mask_result.sinkhorn_iters)
            elif self.config.sparsity_dist == "erk":
                module_cost_fracs = _erdos_renyi_sparsity(
                    self._named_masked_modules.values(), cost_frac
                )
                for k, cost_frac in zip(self._named_masked_modules.keys(), module_cost_fracs):
                    set_mask_result = self._set_masks(cost_frac, sinkhorn_beta, [k])
                    sinkhorn_iters = max(sinkhorn_iters, set_mask_result.sinkhorn_iters)
            else:
                raise RuntimeError(f"Invalid sparsity distribution: {self.config.sparsity_dist}")

        # Freeze mask for fine tuning at the end of training. We freeze after
        # updating the mask one last time on finetune_start. After that, check
        # that the masker is frozen in each iteration in case we loaded from a
        # checkpoint.
        if self._step_count >= self.finetune_start:
            if not self._frozen:
                self._freeze_masks()

        return self.StepResult(
            step_count=self._step_count,
            cost_frac=cost_frac,
            sinkhorn_iters=sinkhorn_iters,
            sinkhorn_beta=sinkhorn_beta,
        )

    def finalize(self) -> None:
        """Finalize the sparsifier at the end of training by converting
        MaskedModules back to standard nn.Modules. step() cannot be called
        after the sparsifier is finalized."""
        self._finalized = True
        conversion_fns = dict(convert.default_masked_conversion_fns)
        if self.custom_masked_conversion_fns is not None:
            conversion_fns.update(self.custom_masked_conversion_fns)
        convert.convert_from_masked_modules(self.model, conversion_fns=conversion_fns)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "_step_count": self._step_count,
            "_frozen": self._frozen,
            "_finalized": self._finalized,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

    def total_cost(self) -> float:
        cost = 0.0
        for m in self._named_masked_modules.values():
            cost += m.mask_costs(self._cost_fn).sum().item()
        return cost

    def named_masked_modules(self):
        yield from self._named_masked_modules.items()

    def masked_modules(self):
        yield from self._named_masked_modules.values()

    def is_frozen(self) -> bool:
        return self._frozen

    def is_finalized(self) -> bool:
        return self._finalized

    def _set_masks(
        self,
        cost_frac: float,
        sinkhorn_beta: Optional[float] = None,
        module_names: Optional[Iterable[str]] = None,
    ) -> SetMaskResult:
        if self._frozen:
            raise RuntimeError("Attempted to call set_masks when masker is already frozen")

        if cost_frac < 0 or cost_frac > 1:
            raise ValueError("Cost fraction must be in [0, 1]")

        module_names = module_names or self._named_masked_modules.keys()
        modules = [self._named_masked_modules[k] for k in module_names]

        if cost_frac == 0:
            for module in modules:
                module.mask = torch.zeros_like(module.mask)
            return self.SetMaskResult(sinkhorn_iters=0)

        if cost_frac == 1:
            for module in modules:
                module.mask = torch.ones_like(module.mask)
            return self.SetMaskResult(sinkhorn_iters=0)

        # Each mask entry has a corresponding value and cost.
        # Entries with a higher value per unit cost are assigned values closer
        # to 1.
        values = [m.mask_values(self._value_fn) for m in modules]
        offsets = np.cumsum([0] + [v.numel() for v in values])
        values_flat = torch.concat([v.view(-1) for v in values])
        costs_flat = torch.concat([m.mask_costs(self._cost_fn).view(-1) for m in modules])
        total_cost = costs_flat.sum()
        budget = cost_frac * total_cost
        unit_values = values_flat / costs_flat

        # Order entries by their unit value and compute the threshold for hard
        # masking
        unit_values_desc, idxs = torch.sort(unit_values, descending=True)
        costs_cum = torch.cumsum(costs_flat[idxs], 0)
        threshold_idx = torch.searchsorted(costs_cum, budget)
        if threshold_idx == len(values_flat):
            hard_threshold = unit_values_desc[-1] - 1
        else:
            hard_threshold = unit_values_desc[threshold_idx]

        if self.config.sparsity_mode == "spartan":
            # Compute soft mask using the Sinkhorn algorithm
            masks_flat, _, _, sinkhorn_iters = SinkhornTopK.apply(
                values_flat,
                costs_flat,
                budget,
                sinkhorn_beta,
                self.config.sinkhorn_max_iters,
                self.config.sinkhorn_tol,
                unit_values,
                hard_threshold,
            )

            # Hard cutoff for forward pass
            hard_mask_threshold = masks_flat[idxs[threshold_idx]].clone().detach()
        else:
            # Compute hard mask
            masks_flat = unit_values > hard_threshold
            hard_mask_threshold = None
            sinkhorn_iters = 0

        # Assign masks to modules
        for i, module in enumerate(modules):
            module.mask = torch.clone(masks_flat[offsets[i] : offsets[i + 1]]).reshape_as(
                module.mask
            )

            if hard_mask_threshold is not None:
                module.hard_mask_threshold = hard_mask_threshold

        return self.SetMaskResult(sinkhorn_iters=sinkhorn_iters)

    def _freeze_masks(self) -> None:
        if self._frozen:
            return
        self._frozen = True
        for module in self.masked_modules():
            module.mask = module.mask.detach()
            if module.mask_mode == "spartan":
                module.mask *= module.mask >= module.hard_mask_threshold
            module.set_mask_mode("standard")


class SinkhornTopK(torch.autograd.Function):
    """Compute a soft top-k mask using the Sinkhorn algorithm."""

    @staticmethod
    @custom_fwd
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        values: Tensor,
        costs: Tensor,
        budget: Tensor,
        beta: float,
        max_iters: int,
        tol: float,
        unit_values: Optional[Tensor] = None,
        hard_threshold: Optional[Union[float, Tensor]] = None,
    ):
        """Soft top-k forward pass.

        Args:
            values: Value tensor of shape [*, d]
            costs: Cost tensor of shape [*, d]
            budget: Cost budget
            beta: Sinkhorn beta parameter
            max_iters: Maximum number of Sinkhorn iterations
            tol: Stopping tolerance for relative change in objective value
            unit_values: (values / costs) Tensor, if already precomputed
            hard_threshold: Hard top-k threshold, if already precomputed
        """
        logk = torch.log(budget)
        logub = torch.log(costs)
        if unit_values is None:
            unit_values = values / costs

        # shift for numerical stability
        if hard_threshold is None:
            mean_unit_value = unit_values.mean(-1, keepdim=True)
            z = beta * (unit_values - mean_unit_value)
            mu = logk - torch.logsumexp(z, dim=-1, keepdim=True)
        else:
            z = beta * (unit_values - hard_threshold)
            mu = 0
        nu = logub - F.softplus(z + mu)
        output = torch.exp(z + mu + nu - logub)
        prev_obj = torch.sum(output * values)
        for _step in range(max_iters):
            mu = logk - torch.logsumexp(z + nu, dim=-1, keepdim=True)
            nu = logub - F.softplus(z + mu)
            output = torch.exp(z + mu + nu - logub)
            obj = torch.sum(output * values)
            if torch.abs(obj - prev_obj) <= tol * torch.abs(prev_obj):
                break
            prev_obj = obj

        ctx.mark_non_differentiable(mu, nu)  # gradients wrt dual variables currently unsupported
        ctx.save_for_backward(output, costs, budget)
        ctx.beta = beta
        return output, mu, nu, _step

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Tensor,
        grad_mu: Tensor,
        grad_nu: Tensor,
        grad_step: Tensor,
    ):
        """Gradient derived from implicit differentiation. See paper for detailed derivation."""
        output, costs, budget = ctx.saved_tensors
        c1 = torch.sum(grad_output * output * (1 - output), dim=-1, keepdim=True)
        c2 = torch.sum(output * output * costs, dim=-1, keepdim=True)
        c = c1 / (budget - c2)
        grad_values = ctx.beta * output * (1 - output) * (grad_output / costs - c)
        return grad_values, None, None, None, None, None, None, None


def _evaluate_schedule(
    mode: str,
    cur_iter: int,
    start: float,
    end: float,
    start_iter: int,
    end_iter: int,
) -> float:
    if mode == "linear":
        if cur_iter <= start_iter:
            ratio = 0.0
        elif cur_iter >= end_iter:
            ratio = 1.0
        else:
            ratio = (cur_iter - start_iter) / (end_iter - start_iter)
        return start + ratio * (end - start)
    elif mode == "cosine":
        if cur_iter <= start_iter:
            ratio = 0.0
        elif cur_iter >= end_iter:
            ratio = 1.0
        else:
            ratio = 0.5 * (1 - np.cos(np.pi * (cur_iter - start_iter) / (end_iter - start_iter)))
        return start + ratio * (end - start)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented")


def _identity_cost_fn(module: nn.Module, cost: torch.Tensor):
    return cost


def _flops_cost_fn(module: nn.Module, cost: torch.Tensor):
    if isinstance(module, MaskedConv2d):
        spatial_size = 1
        if module.output_shape is not None:
            spatial_size = module.output_shape[-1] * module.output_shape[-2]
        return spatial_size * cost
    return cost


def _adjust_sparsity_fraction(target: float, model: nn.Module) -> float:
    total_params = sum([p.numel() for p in model.parameters()])
    num_masked_params = 0
    for m in model.modules():
        if isinstance(m, MaskedModule):
            num_masked_params += m.num_masked_parameters()
    num_zeros = (1 - target) * total_params
    adjusted_frac = max(0, 1 - num_zeros / num_masked_params)
    return adjusted_frac


def _erdos_renyi_sparsity(modules: Iterable[nn.Module], nnz_frac: float) -> Iterable[float]:
    """Compute target sparsities for each module according to the Erdos-Renyi-Kernel sparsity formula
    from Evci et al., Rigging the Lottery: Making All Tickets Winners (2020)."""
    coefs = []
    num_params = []
    for m in modules:
        s = m.weight.shape
        num_params.append(np.prod(s))
        coefs.append(np.sum(s) / num_params[-1])

    num_params = np.array(num_params)
    coefs = np.array(coefs)
    target = nnz_frac * num_params.sum()

    idxs = np.argsort(coefs)
    coefs_ord = coefs[idxs]
    num_params_ord = num_params[idxs]
    z = np.cumsum(coefs_ord * num_params_ord)

    n = len(coefs)
    a = target / z[n - 1]
    x = a * coefs_ord
    while x[n - 1] > 1:
        x[x > 1] = 1
        n_prev, n = n, np.searchsorted(x, 1)
        if n == 0:
            break
        target -= num_params_ord[n:n_prev].sum()
        a = target / z[n - 1]
        x[:n] = a * coefs_ord[:n]
    out = np.zeros_like(x)
    out[idxs] = x
    return out
