# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Iterable, Mapping, Optional, Tuple, Type
import copy
import torch.nn as nn
from .modules import MaskedModule, MaskedLinear, MaskedConv2d, MaskedMultiheadAttention


default_dense_conversion_fns = {
    nn.Linear: MaskedLinear.from_dense,
    nn.Conv2d: MaskedConv2d.from_dense,
    nn.modules.linear.NonDynamicallyQuantizableLinear: MaskedLinear.from_dense,
    nn.MultiheadAttention: MaskedMultiheadAttention.from_dense,
}


default_masked_conversion_fns = {
    MaskedLinear: MaskedLinear.to_dense,
    MaskedConv2d: MaskedConv2d.to_dense,
    MaskedMultiheadAttention: MaskedMultiheadAttention.to_dense,
}


def convert_modules(
    root_module: nn.Module,
    conversion_fns: Mapping[Type, Callable[..., nn.Module]],
    module_prefixes: Iterable[str] = ("",),
    inplace: bool = True,
    **kwargs,
) -> nn.Module:
    r"""Recursively converts the modules under `root_module` using the
    conversion functions defined by `conversion_fns`. Module types not
    contained as a key in `conversion_fns` are left unmodified.

    Args:
        root_module: The root of the module tree to convert.
        conversion_fns: A Type -> Callable map defining the module types to
            convert and the functions to be used to perform the conversion.
        module_prefixes: Prefixes of module names to convert.
        inplace: If ``True``, modify `root_module` in place.
    """
    if not inplace:
        root_module = copy.deepcopy(root_module)
    return _convert_modules(root_module, conversion_fns, module_prefixes, **kwargs)


def _convert_modules(
    root_module: nn.Module,
    conversion_fns: Mapping[Type, Callable[..., nn.Module]],
    module_prefixes: Iterable[str],
    prefix: str = "",
    **kwargs,
) -> nn.Module:
    if isinstance(root_module, (nn.Sequential, nn.ModuleList)):
        for i in range(len(root_module)):
            name = prefix + ("." if prefix else "") + str(i)
            root_module[i] = _convert_module(
                root_module[i], name, conversion_fns, module_prefixes, **kwargs
            )
    elif isinstance(root_module, nn.ModuleDict):
        for k in list(root_module.keys()):
            name = prefix + ("." if prefix else "") + str(k)
            root_module[k] = _convert_module(
                root_module[k], name, conversion_fns, module_prefixes, **kwargs
            )
    else:
        for attr_str in dir(root_module):
            target_attr = getattr(root_module, attr_str)
            name = prefix + ("." if prefix else "") + attr_str
            if isinstance(target_attr, nn.Module):
                m_new = _convert_module(
                    target_attr, name, conversion_fns, module_prefixes, **kwargs
                )
                setattr(root_module, attr_str, m_new)

    for name, c in root_module.named_children():
        _convert_modules(
            c,
            conversion_fns,
            module_prefixes,
            prefix + ("." if prefix else "") + name,
            **kwargs,
        )

    return root_module


def _convert_module(
    module: nn.Module,
    name: str,
    conversion_fns: Mapping[Type, Callable[..., nn.Module]],
    module_prefixes: Iterable[str],
    **kwargs,
) -> nn.Module:
    if not any(map(name.startswith, module_prefixes)):
        return module
    mtype = type(module)
    if mtype not in conversion_fns:
        return module
    new_module = conversion_fns[mtype](module, **kwargs)
    return new_module


def convert_to_masked_modules(
    root_module: nn.Module,
    conversion_fns: Mapping[Type, Callable[..., MaskedModule]] = default_dense_conversion_fns,
    module_prefixes: Iterable[str] = ("",),
    inplace: bool = True,
    block_dims: Optional[Tuple[int]] = None,
    mask_mode: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    if block_dims is not None:
        kwargs["block_dims"] = block_dims
    if mask_mode is not None:
        kwargs["mask_mode"] = mask_mode
    return convert_modules(root_module, conversion_fns, module_prefixes, inplace, **kwargs)


def convert_from_masked_modules(
    root_module: nn.Module,
    conversion_fns: Mapping[Type, Callable[..., nn.Module]] = default_masked_conversion_fns,
    module_prefixes: Iterable[str] = ("",),
    inplace: bool = True,
    **kwargs,
) -> nn.Module:
    return convert_modules(root_module, conversion_fns, module_prefixes, inplace, **kwargs)
