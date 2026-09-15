"""HIP combine for contiguous eight-head TileLang split-KV partials on gfx950."""

from __future__ import annotations

import logging
from collections import Counter
from functools import lru_cache
from typing import TYPE_CHECKING, Callable

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi import Module

logger = logging.getLogger(__name__)
_fallback_counts: Counter[tuple[int, int, int, int, str]] = Counter()


@lru_cache(maxsize=None)
def supported_device(device: torch.device) -> bool:
    """Check the input device, including its index, before compiling HIP code."""
    return (
        torch.version.hip is not None
        and device.type == "cuda"
        and device.index is not None
        and torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        == "gfx950"
    )


@cache_once
def build() -> Module:
    return load_jit(
        "split_kv_combine_hip",
        cuda_files=["attention/split_kv_combine_hip.cuh"],
        cuda_wrappers=[("run", "SplitKVCombineHIP")],
    )


def split_kv_combine_hip(
    partial_o: torch.Tensor, partial_lse: torch.Tensor
) -> torch.Tensor:
    """Combine [1, T, 32, 8, 512] BF16 partials using FP32 base-2 LSE.

    T is one or four. Both inputs must be contiguous and on the same gfx950
    device; partial_o must be eight-byte aligned. The binding checks the complete
    tensor contract. Output is [1, T, 8, 512] BF16. Compilation and launch errors
    propagate to the caller.
    """
    if partial_o.ndim != 5:
        raise ValueError("partial_o must have five dimensions")
    if not supported_device(partial_o.device):
        raise ValueError("split-KV HIP combine requires a gfx950 tensor")
    with torch.cuda.device(partial_o.device):
        out = torch.empty(
            (1, partial_o.shape[1], 8, 512),
            dtype=torch.bfloat16,
            device=partial_o.device,
        )
        build().run(partial_o, partial_lse, out)
    return out


def combine_or_fallback(
    partial_o: torch.Tensor,
    partial_lse: torch.Tensor,
    heads: int,
    value_width: int,
    splits: int,
    fallback: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Select the one/four-token specialization; count unsupported dispatches.

    Tensor-contract violations on a selected specialization raise in its binding.
    Prefill and other shapes retain the caller's TileLang implementation.
    """
    tokens = partial_o.shape[1] if partial_o.ndim >= 2 else -1
    if (
        (heads, value_width, splits) != (8, 512, 32)
        or tokens not in (1, 4)
        or not supported_device(partial_o.device)
    ):
        key = (heads, value_width, splits, tokens, str(partial_o.device))
        _fallback_counts[key] += 1
        if _fallback_counts[key] == 1:
            logger.debug("TileLang split-KV combine fallback: %s", key)
        return fallback(partial_o, partial_lse)
    return split_kv_combine_hip(partial_o, partial_lse)
