"""A replayed DSpark cycle's result without per-cycle events
(SGLANG_OPT_DSV41_DSPARK_CYCLE_RING).

The static accept leaves a cycle's out tokens and commit lengths in one device span
(``DsparkVerifyEpilogue``'s result buffers). After each cycle graph launch the host
enqueues one copy of that span into the next row of a pinned ring, which it filled with
-1 first. The result's ``copy_done`` is a ``RingRowDone``: it waits until the row's
commit lengths and the tokens they cover no longer read -1 (every valid value is >= 0,
so the check holds for any order in which the copy lands). No result event, no
``copy_to_cpu``.

The FutureMap's publish event becomes a ``LazyPublishEvent``: a replay marks the stream
instead of recording, and the event is recorded on that stream when someone waits on it
(a later per-stage step, a mixed batch), so it still fences the latest cycle.
"""

from __future__ import annotations

import ctypes
from typing import Optional

import numpy as np
import torch

_RING_ROWS = 8
_HIP_MEMCPY_D2H = 2
_hip = None


def _memcpy_d2h_async(dst_ptr: int, src_ptr: int, nbytes: int, stream: int) -> None:
    global _hip
    if _hip is None:
        _hip = ctypes.CDLL("libamdhip64.so")
        _hip.hipMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        _hip.hipMemcpyAsync.restype = ctypes.c_int
    err = _hip.hipMemcpyAsync(dst_ptr, src_ptr, nbytes, _HIP_MEMCPY_D2H, stream)
    if err != 0:
        raise RuntimeError(f"DSpark cycle ring: hipMemcpyAsync failed ({err})")


class LazyPublishEvent(torch.cuda.Event):
    """A publish event a replay marks instead of recording; recorded on the marked
    stream when it is waited on or queried."""

    _pending: Optional[torch.cuda.Stream] = None

    def mark(self, stream: torch.cuda.Stream) -> None:
        self._pending = stream

    def _flush(self) -> None:
        stream, self._pending = self._pending, None
        if stream is not None:
            super().record(stream)

    def record(self, stream=None) -> None:
        self._pending = None
        super().record(stream)

    def synchronize(self) -> None:
        self._flush()
        super().synchronize()

    def wait(self, stream=None) -> None:
        self._flush()
        super().wait(stream)

    def query(self) -> bool:
        self._flush()
        return super().query()


class RingRowDone:
    """``copy_done`` of a cycle whose result went to a ring row."""

    def __init__(self, accept: np.ndarray, tokens32: np.ndarray) -> None:
        self._accept = accept  # [bs] int32
        self._tokens32 = tokens32  # [bs, 2 * width] int32 halves of the int64 tokens

    def _landed(self) -> bool:
        acc = self._accept
        if (acc == -1).any():
            return False
        for i, n in enumerate(acc.tolist()):
            if (self._tokens32[i, : 2 * n] == -1).any():
                return False
        return True

    def record(self, *args, **kwargs) -> None:
        pass

    def query(self) -> bool:
        return self._landed()

    def synchronize(self) -> None:
        while not self._landed():
            pass


class CycleResultRing:
    """Pinned rows a replayed cycle's result span is copied into."""

    def __init__(self) -> None:
        self._key = None
        self._host: Optional[torch.Tensor] = None
        self._pos = 0

    @staticmethod
    def check(result) -> None:
        """At record time: the result is what the ring returns (out tokens and commit
        lengths in one storage, nothing else to copy)."""
        tokens, accept = result.next_token_ids, result.accept_lens
        if not (isinstance(tokens, torch.Tensor) and isinstance(accept, torch.Tensor)):
            raise RuntimeError("DSpark cycle ring: the result has no device tokens or lengths")
        if tokens.untyped_storage().data_ptr() != accept.untyped_storage().data_ptr():
            raise RuntimeError("DSpark cycle ring: tokens and lengths in separate storages")
        if tokens.dtype != torch.int64 or accept.dtype != torch.int32:
            raise RuntimeError("DSpark cycle ring: unexpected result dtypes")
        if not (tokens.is_contiguous() and accept.is_contiguous()):
            raise RuntimeError("DSpark cycle ring: non-contiguous result")
        block = result.block_accept_lens
        if block is not None and block is not accept:
            raise RuntimeError("DSpark cycle ring: block accept lengths of their own")
        logits = result.logits_output
        extras = [
            result.cap_lens,
            result.routed_experts_output,
            result.indexer_topk_output,
            result.expert_distribution_metrics,
            result.forward_auxiliary_output,
        ]
        if logits is not None:
            extras += [
                getattr(logits, "sampling_mask_output", None),
                getattr(logits, "auxiliary_device_output", None),
                logits.next_token_logprobs,
            ]
        if any(x is not None for x in extras):
            raise RuntimeError("DSpark cycle ring: the result carries more than tokens and lengths")

    def issue(self, result) -> None:
        """Enqueue the copy of ``result``'s span into the next row (current stream);
        point the result at that row."""
        tokens, accept = result.next_token_ids, result.accept_lens
        base = tokens.untyped_storage().data_ptr()
        t_off = tokens.data_ptr() - base
        a_off = accept.data_ptr() - base
        lo = min(t_off, a_off)
        hi = max(t_off + tokens.numel() * 8, a_off + accept.numel() * 4)
        key = (base, lo, hi, t_off, a_off, tuple(tokens.shape), accept.numel())
        if key != self._key:
            self._key = key
            stride = (hi - lo + 63) // 64 * 64  # rows 8-byte aligned for the token view
            self._host = torch.empty((_RING_ROWS, stride), dtype=torch.uint8, pin_memory=True)
            self._pos = 0
        row = self._host[self._pos % _RING_ROWS, : hi - lo]
        self._pos += 1
        host_tokens = row[t_off - lo : t_off - lo + tokens.numel() * 8].view(torch.int64)
        host_tokens = host_tokens.view(tokens.shape)
        host_accept = row[a_off - lo : a_off - lo + accept.numel() * 4].view(torch.int32)
        accept_np = host_accept.numpy()
        tokens32 = host_tokens.view(torch.int32).view(accept.numel(), -1).numpy()
        accept_np.fill(-1)
        tokens32.fill(-1)
        _memcpy_d2h_async(
            row.data_ptr(), base + lo, hi - lo, torch.cuda.current_stream().cuda_stream
        )
        block_is_accept = result.block_accept_lens is accept
        result.next_token_ids = host_tokens
        result.accept_lens = host_accept
        if block_is_accept:
            result.block_accept_lens = host_accept
        result.copy_done = RingRowDone(accept_np, tokens32)
