"""A replayed DSpark cycle's result through a host ring
(SGLANG_OPT_DSV41_DSPARK_CYCLE_RING).

The static accept leaves a cycle's out tokens and commit lengths in one device span
(``DsparkVerifyEpilogue``'s result buffers). The cycle after it brings them back without
a stream operation between the two graph launches: its recorded verify window launch, the
cycle graph's first kernel, writes the span into the next row of a ring in coherent host
memory (``RingPublish``), each value in its own 8-byte word whose high half is the row's
tag. The host takes a row once every word carries the tag it expects, so a row is never
read half landed, and a word from an older row is never taken for a newer one.

A replayed cycle whose next step is not a replay of a recording that publishes the same
span (a per-stage step, a prefill, another batch, or no step at all) gets its span copied
by the host instead, onto the stream the cycle ran on and before anything else is queued
there, with an event behind the copy; the host reads that row once the event completed.

Every host wait on a ring row has a deadline and raises past it.

The FutureMap's publish event becomes a ``LazyPublishEvent``: a replay marks the stream
instead of recording, and the event is recorded on that stream when someone waits on it
(a later per-stage step, a mixed batch), so it still fences the latest cycle.
"""

from __future__ import annotations

import ctypes
import time
from typing import Optional

import numpy as np
import torch

from sglang.kernels.ops.speculative.cache_locs import RingPublish

_RING_ROWS = 8
_DEADLINE_S = 60.0
_HIP_MEMCPY_D2H = 2
_HIP_HOST_MALLOC_MAPPED = 0x2
_HIP_HOST_MALLOC_COHERENT = 0x40000000
_TAG_MOD = 2147483647
_hip = None


def _hip_lib():
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
        _hip.hipHostMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        _hip.hipHostMalloc.restype = ctypes.c_int
        _hip.hipHostGetDevicePointer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        _hip.hipHostGetDevicePointer.restype = ctypes.c_int
    return _hip


def _memcpy_d2h_async(dst_ptr: int, src_ptr: int, nbytes: int, stream: int) -> None:
    err = _hip_lib().hipMemcpyAsync(dst_ptr, src_ptr, nbytes, _HIP_MEMCPY_D2H, stream)
    if err != 0:
        raise RuntimeError(f"DSpark cycle ring: hipMemcpyAsync failed ({err})")


def _tag(written: int) -> int:
    """The tag of the row a publishing launch writes when ``written`` rows came before."""
    return written % _TAG_MOD + 1


class HostRows:
    """``rows`` x ``words`` int64 in coherent, device-mapped host memory: the device's
    stores reach host memory without a cache flush, and the host reads them in place.
    Passed to a Triton kernel as its pointer (``data_ptr``, ``dtype``)."""

    dtype = torch.int64

    def __init__(self, rows: int, words: int) -> None:
        hip = _hip_lib()
        host = ctypes.c_void_p()
        nbytes = rows * words * 8
        err = hip.hipHostMalloc(
            ctypes.byref(host), nbytes, _HIP_HOST_MALLOC_MAPPED | _HIP_HOST_MALLOC_COHERENT
        )
        if err != 0:
            raise RuntimeError(f"DSpark cycle ring: hipHostMalloc failed ({err})")
        dev = ctypes.c_void_p()
        err = hip.hipHostGetDevicePointer(ctypes.byref(dev), host, 0)
        if err != 0:
            raise RuntimeError(f"DSpark cycle ring: hipHostGetDevicePointer failed ({err})")
        self._dev = dev.value
        # never freed: the recordings that write it live as long as the process
        self.words = words
        self.host = np.ctypeslib.as_array(
            (ctypes.c_int64 * (rows * words)).from_address(host.value)
        ).reshape(rows, words)
        self.host.fill(0)  # no row carries tag 0
        self.counter: Optional[torch.Tensor] = None  # rows written, on the device
        self.written = 0  # publishing launches issued (the counter after them)

    def data_ptr(self) -> int:
        return self._dev

    def take(self, written: int, decode) -> bool:
        """Decode the row the launch after ``written`` rows wrote, once it landed."""
        live = self.host[written % self.host.shape[0]]
        row = live.copy()
        tag = _tag(written)
        if not np.all((row >> 32) == tag):
            return False
        host_tokens, host_accept = decode
        n_acc = host_accept.numel()
        values = (row & 0xFFFFFFFF).astype(np.uint32).view(np.int32)
        host_accept.numpy()[:] = values[:n_acc]
        host_tokens.numpy().reshape(-1)[:] = values[n_acc:]
        if not np.all((live >> 32) == tag):
            raise RuntimeError(
                "DSpark cycle ring: row %d was overwritten while it was read" % written
            )
        return True


class RingResult:
    """``copy_done`` of a replayed cycle: its result arrives either in a tagged row a
    later window launch publishes (``take``) or through a host-issued copy into its own
    pinned row (``flush``)."""

    def __init__(self, ring: "CycleResultRing", slot: int, row, stream, span) -> None:
        self._ring = ring
        self._slot = slot
        self._row = row  # this result's pinned row (bytes), what its fields view
        self._stream = stream  # the stream the cycle ran on
        self.span = span  # (device base, lo, hi, publish key)
        self._published: Optional[int] = None  # rows written before its publishing launch
        self._rows: Optional[HostRows] = None  # the rows that launch writes
        self._event: Optional[torch.cuda.Event] = None
        self._decode = None  # (host tokens, host accept) to fill from a published row
        self._done = False

    # -- how the result comes back --------------------------------------------------

    def take(self, rows: HostRows, written: int) -> None:
        """The publishing launch after ``written`` rows of ``rows`` writes this result."""
        assert self._published is None and self._event is None
        self._rows, self._published = rows, written

    def flush(self) -> None:
        """Copy the span into this result's row on its stream (nothing published it)."""
        if self._published is not None or self._event is not None:
            return
        base, lo, hi, _ = self.span
        _memcpy_d2h_async(self._row.data_ptr(), base + lo, hi - lo, self._stream.cuda_stream)
        self._event = self._ring._done[self._slot]
        self._event.record(self._stream)

    @property
    def pending(self) -> bool:
        return self._published is None and self._event is None

    # -- the copy_done interface ----------------------------------------------------

    def record(self, *args, **kwargs) -> None:
        pass

    def query(self) -> bool:
        if self._done:
            return True
        if self.pending:
            return False
        if self._event is not None:
            self._done = self._event.query()
        else:
            self._done = self._rows.take(self._published, self._decode)
        return self._done

    def synchronize(self) -> None:
        if self._done:
            return
        if self.pending:
            assert self._ring._pending is self
            self._ring._flush_pending()
        deadline = time.monotonic() + _DEADLINE_S
        spins = 0
        while not self.query():
            spins += 1
            if spins % 1024 == 0 and time.monotonic() > deadline:
                raise RuntimeError(
                    "DSpark cycle ring: a cycle's result did not arrive within "
                    f"{_DEADLINE_S:.0f} s ("
                    + (
                        f"published row {self._published}"
                        if self._published is not None
                        else "copied row"
                    )
                    + ")"
                )


class CycleResultRing:
    """The rows replayed cycles' results come back through, and the pending one."""

    def __init__(self) -> None:
        self._key = None
        self._host: Optional[torch.Tensor] = None
        self._pos = 0
        self._done = [torch.cuda.Event() for _ in range(_RING_ROWS)]
        self._rows: dict = {}  # published rows by row length (words)
        self._pending: Optional[RingResult] = None

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

    @staticmethod
    def publish_key(tokens: torch.Tensor, accept: torch.Tensor) -> tuple:
        return (tokens.data_ptr(), tokens.numel(), accept.data_ptr(), accept.numel())

    # -- the publishing launch ------------------------------------------------------

    def publish(self, tokens: torch.Tensor, accept: torch.Tensor) -> RingPublish:
        """What a recorded window launch publishes: ``tokens`` / ``accept``, the
        previous step's result span."""
        words = tokens.numel() + accept.numel()
        rows = self._rows.get(words)
        if rows is None:
            rows = self._rows[words] = HostRows(_RING_ROWS, words)
            rows.counter = torch.zeros((1,), dtype=torch.int64, device=tokens.device)
        return RingPublish(
            tokens=tokens, accept=accept, ring=rows, counter=rows.counter, rows=_RING_ROWS
        )

    # -- per step -------------------------------------------------------------------

    def _flush_pending(self) -> None:
        pending, self._pending = self._pending, None
        if pending is not None:
            pending.flush()

    def before_step(self, publish_key: Optional[tuple]) -> None:
        """Before a step is queued: ``publish_key`` is the span its recorded window
        launch publishes (a replay of a publishing recording), else None."""
        pending, self._pending = self._pending, None
        if publish_key is not None:
            rows = self._rows[publish_key[1] + publish_key[3]]
            if pending is not None and pending.span[3] == publish_key:
                pending.take(rows, rows.written)
            elif pending is not None:
                pending.flush()
            rows.written += 1
        elif pending is not None:
            pending.flush()

    def issue(self, result, stream) -> None:
        """After a replay's launch on ``stream``: point the result at its row; it
        stays pending until the next step decides how it comes back."""
        assert self._pending is None
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
        slot = self._pos % _RING_ROWS
        row = self._host[slot, : hi - lo]
        self._pos += 1
        host_tokens = row[t_off - lo : t_off - lo + tokens.numel() * 8].view(torch.int64)
        host_tokens = host_tokens.view(tokens.shape)
        host_accept = row[a_off - lo : a_off - lo + accept.numel() * 4].view(torch.int32)
        done = RingResult(
            self, slot, row, stream, (base, lo, hi, self.publish_key(tokens, accept))
        )
        done._decode = (host_tokens, host_accept)
        block_is_accept = result.block_accept_lens is accept
        result.next_token_ids = host_tokens
        result.accept_lens = host_accept
        if block_is_accept:
            result.block_accept_lens = host_accept
        result.copy_done = done
        self._pending = done


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
