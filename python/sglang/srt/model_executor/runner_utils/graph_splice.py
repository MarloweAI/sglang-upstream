"""Splicing captured HIP graphs into an enclosing stream capture.

A graph that replays other graphs inside its capture (SGLANG_OPT_DSV41_DSPARK_CYCLE_GRAPH:
one graph a DSpark decode cycle) cannot launch them: a launch is not capturable. Instead,
while ``splicing()`` is active, ``torch.cuda.CUDAGraph.replay`` on a capturing stream adds
copies of the graph's nodes to the capture, after the stream's work so far, and the
stream continues after them. The copies are flat (no child-graph node): on MI355X a
child-graph boundary costs ≈4 µs of GPU time and a separate graph launch ≈7 µs, a copied
node nothing beyond the node itself.

The spliced graphs must keep their source graph: ``new_cuda_graph`` keeps it when splicing
is configured. A copy reads and writes the same addresses as the original, so
it stays valid while the original's memory pool and buffers do; ``graphs_epoch`` counts
captures of spliceable graphs so a holder of copies can tell when an original was
captured again. Torch's per-replay generator offsets are not carried into a copy: graphs
spliced here must not draw from torch's generators (DSpark's samplers use their own
device counters).

HIP only (``libamdhip64``).
"""

from __future__ import annotations

import ctypes
import threading
from contextlib import contextmanager
from typing import Iterator, List, Optional

import torch

_KEEP_GRAPHS: Optional[bool] = None
_EPOCH = 0
_local = threading.local()

_vp = ctypes.c_void_p
_P = ctypes.POINTER


class _Dim3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint), ("y", ctypes.c_uint), ("z", ctypes.c_uint)]


class _KernelNodeParams(ctypes.Structure):
    _fields_ = [
        ("blockDim", _Dim3),
        ("extra", _P(_vp)),
        ("func", _vp),
        ("gridDim", _Dim3),
        ("kernelParams", _P(_vp)),
        ("sharedMemBytes", ctypes.c_uint),
    ]


class _MemsetParams(ctypes.Structure):
    _fields_ = [
        ("dst", _vp),
        ("elementSize", ctypes.c_uint),
        ("height", ctypes.c_size_t),
        ("pitch", ctypes.c_size_t),
        ("value", ctypes.c_uint),
        ("width", ctypes.c_size_t),
    ]


class _Pos(ctypes.Structure):
    _fields_ = [("x", ctypes.c_size_t), ("y", ctypes.c_size_t), ("z", ctypes.c_size_t)]


class _PitchedPtr(ctypes.Structure):
    _fields_ = [
        ("ptr", _vp),
        ("pitch", ctypes.c_size_t),
        ("xsize", ctypes.c_size_t),
        ("ysize", ctypes.c_size_t),
    ]


class _Extent(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_size_t),
        ("height", ctypes.c_size_t),
        ("depth", ctypes.c_size_t),
    ]


class _Memcpy3DParms(ctypes.Structure):
    _fields_ = [
        ("srcArray", _vp),
        ("srcPos", _Pos),
        ("srcPtr", _PitchedPtr),
        ("dstArray", _vp),
        ("dstPos", _Pos),
        ("dstPtr", _PitchedPtr),
        ("extent", _Extent),
        ("kind", ctypes.c_int),
    ]


_KERNEL, _MEMCPY, _MEMSET, _GRAPH, _EMPTY = 0, 1, 2, 4, 5
_CAPTURE_ACTIVE = 1
_SET_CAPTURE_DEPENDENCIES = 1


class _Hip:
    def __init__(self) -> None:
        lib = ctypes.CDLL("libamdhip64.so")

        def fn(name, *args):
            f = getattr(lib, name)
            f.argtypes = list(args)
            f.restype = ctypes.c_int
            return f

        size_p = _P(ctypes.c_size_t)
        self.capture_info = fn(
            "hipStreamGetCaptureInfo_v2",
            _vp,
            _P(ctypes.c_int),
            _P(ctypes.c_ulonglong),
            _P(_vp),
            _P(_P(_vp)),
            size_p,
        )
        self.set_capture_deps = fn(
            "hipStreamUpdateCaptureDependencies",
            _vp,
            _P(_vp),
            ctypes.c_size_t,
            ctypes.c_uint,
        )
        self.get_nodes = fn("hipGraphGetNodes", _vp, _P(_vp), size_p)
        self.get_edges = fn("hipGraphGetEdges", _vp, _P(_vp), _P(_vp), size_p)
        self.node_type = fn("hipGraphNodeGetType", _vp, _P(ctypes.c_int))
        self.kernel_get = fn(
            "hipGraphKernelNodeGetParams", _vp, _P(_KernelNodeParams)
        )
        self.kernel_add = fn(
            "hipGraphAddKernelNode",
            _P(_vp),
            _vp,
            _P(_vp),
            ctypes.c_size_t,
            _P(_KernelNodeParams),
        )
        self.kernel_attrs = fn("hipGraphKernelNodeCopyAttributes", _vp, _vp)
        self.memset_get = fn("hipGraphMemsetNodeGetParams", _vp, _P(_MemsetParams))
        self.memset_add = fn(
            "hipGraphAddMemsetNode",
            _P(_vp),
            _vp,
            _P(_vp),
            ctypes.c_size_t,
            _P(_MemsetParams),
        )
        self.memcpy_get = fn("hipGraphMemcpyNodeGetParams", _vp, _P(_Memcpy3DParms))
        self.memcpy_add = fn(
            "hipGraphAddMemcpyNode",
            _P(_vp),
            _vp,
            _P(_vp),
            ctypes.c_size_t,
            _P(_Memcpy3DParms),
        )
        self.memcpy1d_add = fn(
            "hipGraphAddMemcpyNode1D",
            _P(_vp),
            _vp,
            _P(_vp),
            ctypes.c_size_t,
            _vp,
            _vp,
            ctypes.c_size_t,
            ctypes.c_int,
        )
        self.empty_add = fn(
            "hipGraphAddEmptyNode", _P(_vp), _vp, _P(_vp), ctypes.c_size_t
        )
        self.child_graph = fn("hipGraphChildGraphNodeGetGraph", _vp, _P(_vp))


_HIP: Optional[_Hip] = None


def _hip() -> _Hip:
    global _HIP
    if _HIP is None:
        _HIP = _Hip()
    return _HIP


def _check(err: int, what: str) -> None:
    if err != 0:
        raise RuntimeError(f"{what} failed: hip error {err}")


def _array(nodes: List[int]):
    return (_vp * max(len(nodes), 1))(*nodes)


def _nodes(graph: int) -> List[int]:
    hip, n = _hip(), ctypes.c_size_t()
    _check(hip.get_nodes(_vp(graph), None, ctypes.byref(n)), "hipGraphGetNodes")
    out = (_vp * max(n.value, 1))()
    _check(hip.get_nodes(_vp(graph), out, ctypes.byref(n)), "hipGraphGetNodes")
    return [out[i] for i in range(n.value)]


def _edges(graph: int):
    hip, n = _hip(), ctypes.c_size_t()
    _check(hip.get_edges(_vp(graph), None, None, ctypes.byref(n)), "hipGraphGetEdges")
    src, dst = (_vp * max(n.value, 1))(), (_vp * max(n.value, 1))()
    _check(hip.get_edges(_vp(graph), src, dst, ctypes.byref(n)), "hipGraphGetEdges")
    return [(src[i], dst[i]) for i in range(n.value)]


def _type(node: int) -> int:
    t = ctypes.c_int()
    _check(_hip().node_type(_vp(node), ctypes.byref(t)), "hipGraphNodeGetType")
    return t.value


def _copy_node(dst: int, node: int, deps: List[int]) -> int:
    hip, new, d = _hip(), _vp(), _array(deps)
    kind = _type(node)
    if kind == _KERNEL:
        p = _KernelNodeParams()
        _check(hip.kernel_get(_vp(node), ctypes.byref(p)), "kernel node params")
        _check(
            hip.kernel_add(ctypes.byref(new), _vp(dst), d, len(deps), ctypes.byref(p)),
            "hipGraphAddKernelNode",
        )
        _check(hip.kernel_attrs(_vp(node), new), "kernel node attributes")
    elif kind == _MEMCPY:
        p = _Memcpy3DParms()
        _check(hip.memcpy_get(_vp(node), ctypes.byref(p)), "memcpy node params")
        flat = (
            not p.srcArray
            and not p.dstArray
            and p.extent.height <= 1
            and p.extent.depth <= 1
            and not (p.srcPos.x or p.srcPos.y or p.srcPos.z)
            and not (p.dstPos.x or p.dstPos.y or p.dstPos.z)
        )
        if flat:
            # A captured 1D copy reads back as 3D parameters; a 3D node copies slower.
            err = hip.memcpy1d_add(
                ctypes.byref(new),
                _vp(dst),
                d,
                len(deps),
                p.dstPtr.ptr,
                p.srcPtr.ptr,
                p.extent.width,
                p.kind,
            )
        else:
            err = hip.memcpy_add(
                ctypes.byref(new), _vp(dst), d, len(deps), ctypes.byref(p)
            )
        _check(err, "memcpy node")
    elif kind == _MEMSET:
        p = _MemsetParams()
        _check(hip.memset_get(_vp(node), ctypes.byref(p)), "memset node params")
        _check(
            hip.memset_add(ctypes.byref(new), _vp(dst), d, len(deps), ctypes.byref(p)),
            "hipGraphAddMemsetNode",
        )
    elif kind == _EMPTY:
        _check(
            hip.empty_add(ctypes.byref(new), _vp(dst), d, len(deps)),
            "hipGraphAddEmptyNode",
        )
    else:
        raise RuntimeError(f"a graph node of type {kind} cannot be spliced")
    return new.value


def _copy_graph(dst: int, src: int, deps: List[int]) -> List[int]:
    """Copies of ``src``'s nodes in ``dst``, after ``deps``; returns the copies the
    rest of ``dst`` must follow."""
    nodes = _nodes(src)
    preds = {n: [] for n in nodes}
    has_succ = set()
    for a, b in _edges(src):
        preds[b].append(a)
        has_succ.add(a)
    ends = {}  # source node -> the copies that complete it
    pending = nodes
    while pending:  # topological order
        later = []
        for n in pending:
            if any(p not in ends for p in preds[n]):
                later.append(n)
                continue
            d = [x for p in preds[n] for x in ends[p]] if preds[n] else list(deps)
            if _type(n) == _GRAPH:
                child = _vp()
                _check(_hip().child_graph(_vp(n), ctypes.byref(child)), "child graph")
                ends[n] = _copy_graph(dst, child.value, d) or d
            else:
                ends[n] = [_copy_node(dst, n, d)]
        if len(later) == len(pending):
            raise RuntimeError("graph has a cycle")
        pending = later
    return [x for n in nodes if n not in has_succ for x in ends[n]]


def splice_into_capture(raw_graph: int) -> int:
    """Add a copy of ``raw_graph`` to the current stream's capture, ordered after the
    stream's captured work; the stream's later work follows it. Returns the number of
    nodes copied."""
    hip = _hip()
    stream = _vp(torch.cuda.current_stream().cuda_stream)
    status, cid, graph = ctypes.c_int(), ctypes.c_ulonglong(), _vp()
    deps_p, num = _P(_vp)(), ctypes.c_size_t()
    _check(
        hip.capture_info(
            stream,
            ctypes.byref(status),
            ctypes.byref(cid),
            ctypes.byref(graph),
            ctypes.byref(deps_p),
            ctypes.byref(num),
        ),
        "hipStreamGetCaptureInfo_v2",
    )
    if status.value != _CAPTURE_ACTIVE:
        raise RuntimeError("splice_into_capture: the current stream is not capturing")
    deps = [deps_p[i] for i in range(num.value)]
    before = len(_nodes(graph.value))
    ends = _copy_graph(graph.value, raw_graph, deps)
    if ends:
        _check(
            hip.set_capture_deps(
                stream, _array(ends), len(ends), _SET_CAPTURE_DEPENDENCIES
            ),
            "hipStreamUpdateCaptureDependencies",
        )
    return len(_nodes(graph.value)) - before


def new_cuda_graph() -> torch.cuda.CUDAGraph:
    """A CUDA graph for a site whose graphs may be spliced: it keeps its source graph
    when splicing is configured. Counts toward ``graphs_epoch``."""
    global _EPOCH
    _EPOCH += 1
    if splicing_configured():
        return torch.cuda.CUDAGraph(keep_graph=True)
    return torch.cuda.CUDAGraph()


def splicing_configured() -> bool:
    """SGLANG_OPT_DSV41_DSPARK_CYCLE_GRAPH: graphs keep their source graph and a replay
    inside ``splicing()`` splices."""
    global _KEEP_GRAPHS
    if _KEEP_GRAPHS is None:
        from sglang.srt.environ import envs

        _KEEP_GRAPHS = bool(envs.SGLANG_OPT_DSV41_DSPARK_CYCLE_GRAPH.get())
        if _KEEP_GRAPHS:
            torch.cuda.CUDAGraph.replay = _splicing_replay
    return _KEEP_GRAPHS


def graphs_epoch() -> int:
    return _EPOCH


_orig_replay = torch.cuda.CUDAGraph.replay


def _splicing_replay(self) -> None:
    if getattr(_local, "splicing", False) and torch.cuda.is_current_stream_capturing():
        _local.spliced_nodes = getattr(_local, "spliced_nodes", 0) + (
            splice_into_capture(self.raw_cuda_graph())
        )
        _local.spliced_graphs = getattr(_local, "spliced_graphs", 0) + 1
        return
    _orig_replay(self)


@contextmanager
def splicing() -> Iterator[dict]:
    """Within: a graph replay on a capturing stream (this thread) splices the graph
    into the capture. Yields a dict that gets ``graphs`` and ``nodes`` spliced."""
    if not splicing_configured():
        raise RuntimeError("splicing needs SGLANG_OPT_DSV41_DSPARK_CYCLE_GRAPH")
    _local.splicing, _local.spliced_nodes, _local.spliced_graphs = True, 0, 0
    stats: dict = {}
    try:
        yield stats
    finally:
        _local.splicing = False
        stats["graphs"] = _local.spliced_graphs
        stats["nodes"] = _local.spliced_nodes


def is_splicing() -> bool:
    return getattr(_local, "splicing", False)


_HOST_TO_DEVICE, _DEVICE_TO_HOST = 1, 2


def describe_graph(raw_graph: int) -> dict:
    """Node counts of a graph by kind, and its host<->device copies (which replay the
    host buffer's contents at launch time, not at capture time)."""
    hip = _hip()
    out = {"kernel": 0, "memcpy": 0, "memset": 0, "other": 0, "h2d": 0, "d2h": 0}
    for n in _nodes(raw_graph):
        kind = _type(n)
        if kind == _KERNEL:
            out["kernel"] += 1
        elif kind == _MEMSET:
            out["memset"] += 1
        elif kind == _MEMCPY:
            out["memcpy"] += 1
            p = _Memcpy3DParms()
            _check(hip.memcpy_get(_vp(n), ctypes.byref(p)), "memcpy node params")
            out["h2d"] += p.kind == _HOST_TO_DEVICE
            out["d2h"] += p.kind == _DEVICE_TO_HOST
        else:
            out["other"] += 1
    return out
