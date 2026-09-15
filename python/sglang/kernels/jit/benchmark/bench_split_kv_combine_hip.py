"""Generated-shape benchmark for the gfx950 TileLang and HIP split-KV combines.

Each timed graph traverses a 512 MiB partial-output ring once. Five warmup
replays are discarded. Results include graph replay overhead, exclude input
construction, and describe generated partials rather than a serving workload.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from sglang.kernels.ops.attention.dsa.split_kv_combine_hip import (
    split_kv_combine_hip,
    supported_device,
)
from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
    sparse_mla_fwd_decode_combine,
)


def run(tokens: int, ring_mib: int, replays: int, first: str) -> dict:
    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or not supported_device(device):
        raise RuntimeError("This benchmark requires gfx950")
    if min(ring_mib, replays) < 1:
        raise ValueError("ring-mib and replays must be positive")
    incumbent = sparse_mla_fwd_decode_combine(8, 512, 2048, 4, block_I=64, threads=256)
    rows = []
    generator = torch.Generator().manual_seed(7)
    po = (
        torch.randn(1, tokens, 32, 8, 512, generator=generator)
        .to(torch.bfloat16)
        .to(device)
    )
    pl = (torch.randn(1, tokens, 32, 8, generator=generator) * 40).to(device)
    torch.testing.assert_close(
        split_kv_combine_hip(po, pl), incumbent(po, pl), rtol=2**-7, atol=0
    )
    count = ring_mib * 1024**2 // (po.numel() * po.element_size())
    po_ring = po.unsqueeze(0).expand(count, *po.shape).contiguous()
    pl_ring = pl.unsqueeze(0).expand(count, *pl.shape).contiguous()
    inputs = [(po_ring[i], pl_ring[i]) for i in range(count)]
    order = (first, "hip" if first == "tilelang" else "tilelang")
    for arm in order:
        op = incumbent if arm == "tilelang" else split_kv_combine_hip
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for a, b in inputs[:16]:
                op(a, b)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = [op(a, b) for a, b in inputs]
            for _ in range(5):
                graph.replay()
            stream.synchronize()
            samples = []
            for _ in range(replays):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                graph.replay()
                end.record(stream)
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000 / count)
        rows.append(
            {
                "tokens": tokens,
                "arm": arm,
                "calls_per_replay": count,
                "partial_o_ring_bytes": po_ring.numel() * po_ring.element_size(),
                "partial_lse_ring_bytes": pl_ring.numel() * pl_ring.element_size(),
                "us_per_call": samples,
                "median_us": statistics.median(samples),
            }
        )
        del graph, outputs
    del inputs, po_ring, pl_ring
    torch.cuda.synchronize(device)
    modules = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "sgl_kernel_jit_split_kv_combine_hip.so" in line
    }
    if not modules:
        raise RuntimeError("No loaded HIP combine module found")
    return {
        "native_module_sha256": sorted(
            hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in modules
        ),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "input_provenance": "Seeded BF16 normal partials and FP32 normal base-2 LSE scaled by 40",
        "scope": "Generated-shape graph-replay benchmark; one shape and one paired trial per process",
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ring-mib", type=int, default=512)
    parser.add_argument("--replays", type=int, default=30)
    parser.add_argument("--tokens", type=int, choices=(1, 4), default=4)
    parser.add_argument("--first", choices=("tilelang", "hip"), default="tilelang")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = (
        json.dumps(run(args.tokens, args.ring_mib, args.replays, args.first), indent=2)
        + "\n"
    )
    if args.output is not None:
        args.output.write_text(result)
    print(result)
