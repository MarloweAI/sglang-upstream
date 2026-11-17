"""SGLANG_OPT_DSV41_DSPARK_CYCLE_RING: a replayed cycle's result comes back through the
next replay's window launch, or through a host copy when no replay follows.

A captured "cycle" stands in for the cycle graph: the verify window launch first (it
publishes the previous step's span into the host ring), a sleep, then an "accept" that
copies this step's tokens and commit lengths into an epilogue-shaped device storage. The
host reads each step's result one step late, as the overlap scheduler does, and between
replays it sometimes runs another step (the ring copies the pending result instead) or
reads with nothing queued after it (an idle scheduler). Every result must equal what its
step wrote."""

import random

import pytest
import torch

from sglang.kernels.ops.speculative.cache_locs import (
    assign_extend_cache_locs_uniform_func,
)
from sglang.srt.speculative.dspark_components.dspark_cycle_ring import (
    CycleResultRing,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

MAX_BS, STRIDE, POOL = 4, 6, 4096


class _Result:
    def __init__(self, tokens, accept):
        self.next_token_ids, self.accept_lens, self.block_accept_lens = tokens, accept, None
        self.copy_done = None


def _cycle(bs, ring, results, staged, rows, r2t, relay, outs):
    tokens = results["tokens"][:bs].reshape(-1)
    accept = results["accept"][:bs]
    publish = ring.publish(tokens, accept)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        graph.capture_begin()
        assign_extend_cache_locs_uniform_func(
            req_pool_indices=rows[:bs],
            req_to_token=r2t,
            start_offset=relay,
            batch_size=bs,
            draft_token_num=STRIDE,
            device=rows.device,
            out=outs[: bs * STRIDE],
            start_by_row=True,
            publish=publish,
        )
        torch.cuda._sleep(3000)
        results["tokens"][:bs].copy_(staged["tokens"][:bs])
        results["accept"][:bs].copy_(staged["accept"][:bs])
        graph.capture_end()
    return graph, ring.publish_key(tokens, accept)


def _check(done, want_tok, want_acc, step) -> str:
    """Wait for a result, compare it with what its step wrote; how it came back."""
    done.copy_done.synchronize()
    assert done.accept_lens.tolist() == want_acc, step
    assert done.next_token_ids.tolist() == want_tok, step
    return "published" if done.copy_done._published is not None else "copied"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("bs", [1, 3])
def test_every_result_arrives_once_and_whole(bs):
    dev = torch.device("cuda")
    gen = random.Random(bs)
    n_out = MAX_BS * STRIDE * 8
    storage = torch.zeros(n_out + MAX_BS * 4, dtype=torch.uint8, device=dev)
    results = {
        "tokens": storage[:n_out].view(torch.int64).view(MAX_BS, STRIDE),
        "accept": storage[n_out:].view(torch.int32),
    }
    staged = {
        "tokens": torch.zeros((MAX_BS, STRIDE), dtype=torch.int64, device=dev),
        "accept": torch.zeros((MAX_BS,), dtype=torch.int32, device=dev),
    }
    rows = torch.arange(MAX_BS, dtype=torch.int64, device=dev)
    r2t = torch.arange(MAX_BS * POOL, dtype=torch.int32, device=dev).view(MAX_BS, POOL)
    relay = torch.full((MAX_BS,), 7, dtype=torch.int64, device=dev)
    outs = torch.empty((MAX_BS * STRIDE,), dtype=torch.int64, device=dev)
    ring = CycleResultRing()
    graph, key = _cycle(bs, ring, results, staged, rows, r2t, relay, outs)
    main = torch.cuda.current_stream()
    pending = []
    ways = {"published": 0, "copied": 0}
    for step in range(3000):
        tok = torch.tensor(
            [[gen.randrange(-1, 129280) for _ in range(STRIDE)] for _ in range(bs)],
            dtype=torch.int64,
        )
        acc = torch.tensor([gen.randrange(0, STRIDE + 1) for _ in range(bs)], dtype=torch.int32)
        kind = gen.random()
        if kind < 0.1:
            # another step (a prefill, a per-stage step): the pending result is copied
            ring.before_step(None)
            torch.cuda._sleep(2000)
        staged["tokens"][:bs].copy_(tok.pin_memory(), non_blocking=True)
        staged["accept"][:bs].copy_(acc.pin_memory(), non_blocking=True)
        ring.before_step(key)
        graph.replay()
        result = _Result(results["tokens"][:bs].reshape(-1), results["accept"][:bs])
        ring.issue(result, main)
        pending.append((result, tok.reshape(-1).tolist(), acc.tolist()))
        if kind > 0.97:
            # an idle scheduler: read the latest result with nothing queued after it
            while pending:
                ways[_check(*pending.pop(0), step)] += 1
            continue
        if len(pending) == 2:
            ways[_check(*pending.pop(0), step)] += 1
    for item in pending:
        ways[_check(*item, -1)] += 1
    assert ways["published"] > 1000 and ways["copied"] > 100, ways
    # the window's own outputs are unchanged by the publish
    assert outs[: bs * STRIDE].tolist() == r2t[:bs, 7 : 7 + STRIDE].reshape(-1).tolist()
