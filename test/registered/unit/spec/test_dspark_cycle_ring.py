"""SGLANG_OPT_DSV41_DSPARK_CYCLE_RING: a ring row is read only after its copy completed.

Each step writes fresh out tokens and commit lengths into one epilogue-shaped device
storage behind a sleep kernel, and the ring copies them into its next row. Once the
result's ``copy_done`` reports the step done, the host row holds that step's values: the
span copy can land a word piece by piece, so nothing may read the row earlier."""

import pytest
import torch

from sglang.srt.speculative.dspark_components.dspark_cycle_ring import (
    CycleResultRing,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd-mi35x")

MAX_BS, STRIDE = 64, 6


class _Result:
    def __init__(self, tokens, accept):
        self.next_token_ids, self.accept_lens, self.block_accept_lens = tokens, accept, None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("bs", [1, 3])
def test_a_done_row_holds_its_step(bs):
    dev = torch.device("cuda")
    n_out = MAX_BS * STRIDE * 8
    results = torch.zeros(n_out + MAX_BS * 4, dtype=torch.uint8, device=dev)
    tokens = results[:n_out].view(torch.int64).view(MAX_BS, STRIDE)
    commit = results[n_out:].view(torch.int32)
    gen = torch.Generator(device=dev).manual_seed(0)
    ring = CycleResultRing()
    pending = []
    for step in range(4000):
        torch.cuda._sleep(2000)
        torch.randint(1, STRIDE + 1, (bs,), generator=gen, device=dev, dtype=torch.int32, out=commit[:bs])
        torch.randint(0, 129280, (bs, STRIDE), generator=gen, device=dev, out=tokens[:bs])
        want = (commit[:bs].clone(), tokens[:bs].clone())
        result = _Result(tokens[:bs], commit[:bs])
        ring.issue(result)
        pending.append((result, want))
        # the scheduler reads a step's result while the next one runs
        if len(pending) == 2:
            done, (want_commit, want_tokens) = pending.pop(0)
            done.copy_done.synchronize()
            assert done.accept_lens.tolist() == want_commit.tolist(), step
            assert done.next_token_ids.tolist() == want_tokens.tolist(), step
