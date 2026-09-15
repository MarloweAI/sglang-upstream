"""Generated-input numerics and dispatch contracts for the gfx950 HIP combine."""

import sys
from unittest.mock import Mock

import pytest
import torch

from sglang.kernels.ops.attention.dsa import split_kv_combine_hip as hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

CASES = (
    "realistic",
    "wide_dynamic_range",
    "one_dominant",
    "all_equal",
    "near_fp8_max",
    "tiny_magnitudes",
)
SM_SCALE = 576**-0.5


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available() or not hip.supported_device(
        torch.device("cuda:0")
    ):
        pytest.skip("Requires gfx950")
    # A supported device with a broken compiler must fail, not skip.
    with torch.cuda.device(0):
        hip.build()
    return torch.device("cuda:0")


def make_case(kind, tokens, device, pool=131072):
    """Seeded inputs; 'realistic' is a distribution name, not captured data."""
    gen = torch.Generator().manual_seed(20260903)

    def randn(*shape):
        return torch.randn(*shape, generator=gen)

    if kind in ("realistic", "wide_dynamic_range"):
        q, kv = randn(tokens, 8, 576) * 0.125, randn(pool, 576) * 0.125
        if kind == "wide_dynamic_range":
            kv[: pool // 8] *= 64.0
    elif kind == "one_dominant":
        q, kv = randn(tokens, 8, 576) * 0.125, randn(pool, 576) * 0.02
        kv[0] = 4.0
    elif kind == "all_equal":
        q, kv = torch.full((tokens, 8, 576), 0.05), torch.full((pool, 576), 0.05)
    elif kind == "near_fp8_max":
        q, kv = (
            randn(tokens, 8, 576).sign() * (448 * 0.9),
            randn(pool, 576).sign() * (448 * 0.9),
        )
    elif kind == "tiny_magnitudes":
        q, kv = randn(tokens, 8, 576) * 1e-3, randn(pool, 576) * 1e-3
    else:
        raise ValueError(kind)
    q = q.to(torch.bfloat16).to(device).to(torch.float8_e4m3fn)
    kv = kv.to(torch.bfloat16).to(device).to(torch.float8_e4m3fn).reshape(pool, 1, 576)
    indices = torch.stack(
        [torch.randperm(pool, generator=gen)[:2048] for _ in range(tokens)]
    )
    return q, kv, indices.to(device=device, dtype=torch.int32)


def fp32_reference(q, kv, indices):
    outputs = []
    for row in range(q.shape[0]):
        keys = kv.reshape(-1, 576).float()[indices[row].long()]
        scores = torch.einsum("hd,td->ht", q[row].float(), keys) * SM_SCALE
        outputs.append(torch.einsum("ht,td->hd", scores.softmax(-1), keys[:, :512]))
    return torch.stack(outputs).unsqueeze(0)


def incumbent():
    from sglang.kernels.ops.attention.dsa import tilelang_kernel as tl

    stage1 = tl.sparse_mla_fwd_decode_partial_fp8(
        8,
        512,
        64,
        2048,
        sm_scale=SM_SCALE,
        block_I=64,
        inner_iter=1,
        threads=256,
    )
    combine = tl.sparse_mla_fwd_decode_combine(
        8,
        512,
        2048,
        4,
        block_I=64,
        threads=256,
    )
    return stage1, combine


def synthetic(tokens, kind, device):
    gen = torch.Generator().manual_seed(7)
    po = torch.randn(1, tokens, 32, 8, 512, generator=gen).to(torch.bfloat16).to(device)
    if kind == "spread":
        pl = (torch.randn(1, tokens, 32, 8, generator=gen) * 40).to(device)
    elif kind == "one_hot":
        pl = torch.full((1, tokens, 32, 8), -1e4, device=device)
        pl[:, :, 0, :] = 12
    elif kind == "empty_splits":
        pl = torch.randn(1, tokens, 32, 8, generator=gen).to(device)
        pl[:, :, ::4, :] = -(2**30)
    elif kind == "flat":
        pl = torch.zeros(1, tokens, 32, 8, device=device)
    else:
        raise ValueError(kind)
    return po, pl.float().contiguous()


@pytest.mark.parametrize("tokens", [1, 4])
@pytest.mark.parametrize("kind", CASES)
def test_stage1_generated_reference_and_repeatability(device, tokens, kind):
    stage1, combine = incumbent()
    q, kv, indices = make_case(kind, tokens, device)
    po, pl = stage1(q.unsqueeze(0), kv.unsqueeze(0), indices[:, None].unsqueeze(0))
    ref = fp32_reference(q, kv, indices)
    old, got = combine(po, pl), hip.split_kv_combine_hip(po, pl)
    assert torch.isfinite(got).all()
    # The combine may round differently. Require no worse aggregate FP32
    # reference error, allowing one FP32 epsilon of reference peak for the check.
    peak = ref.abs().max()
    old_error = (old.float() - ref).abs().max()
    got_error = (got.float() - ref).abs().max()
    assert got_error <= old_error + torch.finfo(torch.float32).eps * peak
    torch.testing.assert_close(got, old, rtol=2**-7, atol=0)
    for _ in range(3):
        assert torch.equal(got, hip.split_kv_combine_hip(po, pl))
        assert torch.equal(old, combine(po, pl))


@pytest.mark.parametrize("tokens", [1, 4])
@pytest.mark.parametrize("kind", ["flat", "spread", "one_hot", "empty_splits"])
def test_synthetic_partials(device, tokens, kind):
    _, combine = incumbent()
    po, pl = synthetic(tokens, kind, device)
    got = hip.split_kv_combine_hip(po, pl)
    assert torch.equal(got, combine(po, pl))
    weights = torch.softmax(
        pl.float() * torch.log(torch.tensor(2.0, device=device)), dim=2
    )
    ref = (po.float() * weights[..., None]).sum(2)
    torch.testing.assert_close(got.float(), ref, rtol=2**-7, atol=1e-6)


@pytest.mark.parametrize(
    "mutation",
    [
        "batch",
        "tokens",
        "heads",
        "width",
        "splits",
        "lse_shape",
        "po_dtype",
        "lse_dtype",
        "po_stride",
        "lse_stride",
        "alignment",
        "cpu_lse",
    ],
)
def test_reject_invalid_tensor_contract(device, mutation):
    po, pl = synthetic(4, "flat", device)
    if mutation == "batch":
        po = po.expand(2, -1, -1, -1, -1).contiguous()
    elif mutation == "tokens":
        po, pl = po[:, :2].contiguous(), pl[:, :2].contiguous()
    elif mutation == "heads":
        po = po[..., :4, :].contiguous()
    elif mutation == "width":
        po = po[..., :256].contiguous()
    elif mutation == "splits":
        po = po[:, :, :16].contiguous()
    elif mutation == "lse_shape":
        pl = pl[..., :4].contiguous()
    elif mutation == "po_dtype":
        po = po.float()
    elif mutation == "lse_dtype":
        pl = pl.to(torch.bfloat16)
    elif mutation == "po_stride":
        po = po.transpose(1, 2)
    elif mutation == "lse_stride":
        pl = pl.transpose(1, 2)
    elif mutation == "alignment":
        po = torch.empty(po.numel() + 1, device=device, dtype=po.dtype)[1:].reshape(
            po.shape
        )
    elif mutation == "cpu_lse":
        pl = pl.cpu()
    with pytest.raises((RuntimeError, ValueError)):
        hip.split_kv_combine_hip(po, pl)


@pytest.mark.parametrize(
    "heads,width,splits,tokens",
    [(16, 512, 32, 4), (8, 256, 32, 4), (8, 512, 1, 4), (8, 512, 32, 2)],
)
def test_counted_fallback(device, heads, width, splits, tokens):
    po = torch.empty(
        1, tokens, splits, heads, width, dtype=torch.bfloat16, device=device
    )
    pl = torch.empty(1, tokens, splits, heads, dtype=torch.float32, device=device)
    fallback = Mock(return_value=object())
    before = sum(hip._fallback_counts.values())
    got = hip.combine_or_fallback(po, pl, heads, width, splits, fallback)
    assert got is fallback.return_value
    fallback.assert_called_once_with(po, pl)
    assert sum(hip._fallback_counts.values()) == before + 1


def test_selected_errors_propagate(device, monkeypatch):
    po, pl = synthetic(4, "flat", device)
    monkeypatch.setattr(hip, "build", Mock(side_effect=RuntimeError("build failed")))
    fallback = Mock()
    with pytest.raises(RuntimeError, match="build failed"):
        hip.combine_or_fallback(po, pl, 8, 512, 32, fallback)
    fallback.assert_not_called()


def test_stream_and_graph_capture(device):
    po, pl = synthetic(4, "spread", device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        po.mul_(0.5)
        expected = hip.split_kv_combine_hip(po, pl)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            got = hip.split_kv_combine_hip(po, pl)
        graph.replay()
    stream.synchronize()
    assert torch.equal(got, expected)
    # Prove the replay owns the work: change its input and poison its output.
    with torch.cuda.stream(stream):
        po.mul_(0.5)
        got.fill_(float("nan"))
        graph.replay()
        expected = hip.split_kv_combine_hip(po, pl)
    stream.synchronize()
    assert torch.equal(got, expected)


@pytest.mark.parametrize("tokens", [1, 4])
def test_tilelang_caller_selects_native(device, tokens, monkeypatch):
    from sglang.kernels.ops.attention.dsa import tilelang_kernel as tl

    q, kv, indices = make_case("realistic", tokens, device, pool=4096)
    native = Mock(wraps=hip.split_kv_combine_hip)
    monkeypatch.setattr(hip, "split_kv_combine_hip", native)
    got = tl.tilelang_sparse_fwd(q, kv, indices[:, None], SM_SCALE, 512)
    native.assert_called_once()
    stage1, combine = incumbent()
    po, pl = stage1(q.unsqueeze(0), kv.unsqueeze(0), indices[:, None].unsqueeze(0))
    torch.testing.assert_close(got, combine(po, pl), rtol=2**-7, atol=0)


def test_input_device_context(device):
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two GPUs")
    other = torch.device("cuda:1")
    if not hip.supported_device(other):
        pytest.skip("Requires a second gfx950")
    po, pl = synthetic(4, "spread", other)
    with torch.cuda.device(other):
        expected = hip.split_kv_combine_hip(po, pl)
    with torch.cuda.device(device):
        got = hip.split_kv_combine_hip(po, pl)
        assert torch.cuda.current_device() == device.index
        with pytest.raises((RuntimeError, ValueError)):
            hip.split_kv_combine_hip(po, pl.to(device))
    torch.cuda.synchronize(other)
    assert got.device == other and torch.equal(got, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
