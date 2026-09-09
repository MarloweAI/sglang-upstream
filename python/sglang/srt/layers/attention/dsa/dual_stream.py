CUDA_DUAL_STREAM_TOKEN_THRESHOLD = 1024


def can_use_dsa_indexer_dual_stream(num_tokens: int, *, is_cuda: bool) -> bool:
    """Return whether the DSA indexer supports dual-stream for this token count."""
    # Keep this CUDA-only; ROCm multi-stream still serves independent MoE and
    # context-parallel paths.
    return is_cuda and 0 < num_tokens <= CUDA_DUAL_STREAM_TOKEN_THRESHOLD
