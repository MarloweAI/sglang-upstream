CUDA_DUAL_STREAM_TOKEN_THRESHOLD = 1024


def can_use_dsa_indexer_dual_stream(
    num_tokens: int, *, is_cuda: bool, is_hip: bool
) -> bool:
    """Return whether the DSA indexer supports this dual-stream workload."""
    if num_tokens <= 0:
        return False
    # ROCm stream creation is opt-in. CUDA keeps its established tuning envelope.
    if is_hip:
        return True
    return is_cuda and num_tokens <= CUDA_DUAL_STREAM_TOKEN_THRESHOLD
