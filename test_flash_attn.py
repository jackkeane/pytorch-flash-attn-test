#!/usr/bin/env python3
"""Test script to verify Flash Attention functionality using PyTorch's native SDPA."""

import sys


def check_imports():
    """Check if required packages are importable."""
    print("=" * 60)
    print("1. Checking imports...")
    print("=" * 60)

    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU count: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"   ERROR: PyTorch not installed: {e}")
        return False

    # Check SDPA backends
    print(f"   Flash Attention backend: {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"   Memory efficient backend: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"   Math backend: {torch.backends.cuda.math_sdp_enabled()}")

    return True


def test_flash_attn_func():
    """Test Flash Attention via PyTorch's SDPA."""
    print("\n" + "=" * 60)
    print("2. Testing Flash Attention (SDPA)...")
    print("=" * 60)

    import torch
    import torch.nn.functional as F

    # Create test tensors
    batch_size = 2
    seqlen = 128
    nheads = 8
    headdim = 64

    # SDPA expects (batch, nheads, seqlen, headdim)
    q = torch.randn(batch_size, nheads, seqlen, headdim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch_size, nheads, seqlen, headdim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch_size, nheads, seqlen, headdim, device="cuda", dtype=torch.float16)

    print(f"   Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    print(f"   dtype: {q.dtype}")

    # Run flash attention via SDPA
    with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
        output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    print(f"   Output shape: {output.shape}")
    print(f"   Output dtype: {output.dtype}")

    # Verify output is valid (no NaN or Inf)
    assert not torch.isnan(output).any(), "Output contains NaN!"
    assert not torch.isinf(output).any(), "Output contains Inf!"
    print("   PASSED: Flash Attention (SDPA) works correctly")

    return True


def test_flash_attn_causal():
    """Test causal (autoregressive) attention."""
    print("\n" + "=" * 60)
    print("3. Testing causal attention...")
    print("=" * 60)

    import torch
    import torch.nn.functional as F

    batch_size = 2
    seqlen = 256
    nheads = 8
    headdim = 64

    q = torch.randn(batch_size, nheads, seqlen, headdim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch_size, nheads, seqlen, headdim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch_size, nheads, seqlen, headdim, device="cuda", dtype=torch.float16)

    # Run with is_causal=True for autoregressive attention
    with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    assert not torch.isnan(output).any(), "Causal output contains NaN!"
    assert not torch.isinf(output).any(), "Causal output contains Inf!"
    print(f"   Output shape: {output.shape}")
    print("   PASSED: Causal attention works correctly")

    return True


def test_flash_attn_varlen():
    """Test variable length attention using nested tensors."""
    print("\n" + "=" * 60)
    print("4. Testing variable length attention (nested tensors)...")
    print("=" * 60)

    import torch
    import torch.nn.functional as F

    nheads = 8
    headdim = 64

    # Create variable length sequences (3 sequences of lengths 128, 256, 128)
    seq_lens = [128, 256, 128]

    # Create nested tensors for variable length sequences
    q_list = [torch.randn(nheads, seq_len, headdim, device="cuda", dtype=torch.float16) for seq_len in seq_lens]
    k_list = [torch.randn(nheads, seq_len, headdim, device="cuda", dtype=torch.float16) for seq_len in seq_lens]
    v_list = [torch.randn(nheads, seq_len, headdim, device="cuda", dtype=torch.float16) for seq_len in seq_lens]

    q = torch.nested.nested_tensor(q_list, layout=torch.jagged)
    k = torch.nested.nested_tensor(k_list, layout=torch.jagged)
    v = torch.nested.nested_tensor(v_list, layout=torch.jagged)

    print(f"   Sequence lengths: {seq_lens}")
    print(f"   Nested tensor Q shape: {q.shape}")

    # Run flash attention with nested tensors
    output = F.scaled_dot_product_attention(q, k, v)

    print(f"   Output is nested tensor: {output.is_nested}")
    print("   PASSED: Variable length attention works correctly")

    return True


def test_backward_pass():
    """Test that gradients flow correctly."""
    print("\n" + "=" * 60)
    print("5. Testing backward pass (gradient computation)...")
    print("=" * 60)

    import torch
    import torch.nn.functional as F

    batch_size = 2
    seqlen = 128
    nheads = 4
    headdim = 64

    q = torch.randn(batch_size, nheads, seqlen, headdim, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(batch_size, nheads, seqlen, headdim, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(batch_size, nheads, seqlen, headdim, device="cuda", dtype=torch.float16, requires_grad=True)

    with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
        output = F.scaled_dot_product_attention(q, k, v)
    loss = output.sum()
    loss.backward()

    assert q.grad is not None, "Q gradient is None!"
    assert k.grad is not None, "K gradient is None!"
    assert v.grad is not None, "V gradient is None!"
    assert not torch.isnan(q.grad).any(), "Q gradient contains NaN!"
    assert not torch.isnan(k.grad).any(), "K gradient contains NaN!"
    assert not torch.isnan(v.grad).any(), "V gradient contains NaN!"

    print(f"   Q gradient shape: {q.grad.shape}")
    print(f"   K gradient shape: {k.grad.shape}")
    print(f"   V gradient shape: {v.grad.shape}")
    print("   PASSED: Backward pass works correctly")

    return True


def benchmark_comparison():
    """Compare Flash Attention vs Math backend attention."""
    print("\n" + "=" * 60)
    print("6. Benchmark: Flash Attention vs Math Backend")
    print("=" * 60)

    import torch
    import torch.nn.functional as F
    import time

    batch_size = 4
    seqlen = 1024
    nheads = 16
    headdim = 64

    q = torch.randn(batch_size, nheads, seqlen, headdim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch_size, nheads, seqlen, headdim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch_size, nheads, seqlen, headdim, device="cuda", dtype=torch.float16)

    # Warmup Flash Attention
    with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
        for _ in range(10):
            _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    # Flash Attention benchmark
    start = time.perf_counter()
    with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
        for _ in range(100):
            _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    flash_time = (time.perf_counter() - start) / 100 * 1000

    # Warmup Math backend
    with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.MATH]):
        for _ in range(10):
            _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    # Math backend benchmark
    start = time.perf_counter()
    with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.MATH]):
        for _ in range(100):
            _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    math_time = (time.perf_counter() - start) / 100 * 1000

    print(f"   Sequence length: {seqlen}")
    print(f"   Batch size: {batch_size}, Heads: {nheads}, Head dim: {headdim}")
    print(f"   Flash Attention: {flash_time:.3f} ms")
    print(f"   Math Backend:    {math_time:.3f} ms")
    print(f"   Speedup: {math_time/flash_time:.2f}x")

    return True


def main():
    print("\n" + "#" * 60)
    print("# Flash Attention Test Suite (PyTorch SDPA)")
    print("#" * 60)
    print(f"Python version: {sys.version}")

    # Check imports first
    if not check_imports():
        print("\nFATAL: Required packages not available. Exiting.")
        sys.exit(1)

    import torch
    if not torch.cuda.is_available():
        print("\nFATAL: CUDA not available. Flash Attention requires a GPU.")
        sys.exit(1)

    # Run all tests
    tests = [
        ("Flash Attention (SDPA)", test_flash_attn_func),
        ("Causal Attention", test_flash_attn_causal),
        ("Variable Length (Nested Tensors)", test_flash_attn_varlen),
        ("Backward Pass", test_backward_pass),
        ("Benchmark", benchmark_comparison),
    ]

    results = []
    for name, test_fn in tests:
        try:
            test_fn()
            results.append((name, True, None))
        except Exception as e:
            print(f"\n   FAILED: {e}")
            results.append((name, False, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for name, success, error in results:
        status = "PASS" if success else "FAIL"
        print(f"   [{status}] {name}")
        if error:
            print(f"         Error: {error}")

    print(f"\n   {passed}/{total} tests passed")

    if passed == total:
        print("\n   All tests passed! Flash Attention (via PyTorch SDPA) is working correctly.")
        return 0
    else:
        print("\n   Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
