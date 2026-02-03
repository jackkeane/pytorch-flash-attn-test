# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains utilities for testing and using Flash Attention via PyTorch's native Scaled Dot-Product Attention (SDPA) backend. It avoids the need to install the `flash-attn` package separately.

## Running Tests

```bash
python test_flash_attn.py
```

This runs the Flash Attention test suite which validates:
- SDPA with Flash Attention backend
- Causal (autoregressive) attention
- Variable length sequences using nested tensors
- Backward pass gradient computation
- Performance benchmark (Flash Attention vs Math backend)

## Key Technical Details

**SDPA Tensor Shape**: PyTorch's `scaled_dot_product_attention` expects shape `(batch, nheads, seqlen, headdim)`, unlike the `flash-attn` package which uses `(batch, seqlen, nheads, headdim)`.

**Forcing Flash Attention Backend**:
```python
with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
    output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
```

**Variable Length Sequences**: Use nested tensors with jagged layout instead of `flash_attn_varlen_func`:
```python
q = torch.nested.nested_tensor(q_list, layout=torch.jagged)
```

## LLaMA-Factory Integration

See `llamafactory_flash_attention.md` for configuring Flash Attention in LLaMA-Factory. Use `--flash_attn sdpa` to enable PyTorch's native SDPA backend.
