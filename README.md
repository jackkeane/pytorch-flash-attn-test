# PyTorch Flash Attention Test

A test suite for Flash Attention using PyTorch's native Scaled Dot-Product Attention (SDPA) backend. This approach avoids the need to install the `flash-attn` package separately, which can have complex build requirements.

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- NVIDIA GPU with compute capability 8.0+ (Ampere or newer)

## Installation

No additional installation required beyond PyTorch with CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## Usage

### Run Test Suite

```bash
python test_flash_attn.py
```

The test suite validates:
- Flash Attention via SDPA
- Causal (autoregressive) attention
- Variable length sequences using nested tensors
- Backward pass gradient computation
- Performance benchmark (Flash Attention vs Math backend)

### Using Flash Attention in Your Code

```python
import torch
import torch.nn.functional as F

# Create tensors: (batch, nheads, seqlen, headdim)
q = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.float16)

# Use Flash Attention backend explicitly
with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
    output = F.scaled_dot_product_attention(q, k, v, is_causal=False)

# For causal/autoregressive attention
with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
    output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

### Variable Length Sequences

Use nested tensors for variable length sequences:

```python
seq_lens = [128, 256, 128]
q_list = [torch.randn(nheads, seq_len, headdim, device="cuda", dtype=torch.float16)
          for seq_len in seq_lens]

q = torch.nested.nested_tensor(q_list, layout=torch.jagged)
k = torch.nested.nested_tensor(k_list, layout=torch.jagged)
v = torch.nested.nested_tensor(v_list, layout=torch.jagged)

output = F.scaled_dot_product_attention(q, k, v)
```

## LLaMA-Factory Integration

For using Flash Attention with LLaMA-Factory QLoRA fine-tuning, see [llamafactory_flash_attention.md](llamafactory_flash_attention.md).

Quick start:
```bash
llamafactory-cli train --flash_attn sdpa ...
```

## Benchmark Results

On NVIDIA RTX 4090 with sequence length 1024:

| Backend | Time | Speedup |
|---------|------|---------|
| Flash Attention | 0.12 ms | 25x |
| Math Backend | 3.13 ms | 1x |

## Why Use PyTorch SDPA Instead of flash-attn Package?

- **No build issues**: Avoids CUDA version mismatch problems during compilation
- **Built into PyTorch**: No additional dependencies
- **Same performance**: Uses the same Flash Attention algorithm under the hood
- **Automatic fallback**: Falls back to other backends if Flash Attention isn't available

## License

MIT
