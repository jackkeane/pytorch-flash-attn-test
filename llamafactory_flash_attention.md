# LLaMA-Factory Flash Attention Configuration

## Flash Attention Options

In LLaMA-Factory, you can use PyTorch's native SDPA (which includes Flash Attention) by setting the `--flash_attn` parameter.

```bash
# Use PyTorch's native SDPA (recommended for your setup)
llamafactory-cli train \
    --flash_attn sdpa \
    ... other options ...
```

The `--flash_attn` parameter accepts these values:

| Value | Description |
|-------|-------------|
| `disabled` | Use standard attention (slow, high memory) |
| `sdpa` | **Use PyTorch's native SDPA** (includes Flash Attention backend) |
| `fa2` | Use flash-attn package (requires `flash-attn` installed) |

## Example QLoRA Training Command

```bash
llamafactory-cli train \
    --stage sft \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --flash_attn sdpa \
    --quantization_bit 4 \
    --quantization_method bitsandbytes \
    --lora_rank 8 \
    --lora_target q_proj,v_proj \
    --dataset alpaca_en \
    --template llama2 \
    --output_dir ./output \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --bf16 true
```

## Using YAML Config

If you're using a YAML config file:

```yaml
### model
model_name_or_path: meta-llama/Llama-2-7b-hf
flash_attn: sdpa  # Use PyTorch native SDPA

### method
stage: sft
quantization_bit: 4
quantization_method: bitsandbytes
lora_rank: 8
lora_target: q_proj,v_proj

### dataset
dataset: alpaca_en
template: llama2

### output
output_dir: ./output

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
num_train_epochs: 3.0
bf16: true
```

Then run with:

```bash
llamafactory-cli train config.yaml
```

## Verify It's Working

You can verify SDPA is being used by checking the training logs - it should not show any warnings about Flash Attention being unavailable.
