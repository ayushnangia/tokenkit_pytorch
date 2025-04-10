# Advanced Usage Guide for tokenkit PyTorch

This guide covers advanced usage patterns and techniques for tokenkit PyTorch that go beyond the basic workflows.

## Table of Contents

1. [Custom Tokenizer Handling](#custom-tokenizer-handling)
2. [Mixed Precision Training](#mixed-precision-training)
3. [Distributed Training](#distributed-training)
4. [Custom Loss Functions](#custom-loss-functions)
5. [Hypernets for Token Mapping](#hypernets-for-token-mapping)
6. [LoRA Configuration](#lora-configuration)
7. [Training Optimizations](#training-optimizations)
8. [Multi-Stage Training](#multi-stage-training)
9. [Exporting and Deployment](#exporting-and-deployment)

## Custom Tokenizer Handling

tokenkit allows for advanced tokenizer handling beyond the standard HuggingFace tokenizers.

### Custom Tokenizer Formats

You can use custom tokenizers by specifying a path to a tokenizer directory:

```yaml
target_tokenizer: /path/to/custom/tokenizer
```

### Byte-Level Tokenizers

To use byte-level tokenization:

```yaml
target_tokenizer: byte
```

This enables transferring to a byte-level tokenizer for improved language universality.

### Adding Special Tokens

You can add additional special tokens to the target tokenizer:

```yaml
tokens_to_add:
  - "<extra_token_0>"
  - "<extra_token_1>"
```

### Special Token Handling

Control how special tokens are handled during distillation:

```yaml
special_tokens_mode: identity  # Options: identity, ignore, ignore_pad
```

- `identity`: Map special tokens directly between tokenizers
- `ignore`: Exclude special tokens from distillation
- `ignore_pad`: Only exclude padding tokens

## Mixed Precision Training

Using mixed precision can significantly speed up training:

```yaml
dtype: bfloat16  # Options: float32, float16, bfloat16
```

For certain hardware-specific optimizations:

```bash
python scripts/cross_tokenizer_distill.py \
    dtype=bfloat16 \
    optimizer.use_fused_adam=true \
    use_flash_attention=true
```

## Distributed Training

tokenkit supports various distributed training configurations.

### Multi-GPU Data Parallelism

For data parallel training across multiple GPUs:

```yaml
n_data_parallel: 8
n_model_parallel: 1
```

### Model Parallelism

For very large models that don't fit on a single GPU:

```yaml
n_data_parallel: 1
n_model_parallel: 8
```

### Hybrid Parallelism

For the best performance with large models and datasets:

```yaml
n_data_parallel: 4
n_model_parallel: 2
```

### Gradient Accumulation

To simulate larger batch sizes:

```yaml
optimizer.grad_acc_steps: 8
```

## Custom Loss Functions

You can create custom loss functions by extending the base implementation.

### Creating a Custom Loss

1. Define your loss in `tokenkit/training/losses.py`:

```python
def compute_custom_loss(
    teacher_logits,
    student_logits,
    attention_mask,
    tokenizer_pair_data=None,
    **kwargs
):
    # Your custom loss implementation
    # ...
    return loss, metrics
```

2. Register your loss in the configuration:

```yaml
losses: [custom_loss]
```

### Loss Weighting and Scheduling

For more control over multiple losses:

```yaml
losses: [alm_unbiased, clm]
loss_weights: [0.8, 0.2]  # Static weights
# OR
loss_weight_mode: "uncertainty"  # Dynamic weighting
uncertainty_s_init: 0.2  # Initial uncertainty parameter
# OR
loss_schedules:  # Schedule weights over training steps
  alm_unbiased: "linear(0,1,1000)"  # Linear increase from 0 to 1 over 1000 steps
  clm: "constant(1)"  # Constant weight of 1
```

## Hypernets for Token Mapping

tokenkit supports hypernetworks for more advanced token mapping:

```yaml
hypernet:
  architecture: transformer  # Options: linear, mlp, transformer
  num_layers: 3  # Number of layers
  residual: true  # Use residual connections
  residual_alpha: 0.8  # Residual connection strength
  use_attention: true  # Use self-attention
```

Example usage:

```bash
python scripts/cross_tokenizer_distill.py \
    hypernet.architecture=transformer \
    hypernet.num_layers=2 \
    hypernet.use_attention=true \
    latents_do_project=true
```

## LoRA Configuration

Fine-grained control over LoRA adaptation:

```yaml
train_model_mode: lora
model_lora_rank: 16
model_lora_alpha: 32
model_lora_dropout: 0.05
model_lora_targets:
  - "q_proj"
  - "k_proj"
  - "v_proj"
  - "o_proj"
  - "gate_proj"
model_lora_modules_to_save:
  - "embed_tokens"
  - "lm_head"
```

For even more selective training:

```yaml
model_lora_target_regex:
  - ".*attention.*_proj"  # Target all attention projections
model_lora_layers:
  - "0-8"  # Only layers 0 through 8
  - "20-32"  # And layers 20 through 32
```

## Training Optimizations

### Gradient Checkpointing

To reduce memory usage at the cost of computation time:

```yaml
gradient_checkpointing: true
```

### Optimizer Configuration

Advanced optimizer settings:

```yaml
optimizer:
  name: adamw
  learning_rate: 3e-5
  weight_decay: 0.01
  max_grad_norm: 1.0
  lr_scheduler_type: cosine
  warmup_steps: 1000
  param_groups:
    - pattern: ".*embeddings.*"
      lr_scale: 0.1
    - pattern: ".*lm_head.*"
      lr_scale: 2.0
```

### Custom Learning Rate Schedules

```yaml
optimizer:
  schedule: "linear_warmup_cosine_decay"
  lr_min_ratio: 0.1
  warmup_steps: 1000
```

## Multi-Stage Training

For more complex training regimes, you can use multi-stage training:

### Embeddings-First Training

First train only the embeddings:

```bash
python scripts/cross_tokenizer_distill.py \
    train_embeddings=true \
    train_model_mode=none \
    steps=1000 \
    output_dir=outputs/stage1_embeddings
```

Then train the rest with the trained embeddings:

```bash
python scripts/cross_tokenizer_distill.py \
    train_embeddings=false \
    train_model_mode=lora \
    +student.pretrained_model_name_or_path=outputs/stage1_embeddings \
    steps=5000 \
    output_dir=outputs/stage2_full
```

### Prefix Layer Training

Train just the prefix layers first:

```yaml
n_prefix_layers: 4
prefix_steps: 1000
prefix_lr: 1e-4
prefix_trainable: "non_overlapping_embeddings"
```

## Exporting and Deployment

### Exporting to Hugging Face Format

To export your model in a format compatible with Hugging Face:

```bash
python scripts/export_model.py \
    model_path=outputs/my_model \
    output_path=exported/my_model \
    format=huggingface
```

### Merging LoRA Weights

If you trained with LoRA, you can merge the weights into the base model:

```bash
python scripts/merge_lora.py \
    base_model=meta-llama/Llama-3.2-3B-Instruct \
    lora_model=outputs/my_lora_model \
    output_path=outputs/merged_model
```

### Optimizing for Inference

You can optimize the model for inference:

```bash
python scripts/optimize_for_inference.py \
    model_path=outputs/my_model \
    output_path=outputs/optimized_model \
    quantization=int8 \
    device=cuda
```

## Integration with Other Frameworks

### Integrating with Transformers

You can use tokenkit models with standard Transformers pipelines:

```python
from transformers import pipeline

# Load a tokenkit-trained model
pipe = pipeline("text-generation", model="path/to/tokenkit/model")
result = pipe("Hello, world!", max_length=100)
print(result[0]['generated_text'])
```

### ONNX Export

Export to ONNX format for deployment:

```bash
python scripts/export_to_onnx.py \
    model_path=outputs/my_model \
    output_path=outputs/my_model.onnx \
    dynamic_axes=true
```

## Conclusion

This advanced guide demonstrates the flexibility and power of tokenkit PyTorch for cross-tokenizer knowledge transfer. By leveraging these advanced features, you can customize your training process for specific use cases and achieve better performance in various scenarios.

For more information, refer to:
- [Configuration Guide](CONFIGURATION_GUIDE.md)
- [Distillation Methods](DISTILLATION_METHODS.md)
- [Example scripts in the examples/ directory](/examples/)