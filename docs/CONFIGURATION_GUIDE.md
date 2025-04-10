# tokenkit PyTorch Configuration Guide

This guide provides detailed information about the configuration system used in the PyTorch implementation of tokenkit. Understanding this system is essential for effectively using the library for cross-tokenizer knowledge transfer.

## Overview

tokenkit PyTorch uses [Hydra](https://hydra.cc/) for configuration management, providing a flexible and hierarchical approach to configuring experiments. The configuration system allows you to:

1. Define default configurations in YAML files
2. Override configurations via command line arguments
3. Compose configurations from multiple files
4. Use structured configs with validation

## Configuration Directory Structure

All configurations are stored in the `configs/` directory, organized by purpose:

```
configs/
├── compute_mined_mapping.yaml     # Token mapping computation
├── compute_tokenizer_info.yaml    # Tokenizer analysis
├── cross_tokenizer_distill.yaml   # Main distillation config
├── data/                          # Dataset configurations
│   └── tulu3.yaml                 # Tulu-v3 dataset config
├── eval.yaml                      # Evaluation config
├── eval/                          # Specialized eval configs
│   └── default.yaml               # Default evaluation settings
├── eval_lockstep.yaml             # Lockstep evaluation config
├── models/                        # Model configurations
│   ├── gemma_llama_qwen.yaml      # Multi-model config
│   └── llama_qwen.yaml            # Llama-to-Qwen config
├── optimizer/                     # Optimizer configurations
│   └── adamw.yaml                 # AdamW optimizer config
└── zett.yaml                      # ZeTT config
```

## Main Configuration Files

### Cross-Tokenizer Distillation

The `cross_tokenizer_distill.yaml` file contains the default configuration for cross-tokenizer distillation:

```yaml
# Model configurations
teacher:
  pretrained_model_name_or_path: google/gemma-2-2b-it  # Teacher model
  tokenizer_name: google/gemma-2-2b-it:source=Gemma2   # Teacher tokenizer

new:
  pretrained_model_name_or_path: Qwen/Qwen2.5-1.5B      # Student model
  tokenizer_name: Qwen/Qwen2.5-1.5B:source=Qwen2:target=Gemma2  # Student tokenizer

# Training parameters
name: gemma2_to_qwen2  # Experiment name
train_model_mode: lora  # Options: lora, full, none
train_embeddings: true  # Whether to train embeddings
dtype: float16  # Precision for training
prefix_steps: 100  # Number of steps for prefix training
prefix_lr: 1e-5  # Learning rate for prefix training
seed: 42  # Random seed for reproducibility

# Loss configuration
losses:  # List of losses to use for distillation
  - alm_unconstrained  # ALM with unconstrained alignment
  - alm_space          # ALM with space-constrained alignment
  - clm                # Standard causal language modeling
alm_diff_fn: abs       # Difference function for ALM (abs, binary_ce, etc.)
distill_chunk_sizes: [1]  # Chunk sizes for distillation
distill_main_path_numerator: chunk_count  # Normalization numerator
distill_main_path_denominator: chunk_count  # Normalization denominator

# Tokenizer parameters
max_length_teacher: 1024  # Maximum sequence length for teacher
max_length_new: 1024      # Maximum sequence length for student
pad_to_multiple_of: 64    # Pad vocabulary sizes to multiple of this
special_tokens_mode: identity  # How to handle special tokens
use_chat_template: false  # Whether to use chat templates

# Training hyperparameters
steps: 1000  # Total training steps
batch_size: 64  # Batch size
gradient_accumulation_steps: 1  # Gradient accumulation steps
warmup_steps: 100  # Learning rate warmup steps
eval_interval: 100  # Evaluation interval in steps
save_interval: 200  # Checkpoint saving interval
logging_interval: 10  # Logging interval

# Optimizer configuration
optimizer:
  name: adamw  # Optimizer name
  learning_rate: 1e-4  # Learning rate
  weight_decay: 0.01   # Weight decay
  beta1: 0.9  # Beta1 for Adam
  beta2: 0.999  # Beta2 for Adam
  epsilon: 1e-8  # Epsilon for Adam
  lr_scheduler_type: cosine  # LR scheduler (cosine, linear)
  lr_min_ratio: 0.1  # Minimum LR ratio for cosine schedule

# Data configuration
data:
  dataset_name: Anthropic/tulu-v3-sft-mixtures  # Dataset name
  dataset_config_name: default  # Dataset config
  split: train  # Dataset split
  batch_size: 32  # Dataset batch size (used for counting tokens, etc.)
  num_workers: 4  # Number of dataloader workers

# Output configuration
output_dir: outputs/gemma2_to_qwen2  # Output directory
```

### Zero-Shot Tokenizer Transfer (ZeTT)

The `zett.yaml` file configures Zero-Shot Tokenizer Transfer:

```yaml
# Model configurations
model:
  pretrained_model_name_or_path: meta-llama/Llama-3-8B  # Source model
  tokenizer_name: meta-llama/Llama-3-8B:source=Llama3   # Source tokenizer

# Target tokenizer
target_tokenizer_name: Qwen/Qwen2.5-1.5B:source=Qwen2:target=Llama3

# FVT parameters
extend_output_embeddings: true  # Whether to extend output embeddings
clone_all_layers: false  # Clone all layers or just embeddings
combine_output_embed_with_input: true  # Combine output embeds with input
convert_bos_eos: true  # Convert BOS/EOS tokens
convert_by_special: true  # Convert by special tokens
convert_by_matching: true  # Convert by matching

# Output configuration
output: outputs/llama3_to_qwen2  # Output directory
```

### Evaluation Configuration

The `eval.yaml` file configures model evaluation:

```yaml
# Model to evaluate
model:
  pretrained_model_name_or_path: outputs/gemma2_to_qwen2  # Model path
  tokenizer_name: null  # Tokenizer name (defaults to model's tokenizer)

# Evaluation parameters
tasks:  # List of evaluation tasks
  - wikitext
  - piqa
  - hellaswag
  - lambada_openai
max_samples: 100  # Maximum number of samples to evaluate
temperature: 0.0  # Sampling temperature
num_beams: 1      # Number of beams for beam search
max_new_tokens: 100  # Maximum new tokens to generate
device: cuda      # Device to use (cuda, cpu)
batch_size: 16    # Batch size for evaluation
max_length: 1024  # Maximum sequence length

# Output configuration
output: outputs/gemma2_to_qwen2/eval  # Output directory
```

### Tokenizer Analysis

The `compute_tokenizer_info.yaml` file configures tokenizer analysis:

```yaml
# Tokenizer configurations
teacher_tokenizer_name: google/gemma-2-2b-it:source=Gemma2  # Teacher tokenizer
target_tokenizer_name: Qwen/Qwen2.5-1.5B:source=Qwen2:target=Gemma2  # Target tokenizer

# Sampling parameters
teacher_subsample_percent: 0.1  # Percentage of data to use for teacher
student_subsample_percent: 0.1  # Percentage of data to use for student
additive_smoothing_constant: 1e-10  # Smoothing constant for probabilities

# Data configuration
data:
  dataset_name: Anthropic/tulu-v3-sft-mixtures  # Dataset name
  dataset_config_name: default  # Dataset config
  split: train  # Dataset split
  batch_size: 32  # Batch size for processing
  num_workers: 4  # Number of workers

# Output configuration
output: outputs/tokenizer_data/gemma2_to_qwen2  # Output directory
```

### Token Mapping

The `compute_mined_mapping.yaml` file configures token mapping computation:

```yaml
# Tokenizer configurations
teacher_tokenizer_name: google/gemma-2-2b-it:source=Gemma2  # Teacher tokenizer
target_tokenizer_name: Qwen/Qwen2.5-1.5B:source=Qwen2:target=Gemma2  # Target tokenizer

# Computation parameters
num_workers: 8  # Number of workers for parallel processing

# Output configuration
output: outputs/tokenizer_data/gemma2_to_qwen2_mined  # Output directory
```

## Using Configurations

### Basic Usage

To run a script with the default configuration:

```bash
python scripts/cross_tokenizer_distill.py
```

### Overriding Configuration Values

You can override any configuration value via command line:

```bash
python scripts/cross_tokenizer_distill.py \
    teacher.pretrained_model_name_or_path=google/gemma-2-7b-it \
    train_model_mode=full \
    data.batch_size=64 \
    output_dir=outputs/gemma2_7b_to_qwen2
```

### Using Multiple Configuration Files

You can compose multiple configuration files:

```bash
python scripts/cross_tokenizer_distill.py \
    --config-name=cross_tokenizer_distill \
    --config-path=configs \
    +models=llama_qwen \
    +optimizer=adamw \
    +data=tulu3
```

This loads the base `cross_tokenizer_distill.yaml` configuration and then applies configurations from `models/llama_qwen.yaml`, `optimizer/adamw.yaml`, and `data/tulu3.yaml`.

### Multirun Sweeps

Hydra supports parameter sweeps for experiments:

```bash
python scripts/cross_tokenizer_distill.py \
    -m optimizer.learning_rate=1e-4,5e-5,1e-5 \
    losses=clm,alm_unconstrained,alm_space
```

This runs 9 experiments with all combinations of the specified learning rates and loss functions.

## Special Configuration Features

### Tokenizer Name Format

tokenkit uses a special format for tokenizer names to specify the tokenizer family and target:

```
huggingface/model-name:source=SourceTokenizer:target=TargetTokenizer
```

For example:
```
Qwen/Qwen2.5-1.5B:source=Qwen2:target=Gemma2
```

- `source`: Indicates the tokenizer's original model family (required)
- `target`: Indicates the target model family for transfer (optional)

This format helps tokenkit identify the correct tokenizer processing and special token handling.

### Loss Function Configuration

tokenkit supports multiple distillation methods, configured via the `losses` parameter:

```yaml
losses:
  - alm_unconstrained  # ALM with unconstrained alignment
  - alm_space          # ALM with space-constrained alignment
  - alm_unbiased       # ALM with bias-constrained alignment
  - alm_latents        # ALM for hidden states
  - alm_side_path      # ALM for specific token mappings
  - clm                # Standard causal language modeling
  - baseline_dskd      # Dual-Space Knowledge Distillation
  - baseline_uld       # Universal Logit Distillation
  - baseline_minED     # Minimum Edit Distance
  - baseline_mined     # Mined mapping-based distillation
```

Additional parameters for ALM:

```yaml
# ALM parameters
alm_diff_fn: abs          # Difference function (abs, binary_ce, reverse_binary_kl, etc.)
alm_mode: append_space    # ALM mode (append_space, eos_as_space, merge_by_space_prob)
distill_chunk_sizes: [1]  # Chunk sizes for distillation (can be multiple: [1, 2, 4])
```

### LoRA Configuration

For Low-Rank Adaptation (LoRA) fine-tuning:

```yaml
# LoRA parameters
train_model_mode: lora    # Use LoRA for training
model_lora_alpha: 16      # LoRA alpha parameter
model_lora_r: 8           # LoRA rank
model_lora_targets:       # Layers to apply LoRA to
  - "q_proj"
  - "k_proj"
  - "v_proj"
  - "o_proj"
```

### Distributed Training Configuration

For distributed training:

```yaml
# Distributed training parameters
use_distributed: true           # Enable distributed training
distributed_backend: nccl       # Backend (nccl, gloo)
n_data_parallel: 8              # Number of data parallel devices
n_model_parallel: 1             # Number of model parallel devices
```

## Configuration Files Reference

For each script, here are the primary configuration fields:

### cross_tokenizer_distill.py

| Field | Description | Default |
|-------|-------------|---------|
| `teacher.pretrained_model_name_or_path` | Teacher model path or name | - |
| `teacher.tokenizer_name` | Teacher tokenizer name | - |
| `new.pretrained_model_name_or_path` | Student model path or name | - |
| `new.tokenizer_name` | Student tokenizer name | - |
| `train_model_mode` | Training mode (lora, full, none) | lora |
| `losses` | List of loss functions to use | - |
| `data.dataset_name` | Dataset name | - |
| `output_dir` | Output directory | - |

### zett.py

| Field | Description | Default |
|-------|-------------|---------|
| `model.pretrained_model_name_or_path` | Source model path or name | - |
| `model.tokenizer_name` | Source tokenizer name | - |
| `target_tokenizer_name` | Target tokenizer name | - |
| `output` | Output directory | - |

### compute_tokenizer_info.py

| Field | Description | Default |
|-------|-------------|---------|
| `teacher_tokenizer_name` | Teacher tokenizer name | - |
| `target_tokenizer_name` | Target tokenizer name | - |
| `data.dataset_name` | Dataset name for token statistics | - |
| `output` | Output directory | - |

### compute_mined_mapping.py

| Field | Description | Default |
|-------|-------------|---------|
| `teacher_tokenizer_name` | Teacher tokenizer name | - |
| `target_tokenizer_name` | Target tokenizer name | - |
| `num_workers` | Number of worker processes | 8 |
| `output` | Output directory | - |

### eval.py

| Field | Description | Default |
|-------|-------------|---------|
| `model.pretrained_model_name_or_path` | Model path or name | - |
| `model.tokenizer_name` | Tokenizer name | null |
| `tasks` | Evaluation tasks | - |
| `output` | Output directory | - |

## Debugging Configurations

To debug configurations without running the actual script:

```bash
python scripts/cross_tokenizer_distill.py --cfg=job
```

This prints the configuration that would be used for the job.

For more detailed information:

```bash
python scripts/cross_tokenizer_distill.py --cfg=job --resolve
```

This shows the fully resolved configuration with all defaults and overrides applied.

## Best Practices

1. **Create Task-Specific Configs**: Create specialized configurations for common tasks and compose them as needed.

2. **Version Your Configurations**: Store important configurations in your experiment directory for reproducibility.

3. **Use Environment Variables**: For sensitive information or machine-specific settings, use environment variable interpolation:

   ```yaml
   output_dir: ${oc.env:OUTPUT_DIR,/tmp}/experiment
   ```

4. **Document Your Configurations**: Add comments to explain key parameters and their effects.

5. **Group Related Parameters**: Organize parameters into logical groups for better readability.

6. **Use Config Groups**: For parameters that always go together, use Hydra's config groups.

7. **Validate Early**: Use the `--cfg=job` flag to validate configurations before running long experiments.

## Common Issues and Solutions

### Unknown Field Error

```
Error: In primary config: Validation error for tokenizer_name: tokenizer_name is not a valid field
```

**Solution**: Check the configuration structure. The field might be nested under a parent group, e.g., `teacher.tokenizer_name` instead of `tokenizer_name`.

### Type Mismatch Error

```
Error: In primary config: Validation error for batch_size: Value '64' could not be converted to Integer
```

**Solution**: Ensure you're providing the correct type for each parameter. For lists, use comma-separated values without spaces.

### Missing Required Field

```
Error: Missing mandatory value for model.pretrained_model_name_or_path
```

**Solution**: Provide all required fields either in the configuration files or via command line.

### Config Path Not Found

```
Error: Config not found: 'models/custom_model'
```

**Solution**: Check the path to your configuration file. It should be relative to the config directory.

## Advanced Configuration Tips

### Creating Defaults List

For complex configurations, use Hydra's `defaults` list to compose multiple configs:

```yaml
defaults:
  - _self_
  - models: llama_qwen
  - optimizer: adamw
  - data: tulu3
  - overrides: base
```

### Conditional Configuration

Use `hydra-zen` for more complex configuration logic:

```python
from hydra_zen import store, zen

# Create a config store
config_store = store()

# Register model configs based on conditions
if use_lora:
    config_store(group="model", name="lora", node=LoRAConfig)
else:
    config_store(group="model", name="full", node=FullFinetuneConfig)

# Run with dynamic config
@zen(config_store)
def run_experiment(cfg):
    # ...
```

### Configuration Templates

Create template configurations for common experiments:

```yaml
# templates/gemma_to_qwen.yaml
teacher:
  pretrained_model_name_or_path: google/gemma-2-${size}-it
  tokenizer_name: google/gemma-2-${size}-it:source=Gemma2

new:
  pretrained_model_name_or_path: Qwen/Qwen2.5-${target_size}
  tokenizer_name: Qwen/Qwen2.5-${target_size}:source=Qwen2:target=Gemma2

# Use with:
# python scripts/cross_tokenizer_distill.py --config-name=templates/gemma_to_qwen size=2b target_size=1.5B
```

## Conclusion

This guide covers the key aspects of the configuration system in tokenkit PyTorch. By understanding and leveraging this system, you can efficiently perform cross-tokenizer knowledge transfer experiments with minimal code changes.

For further details, refer to:
- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [tokenkit README.md](/README.md)
- [Example scripts in the examples/ directory](/examples/)