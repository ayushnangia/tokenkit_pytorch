# Distillation Methods in tokenkit PyTorch

This guide explains the various distillation methods available in tokenkit PyTorch for cross-tokenizer knowledge transfer. These methods allow you to transfer knowledge from a model using one tokenizer to a model using another tokenizer.

## Overview of Distillation Methods

tokenkit supports the following distillation methods:

1. **Approximate Likelihood Matching (ALM)** - The core method developed for tokenkit
2. **Side Path Distillation** - Token-specific alignment approach
3. **Mined Mapping Distillation** - Based on minimum edit distance token mappings
4. **Baseline Methods** - Including DSKD, ULD, and MinED

## 1. Approximate Likelihood Matching (ALM)

ALM is the primary method for cross-tokenizer distillation in tokenkit. It works by aligning token probabilities between tokenizers despite their different vocabularies.

### ALM Variants

- **alm_unconstrained**: Unconstrained alignment between tokenizers
- **alm_space**: Space-constrained alignment (uses spaces as anchor points)
- **alm_unbiased**: Bias-constrained alignment using tokenizer statistics
- **alm_greedy**: Greedy alignment strategy
- **alm_latents**: Alignment of hidden states rather than probabilities

### ALM Configuration

```yaml
losses: [alm_unconstrained]  # The ALM variant to use
alm_diff_fn: binary_ce       # Difference function (binary_ce, abs, reverse_binary_kl)
alm_mode: append_space       # Token handling mode
tokenizer_pair_data_path: outputs/tokenizer_data/llama3_to_qwen2  # Pre-computed token statistics
```

### ALM Example

Example of using ALM for transferring from Llama3 to Qwen2:

```bash
python scripts/cross_tokenizer_distill.py \
    losses=[alm_unbiased] \
    alm_diff_fn=binary_ce \
    +student.pretrained_model_name_or_path="benjamin/Llama-3.2-3B-Instruct-flax" \
    +student.tokenizer_name='meta-llama/Llama-3.2-3B-Instruct:source=Llama3' \
    +target_tokenizer_name='Qwen/Qwen2.5-1.5B:source=Qwen2:target=Llama3' \
    tokenizer_pair_data_path='outputs/tokenizer_data/llama3_to_qwen2' \
    output=outputs/llama3_to_qwen2_alm
```

## 2. Side Path Distillation

Side path distillation extends ALM by focusing on specific token mappings. It uses an additional "side path" in the model that handles token-specific relationships.

### Side Path Configuration

```yaml
losses: [alm_side_path]
side_path_mapping_mode: bias_threshold  # How to select tokens for side path
side_path_distance_fn: kl              # Distance function for side path
tokenizer_pair_bias_threshold_side_path: 1e-3  # Threshold for token selection
```

### Side Path Example

```bash
python scripts/cross_tokenizer_distill.py \
    losses=[alm_side_path] \
    side_path_mapping_mode=bias_threshold \
    side_path_distance_fn=kl \
    tokenizer_pair_bias_threshold_side_path=1e-3 \
    +student.pretrained_model_name_or_path="benjamin/Llama-3.2-3B-Instruct-flax" \
    +student.tokenizer_name='meta-llama/Llama-3.2-3B-Instruct:source=Llama3' \
    +target_tokenizer_name='Qwen/Qwen2.5-1.5B:source=Qwen2:target=Llama3' \
    tokenizer_pair_data_path='outputs/tokenizer_data/llama3_to_qwen2' \
    output=outputs/llama3_to_qwen2_side_path
```

## 3. Mined Mapping Distillation

Mined mapping distillation uses pre-computed token mappings based on minimum edit distance. This approach is useful when you want explicit control over token relationships.

### Mined Mapping Prerequisites

Before using mined mapping, you need to compute the mapping:

```bash
python scripts/compute_mined_mapping.py \
    teacher_tokenizer_name='meta-llama/Llama-3.2-3B-Instruct:source=Llama3' \
    target_tokenizer_name='Qwen/Qwen2.5-1.5B:source=Qwen2:target=Llama3' \
    output='outputs/tokenizer_data/llama3_to_qwen2_mined'
```

### Mined Mapping Configuration

```yaml
losses: [baseline_mined]
baseline:
  divergence: srkl  # Divergence function (srkl, kl, js)
  teacher_temperature: 1.0  # Temperature for teacher logits
```

### Mined Mapping Example

```bash
python scripts/cross_tokenizer_distill.py \
    losses=[baseline_mined] \
    baseline.divergence=srkl \
    baseline.teacher_temperature=1.0 \
    +student.pretrained_model_name_or_path="benjamin/Llama-3.2-3B-Instruct-flax" \
    +student.tokenizer_name='meta-llama/Llama-3.2-3B-Instruct:source=Llama3' \
    +target_tokenizer_name='Qwen/Qwen2.5-1.5B:source=Qwen2:target=Llama3' \
    tokenizer_pair_data_path='outputs/tokenizer_data/llama3_to_qwen2' \
    mined_mapping_path='outputs/tokenizer_data/llama3_to_qwen2_mined/mapping.pt' \
    output=outputs/llama3_to_qwen2_mined
```

## 4. Baseline Methods

tokenkit implements several baseline methods from the literature for comparison:

### DSKD (Dual-Space Knowledge Distillation)

```yaml
losses: [baseline_dskd]
baseline:
  dskd_use_causal_attention_mask: true  # Use causal mask (for auto-regressive models)
  teacher_temperature: 1.0  # Temperature for teacher logits
```

### ULD (Universal Logit Distillation)

```yaml
losses: [baseline_uld]
baseline:
  kd_rate: 0.5  # Weight for knowledge distillation
  kd_temp: 2.0  # Temperature for distillation
```

### MinED (Minimum Edit Distance)

```yaml
losses: [baseline_minED]
baseline:
  skew_lambda: 0.1  # Parameter for skew calculation
```

### Baseline Example

```bash
python scripts/cross_tokenizer_distill.py \
    losses=[baseline_dskd] \
    baseline.dskd_use_causal_attention_mask=true \
    baseline.teacher_temperature=1.0 \
    +student.pretrained_model_name_or_path="benjamin/Llama-3.2-3B-Instruct-flax" \
    +student.tokenizer_name='meta-llama/Llama-3.2-3B-Instruct:source=Llama3' \
    +target_tokenizer_name='Qwen/Qwen2.5-1.5B:source=Qwen2:target=Llama3' \
    tokenizer_pair_data_path='outputs/tokenizer_data/llama3_to_qwen2' \
    output=outputs/llama3_to_qwen2_dskd
```

## 5. Combining Multiple Loss Functions

tokenkit supports combining multiple loss functions for distillation. This can often improve results as different methods capture different aspects of knowledge.

### Multi-Loss Configuration

```yaml
losses: [alm_unbiased, clm]  # Combine ALM with standard language modeling
loss_weights: [0.8, 0.2]     # Weights for each loss
```

### Multi-Loss Example

```bash
python scripts/cross_tokenizer_distill.py \
    losses=[alm_unbiased,clm] \
    loss_weights=[0.8,0.2] \
    alm_diff_fn=binary_ce \
    +student.pretrained_model_name_or_path="benjamin/Llama-3.2-3B-Instruct-flax" \
    +student.tokenizer_name='meta-llama/Llama-3.2-3B-Instruct:source=Llama3' \
    +target_tokenizer_name='Qwen/Qwen2.5-1.5B:source=Qwen2:target=Llama3' \
    tokenizer_pair_data_path='outputs/tokenizer_data/llama3_to_qwen2' \
    output=outputs/llama3_to_qwen2_multi
```

## Workflow for Cross-Tokenizer Distillation

Here's a complete workflow for cross-tokenizer distillation:

1. **Compute tokenizer statistics**:
   ```bash
   python scripts/compute_tokenizer_info.py \
       teacher_tokenizer_name='meta-llama/Llama-3.2-3B-Instruct:source=Llama3' \
       target_tokenizer_name='Qwen/Qwen2.5-1.5B:source=Qwen2:target=Llama3' \
       output='outputs/tokenizer_data/llama3_to_qwen2'
   ```

2. **Optional: Compute mined mapping** (if using mined mapping distillation):
   ```bash
   python scripts/compute_mined_mapping.py \
       teacher_tokenizer_name='meta-llama/Llama-3.2-3B-Instruct:source=Llama3' \
       target_tokenizer_name='Qwen/Qwen2.5-1.5B:source=Qwen2:target=Llama3' \
       output='outputs/tokenizer_data/llama3_to_qwen2_mined'
   ```

3. **Run distillation**:
   ```bash
   python scripts/cross_tokenizer_distill.py \
       losses=[alm_unbiased] \
       alm_diff_fn=binary_ce \
       +student.pretrained_model_name_or_path="benjamin/Llama-3.2-3B-Instruct-flax" \
       +student.tokenizer_name='meta-llama/Llama-3.2-3B-Instruct:source=Llama3' \
       +target_tokenizer_name='Qwen/Qwen2.5-1.5B:source=Qwen2:target=Llama3' \
       tokenizer_pair_data_path='outputs/tokenizer_data/llama3_to_qwen2' \
       output=outputs/llama3_to_qwen2
   ```

4. **Evaluate the model**:
   ```bash
   python scripts/eval.py \
       model.pretrained_model_name_or_path=outputs/llama3_to_qwen2 \
       tasks=[arc_easy,arc_challenge,piqa,hellaswag,boolq] \
       output=outputs/llama3_to_qwen2/eval
   ```

## Performance Considerations

Different distillation methods have different computational requirements:

- **ALM**: Moderate computation, good all-around performance
- **Side Path**: Higher computation due to additional network components
- **Mined Mapping**: Requires pre-computation of mappings but efficient during training
- **DSKD/ULD**: Generally more computationally efficient but may have lower transfer quality

For large models, consider using:
- Lower precision (bfloat16)
- Gradient accumulation
- LoRA adaptation instead of full fine-tuning

## Tips for Choosing Distillation Methods

1. **ALM Variants**: Start with `alm_unbiased` as it generally performs well across different tokenizer pairs.

2. **Token Similarity**: If the tokenizers are very different, combine ALM with mined mapping.

3. **Model Size**: For larger models, `alm_unconstrained` with LoRA is a good starting point.

4. **Language Specificity**: For specific languages or domains, `alm_side_path` often performs better.

5. **Multiple Languages**: For multilingual settings, combining ALM with CLM (`[alm_unbiased, clm]`) tends to work well.

## Conclusion

tokenkit PyTorch provides a comprehensive set of distillation methods for cross-tokenizer knowledge transfer. By understanding the different methods and their configurations, you can effectively transfer knowledge between models with different tokenizers.

For more information, refer to:
- [Configuration Guide](CONFIGURATION_GUIDE.md)
- [Example scripts in the examples/ directory](/examples/)
- [Research paper](https://arxiv.org/abs/2302.03169) describing the ALM method