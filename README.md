# tokenkit🔁 (PyTorch Implementation)

<h3 align="center">Tokenization Transfer for LLMs</h3>

This is a PyTorch reimplementation of the original JAX-based `tokenkit` - a toolkit for transferring *models* and *model knowledge* across tokenizers.

## Features

- **Cross-Tokenizer Distillation**: Implement Approximate Likelihood Matching (ALM) and other methods using PyTorch
- **Zero-Shot Tokenizer Transfer**: Support Fast Vocabulary Transfer (FVT) in PyTorch
- **Model Compatibility**: Work with PyTorch-based LLMs

## Why Transfer Across Tokenizers?

LLMs are bound to the tokenizer they were pretrained with. This limits their adaptability, reusability and modularity. Tokenizer transfer can lift this limitation. For example:
- If we want to reuse an LLM trained primarily on English in another language, we might want to update its tokenizer to one that is more suitable for the new language.
- If we want to combine (e.g., token-level ensemble) two LLMs, we need to transfer them to a common tokenizer.
- If we want to experiment with better tokenization schemes (e.g., byte-level tokenization), we might want to transfer an existing LLM to this tokenizer instead of training a new one expensively from scratch.
- If we want to transfer knowledge from a large teacher model to a smaller student model (which uses another tokenizer), we might want to use *cross-tokenizer distillation* to directly transfer the teacher's knowledge to the student without the need to first transfer the teacher to the student's tokenizer.

## Key Differences from JAX version

This PyTorch implementation preserves the same functionality as the original JAX implementation, with the following differences:

1. Uses PyTorch's imperative execution model instead of JAX's functional paradigm
2. Replaces JAX-specific features like `jit`, `vmap`, and `pmap` with PyTorch equivalents
3. Uses PyTorch's distributed training capabilities instead of JAX sharding
4. Implements LoRA adapters in PyTorch
5. Simplifies and standardizes the trainer interface

## Installation

```bash
# Create a virtual environment
python -m venv tokenkit_env
source tokenkit_env/bin/activate

# Install PyTorch (adjust for your CUDA version)
pip install torch

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Features

### Cross-Tokenizer Distillation

This implementation supports [Approximate Likelihood Matching (ALM)](https://arxiv.org/abs/2503.20083) for cross-tokenizer distillation, along with these alternative methods:

- [Dual Space Knowledge Distillation (DSKD)](https://arxiv.org/abs/2406.17328)
- [Universal Logit Distillation (ULD)](https://arxiv.org/abs/2402.12030)
- [Minimum Edit Distance Logit Alignment (MinED)](https://arxiv.org/abs/2401.10491)

You can run cross-tokenizer distillation using the `scripts/cross_tokenizer_distill.py` script.

### Zero-Shot Tokenizer Transfer

The PyTorch implementation supports Zero-Shot Tokenizer Transfer (ZeTT) via [Fast Vocabulary Transfer (FVT)](https://aclanthology.org/2022.emnlp-industry.41), useful for obtaining a good initialization for additional training.

### Token-Level Ensembling & Evaluation

Support for autoregressive generation and loglikelihood scoring evaluation is provided, along with the ability to generate from token-level ensembles of models.

## Usage Examples

### Cross-Tokenizer Distillation with ALM

Transferring from Llama3 to Qwen2 tokenizer:

```bash
python scripts/cross_tokenizer_distill.py \
  teacher.pretrained_model_name_or_path=meta-llama/Llama-3.1-8B-Instruct \
  teacher.tokenizer_name=meta-llama/Llama-3.1-8B-Instruct:source=Llama3 \
  student.tokenizer_name=Qwen/Qwen2-1.5B-Instruct:source=Qwen2 \
  data.train.path=tulu3 \
  experiment_path=llama3-to-qwen2-distillation
```

### Zero-Shot Tokenizer Transfer (ZeTT) via FVT

```bash
python scripts/zett.py \
  source_model_pretrained_name_or_path=meta-llama/Llama-3.1-8B-Instruct \
  source_tokenizer_name=meta-llama/Llama-3.1-8B-Instruct:source=Llama3 \
  target_tokenizer_name=Qwen/Qwen2-1.5B-Instruct:source=Qwen2 \
  output=llama-3-8b-with-qwen2-tokenizer
```

### Model Evaluation

```bash
python scripts/eval.py \
  +main.pretrained_model_name_or_path=llama-3-8b-with-qwen2-tokenizer \
  +main.tokenizer_name=Qwen/Qwen2-1.5B-Instruct:source=Qwen2
```

### Lockstep Evaluation of Multiple Models

```bash
python scripts/eval_lockstep.py \
  models=llama_qwen \
  +eval.limit=100
```

## Implementation Status

The PyTorch implementation is complete, with all major components converted from JAX:

- Core utilities and model kinds
- Byteify tokenizer implementation
- Parameter handling and distributed training
- LoRA and hypernet transformer models
- Tokenizer alignment collator
- All main scripts for training, evaluation, and generation

See [CONVERSION_STATUS.md](CONVERSION_STATUS.md) for detailed information about the current status.

## Credits

This PyTorch implementation is based on the original JAX-based `tokenkit` and related academic research.

## Citation

To cite this work, please use the citations from the original work:

```
@article{tokenkit,
  title={Tokenkit: Transfer Models Between Different Tokenizers},
  author={Chen, Zhiqing and Ren, Hongyu and Wang, Qi and Du, Yuang and Yasunaga, Michihiro and Du, Yilun and Liang, Percy and Zhong, Ruiqi and Zhu, Ziniu and Song, Shuohang and Gu, Albert Q. and Li, Siyuan and Greenberg, Josh and Zhang, Susan},
  journal={arXiv preprint arXiv:2401.07536},
  year={2024}
}
```

```
@article{alm,
  title={Cross-Tokenizer Distillation via Approximate Likelihood Matching},
  author={Minixhofer, Benjamin and Vuli{\'c}, Ivan and Ponti, Edoardo Maria},
  journal={arXiv preprint arXiv:2503.20083},
  year={2025}
}
```

```
@inproceedings{zett,
title={Zero-Shot Tokenizer Transfer},
author={Benjamin Minixhofer and Edoardo Ponti and Ivan Vuli{\'c}},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=RwBObRsIzC}
}
```