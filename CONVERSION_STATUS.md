# tokenkit JAX-to-PyTorch Conversion Status

This document tracks the conversion progress from the original JAX-based tokenkit implementation to PyTorch.

## Completed Components

- ✅ Project structure setup
- ✅ Core constants and utilities (`constants.py`)
- ✅ Model kinds implementation (`model_kinds.py`) 
- ✅ Byteify tokenizer implementation (`byteify.py`)
- ✅ Core utility functions in `utils.py`)
- ✅ Learning rate schedulers (`training/lr.py`)
- ✅ Optimizer wrapper (`training/opt.py`)
- ✅ LoRA implementation (`models/lora.py`)
- ✅ Hypernet transformer model (`models/hypernet/__init__.py`)
- ✅ Parameter handling utilities (`models/param.py`)
- ✅ Distributed training utilities (`models/sharding.py`)
- ✅ Checkpointing system (`training/checkpoint.py`)
- ✅ Tokenizer alignment collator (`training/collators/tokenizer_aligner.py`)
- ✅ Core alignment algorithm (`align.py`)
- ✅ Cross-tokenizer distillation script (`scripts/cross_tokenizer_distill.py`)
- ✅ Zero-shot tokenizer transfer script (`scripts/zett.py`)
- ✅ Evaluation script (`scripts/eval.py`)
- ✅ Lockstep evaluation script (`scripts/eval_lockstep.py`)
- ✅ Generation module (`eval/generate.py`)
- ✅ Basic loss functions implementation (`training/losses.py`)
- ✅ Side path loss implementation (`training/losses.py`)
- ✅ Baseline mined mapping loss (`training/losses.py`)

## Pending Components

- ✅ Missing utility scripts:
  - ✅ `compute_tokenizer_info.py` - Analyzes and reports tokenizer properties
  - ✅ `compute_mined_mapping.py` - Generates token mappings between different tokenizers
- ⬜️ Unit tests and integration tests
- ⬜️ Documentation updates
- ⬜️ Performance optimizations

## Remaining Work Tracking

### High Priority

1. **Missing Utility Scripts**
   - [x] `compute_tokenizer_info.py` implementation
   - [x] `compute_mined_mapping.py` implementation
   - [x] `baseline_utils.py` implementation with MinED and KL divergence functions

2. **Core Loss Function Verification**
   - [x] Test ALM loss function with different chunk types
   - [x] Verify side path loss implementation
   - [x] Test mined mapping loss

### Medium Priority

1. **Unit Tests**
   - [ ] Tests for alignment algorithms
   - [x] Tests for loss functions
   - [ ] Tests for tokenizer handling
   - [ ] Tests for hypernetwork model

2. **Integration Tests**
   - [ ] End-to-end test for cross-tokenizer distillation
   - [ ] End-to-end test for zero-shot tokenizer transfer
   - [ ] Output comparison between JAX and PyTorch implementations

3. **Documentation**
   - [ ] API documentation updates
   - [ ] Example notebooks
   - [ ] Configuration file documentation

### Low Priority

1. **Performance Optimizations**
   - [ ] Gradient checkpointing
   - [ ] Mixed precision training
   - [ ] More efficient operations for core functions
   - [ ] Distributed training improvements

## Testing Strategy

To ensure the conversion maintains the functionality of the original JAX implementation:

1. Create unit tests for each component to verify their behavior matches the JAX version
2. Implement simple end-to-end tests for:
   - Cross-tokenizer distillation with ALM
   - Zero-shot tokenizer transfer 
   - Evaluation on a small dataset
3. Compare outputs from both implementations on the same inputs to ensure they produce similar results

## Key Architectural Changes

1. **Model Architecture**:
   - Rewrote JAX Flax modules as PyTorch `nn.Module` classes
   - Changed functional programming approach to object-oriented approach

2. **Optimizers and Training**:
   - Replaced Optax optimizers with PyTorch optimizers
   - Implemented PyTorch-based learning rate schedulers
   - Created a more intuitive optimizer wrapper for different parameter groups

3. **Distributed Training**:
   - Replaced JAX's SPMD sharding with PyTorch's DistributedDataParallel
   - Implemented model and data parallelism utilities

4. **Data Processing**:
   - Adapted batch collation to work with PyTorch DataLoader
   - Ensured tensor types and dimensions are consistent with PyTorch conventions

5. **Loss Functions**:
   - Implemented all distillation losses in PyTorch
   - Preserved the same loss computation logic while adapting to PyTorch tensors

6. **Alignment Algorithm**:
   - Converted alignment algorithm to work with both NumPy arrays and PyTorch tensors
   - Added type hints and comprehensive documentation
   - Improved error handling and debugging output

7. **Evaluation and Generation**:
   - Implemented evaluation modules compatible with lm_eval framework
   - Created a PyTorch-based text generation module replacing JAX's while_loop approach
   - Adapted model ensembling for lockstep evaluation

## Next Steps

1. Complete missing utility scripts
2. Write comprehensive unit tests for critical components
3. Add more detailed documentation for PyTorch-specific modifications
4. Profile performance and implement optimizations
5. Test the complete pipeline with various models and tokenizers