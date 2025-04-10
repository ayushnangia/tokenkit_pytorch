# tokenkit PyTorch Conversion Plan

This document outlines the step-by-step plan for completing the conversion of tokenkit from JAX to PyTorch.

## Phase 1: Core Components (Completed)

- ✅ Project structure setup
- ✅ Core utilities and constants implementation
- ✅ Model kinds and tokenizer components
- ✅ Basic utilities and LoRA implementation
- ✅ Loss functions in PyTorch
- ✅ Optimizer and learning rate scheduler implementation
- ✅ Hypernet transformer model architecture

## Phase 2: Model & Training Infrastructure

1. Parameter handling utilities (`models/param.py`)
   - Implement clean PyTorch state dict management
   - Replace JAX tree manipulation with PyTorch-friendly alternatives

2. Replace sharding with distributed training (`models/sharding.py`)
   - Implement PyTorch DistributedDataParallel wrapper
   - Create device placement utilities

3. Checkpointing system (`training/checkpoint.py`)
   - Create PyTorch model checkpointing
   - Implement state loading and saving

4. Tokenizer alignment collator (`training/collators/tokenizer_aligner.py`)
   - Implement PyTorch DataLoader-compatible batch creation
   - Create collation functions for alignment matrices

5. Implement alignment algorithms (`align.py`)
   - Port the core token alignment algorithm to PyTorch

## Phase 3: Script Implementation

1. Cross-tokenizer distillation script (`scripts/cross_tokenizer_distill.py`)
   - Create main training loop
   - Implement logging and monitoring
   - Add support for all distillation methods (ALM, DSKD, ULD, MinED)

2. Zero-shot tokenizer transfer script (`scripts/zett.py`)
   - Implement FVT in PyTorch
   - Create transfer utilities

3. Evaluation script (`scripts/eval.py`)
   - Implement model loading and generation
   - Create evaluation metrics
   - Add token-level ensembling support

## Phase 4: Testing and Validation

1. Create unit tests for core components
   - Tokenizer and model loading tests
   - LoRA and distillation tests
   - Alignment algorithm tests

2. End-to-end tests
   - Test the full distillation pipeline on small models
   - Test ZeTT on small vocabulary subsets
   - Verify evaluation results match JAX implementation

3. Comparison with JAX implementation
   - Compare model outputs for the same inputs
   - Verify loss values match between implementations
   - Ensure training dynamics are similar

## Phase 5: Documentation and Deployment

1. Update documentation
   - Add PyTorch-specific notes to README
   - Create detailed usage examples
   - Document API changes

2. Add script examples
   - Create example scripts for common use cases
   - Create configuration examples

3. Performance optimization
   - Profile and optimize critical parts of the code
   - Add mixed precision training support
   - Tune batch sizes and learning rates

## Timeline

- Phase 1: Completed
- Phase 2: ~1 week
- Phase 3: ~1 week
- Phase 4: ~1 week
- Phase 5: ~3 days

Total estimated completion time: ~3-4 weeks