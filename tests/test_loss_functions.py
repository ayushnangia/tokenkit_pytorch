"""
Tests for loss functions in tokenkit.training.losses
"""

import torch
import numpy as np
import pytest
from typing import Dict, Any

from tokenkit.training.losses import (
    compute_clm_loss,
    compute_alm_loss,
    compute_alm_side_path_loss,
    compute_minED_loss,
    compute_baseline_mined_loss,
    compute_uld_loss,
    compute_dskd_loss,
    LossArgs,
    get_last_index_per_column
)


class SimpleArgs:
    """Simple class to hold arguments for loss functions."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockConfig:
    """Mock configuration object."""
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size


@pytest.fixture
def loss_args():
    """Create mock loss arguments for testing."""
    batch_size = 2
    seq_len = 8
    vocab_size_teacher = 32000
    vocab_size_student = 28000
    
    # Create mock inputs
    input_ids_teacher = torch.randint(0, vocab_size_teacher, (batch_size, seq_len))
    input_ids_student = torch.randint(0, vocab_size_student, (batch_size, seq_len))
    
    # Create mock attention masks
    attention_mask_teacher = torch.ones_like(input_ids_teacher)
    attention_mask_student = torch.ones_like(input_ids_student)
    
    # Create mock loss masks
    loss_mask_teacher = torch.ones_like(input_ids_teacher)
    loss_mask_student = torch.ones_like(input_ids_student)
    
    # Create mock logits and probabilities
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size_teacher)
    student_logits = torch.randn(batch_size, seq_len, vocab_size_student)
    
    teacher_logprobs = torch.log_softmax(teacher_logits, dim=-1)
    student_logprobs = torch.log_softmax(student_logits, dim=-1)
    
    teacher_probs = torch.softmax(teacher_logits, dim=-1)
    student_probs = torch.softmax(student_logits, dim=-1)
    
    # Create mock alignment matrices
    alignment_matrix_a_unconstrained = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
    alignment_matrix_b_unconstrained = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
    
    alignment_matrix_a_space = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
    alignment_matrix_b_space = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
    
    alignment_matrix_a_unbiased = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
    alignment_matrix_b_unbiased = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
    
    # Set some elements to True to create valid alignment chunks
    for i in range(batch_size):
        for j in range(seq_len - 1):
            # Each position aligns with itself and the next position
            alignment_matrix_a_unconstrained[i, j, j] = True
            alignment_matrix_b_unconstrained[i, j, j] = True
            
            alignment_matrix_a_space[i, j, j] = True
            alignment_matrix_b_space[i, j, j] = True
            
            alignment_matrix_a_unbiased[i, j, j] = True
            alignment_matrix_b_unbiased[i, j, j] = True
    
    # Create mock batch
    batch = {
        "input_ids_original": input_ids_teacher,
        "input_ids_new": input_ids_student,
        "attention_mask_original": attention_mask_teacher,
        "attention_mask_new": attention_mask_student,
        "loss_mask_original": loss_mask_teacher,
        "loss_mask_new": loss_mask_student,
        "alignment_matrix_a_unconstrained": alignment_matrix_a_unconstrained,
        "alignment_matrix_b_unconstrained": alignment_matrix_b_unconstrained,
        "alignment_matrix_a_space": alignment_matrix_a_space,
        "alignment_matrix_b_space": alignment_matrix_b_space,
        "alignment_matrix_a_unbiased": alignment_matrix_a_unbiased,
        "alignment_matrix_b_unbiased": alignment_matrix_b_unbiased
    }
    
    # Mock model outputs
    class MockOutput:
        def __init__(self, hidden_states, attentions=None):
            self.hidden_states = hidden_states
            self.attentions = attentions
    
    hidden_dim = 768
    n_layers = 3
    
    # Create mock hidden states
    teacher_hidden_states = [torch.randn(batch_size, seq_len, hidden_dim) for _ in range(n_layers)]
    student_hidden_states = [torch.randn(batch_size, seq_len, hidden_dim) for _ in range(n_layers)]
    
    # Create mock teacher/student outputs
    teacher_out = MockOutput(hidden_states=teacher_hidden_states)
    student_out = MockOutput(hidden_states=student_hidden_states)
    
    # Create mock tokenizers
    class MockTokenizer:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
            self.eos_token_id = vocab_size - 1
            self.pad_token_id = 0
    
    tokenizer_teacher = MockTokenizer(vocab_size_teacher)
    tokenizer_student = MockTokenizer(vocab_size_student)
    
    # Create mock configs
    teacher_config = MockConfig(vocab_size_teacher)
    student_config = MockConfig(vocab_size_student)
    
    # Create predicted embeddings
    predicted_embeddings = torch.randn(vocab_size_student, 1, hidden_dim)
    
    # Create scalar report
    scalar_report = {}
    
    # Create loss arguments
    return LossArgs(
        state=None,
        params=None,
        batch=batch,
        global_batch=batch,
        teacher_config=teacher_config,
        new_config=student_config,
        teacher_out=teacher_out,
        student_out=student_out,
        tokenizer_teacher=tokenizer_teacher,
        tokenizer_new=tokenizer_student,
        teacher_probs=teacher_probs,
        teacher_logprobs=teacher_logprobs,
        teacher_logits=teacher_logits,
        student_probs=student_probs,
        student_logprobs=student_logprobs,
        student_logits=student_logits,
        predicted_embeddings=predicted_embeddings,
        scalar_report=scalar_report
    )


def test_get_last_index_per_column():
    """Test get_last_index_per_column function."""
    matrix = torch.zeros(2, 3, 2, dtype=torch.bool)
    
    # Set some elements to True
    matrix[0, 0, 0] = True
    matrix[0, 1, 0] = True
    matrix[0, 2, 0] = True
    
    matrix[1, 0, 1] = True
    matrix[1, 1, 1] = True
    
    # Get last indices and mask
    indices, mask = get_last_index_per_column(matrix)
    
    # Check indices
    assert indices[0, 0] == 2  # Last True value in batch 0, column 0
    assert indices[1, 1] == 1  # Last True value in batch 1, column 1
    
    # Check mask
    assert mask[0, 0] == True  # Column 0 in batch 0 has a True value
    assert mask[0, 1] == False  # Column 1 in batch 0 has no True value
    assert mask[1, 0] == False  # Column 0 in batch 1 has no True value
    assert mask[1, 1] == True  # Column 1 in batch 1 has a True value


def test_clm_loss(loss_args):
    """Test standard causal language modeling loss."""
    args = SimpleArgs()
    loss = compute_clm_loss(args, loss_args)
    
    # Check that loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_alm_loss_unconstrained(loss_args):
    """Test ALM loss with unconstrained chunks."""
    args = SimpleArgs(
        alm_diff_fn="abs",
        distill_chunk_sizes=[1],
        distill_main_path_numerator="chunk_count",
        distill_main_path_denominator="chunk_count"
    )
    
    loss = compute_alm_loss("unconstrained", args, loss_args)
    
    # Check that loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_alm_loss_space(loss_args):
    """Test ALM loss with space-constrained chunks."""
    args = SimpleArgs(
        alm_diff_fn="abs",
        distill_chunk_sizes=[1],
        distill_main_path_numerator="chunk_count",
        distill_main_path_denominator="chunk_count"
    )
    
    loss = compute_alm_loss("space", args, loss_args)
    
    # Check that loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_alm_side_path_loss(loss_args):
    """Test ALM side path loss."""
    args = SimpleArgs(
        side_path_distance_fn="abs"
    )
    
    # Create student and teacher token mappings
    student_mapping = [0, 1, 2, 3, 4]
    teacher_mapping = [0, 1, 2, 3, 4]
    
    loss = compute_alm_side_path_loss("unconstrained", student_mapping, teacher_mapping, args, loss_args)
    
    # Check that loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_minED_loss(loss_args):
    """Test MinED loss."""
    args = SimpleArgs()
    
    loss = compute_minED_loss(args, loss_args)
    
    # Check that loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_baseline_mined_loss(loss_args):
    """Test baseline mined loss."""
    args = SimpleArgs(
        max_teacher_length=loss_args.batch["input_ids_original"].shape[1],
        max_student_length=loss_args.batch["input_ids_new"].shape[1]
    )
    
    # Create mock mined mapping
    mined_mapping = list(range(min(loss_args.teacher_config.vocab_size, loss_args.new_config.vocab_size)))
    
    loss = compute_baseline_mined_loss(mined_mapping, args, loss_args)
    
    # Check that loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_uld_loss(loss_args):
    """Test Universal Logit Distillation loss."""
    args = SimpleArgs()
    
    loss = compute_uld_loss(args, loss_args)
    
    # Check that loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_dskd_loss(loss_args):
    """Test Dual Space Knowledge Distillation loss."""
    args = SimpleArgs(
        dskd_alpha=0.5
    )
    
    loss = compute_dskd_loss(args, loss_args)
    
    # Check that loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() > 0