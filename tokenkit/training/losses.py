import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor

from tokenkit import baseline_utils, utils

# Small constant to avoid division by zero
EPSILON = 1e-10


def get_last_index_per_column(matrix: Tensor) -> Tuple[Tensor, Tensor]:
    """Find the last non-zero index in each column of the matrix.
    
    Args:
        matrix: A binary matrix of shape [batch_size, seq_len, chunk_len]
        
    Returns:
        Tuple of (indices, mask) where indices has shape [batch_size, seq_len]
        and mask has shape [batch_size, seq_len]
    """
    # Find positions where current is 1 but next is 0 (or end of sequence)
    matrix_last_only = matrix.clone()
    # Set to 0 where next position in chunk is also 1
    matrix_last_only[:, :, :-1] = matrix_last_only[:, :, :-1] & ~matrix[:, :, 1:]
    
    # Get indices of the last tokens in each chunk
    indices = torch.argmax(matrix_last_only.float(), dim=-2)
    mask = torch.max(matrix_last_only, dim=-2)[0]
    
    return indices, mask


def cross_entropy(
    logits: Tensor,
    labels: Tensor,
    attention_mask: Tensor,
    logits_already_shifted: bool = False,
    logit_mask: Optional[Tensor] = None,
    denom: Optional[Tensor] = None,
) -> Tensor:
    """Compute masked cross entropy loss.
    
    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        labels: Target token ids [batch_size, seq_len]
        attention_mask: Mask for padding [batch_size, seq_len]
        logits_already_shifted: Whether logits are already shifted (don't need to drop last token)
        logit_mask: Optional mask to apply to logits (e.g., for preventing certain tokens)
        denom: Optional normalization factor
        
    Returns:
        Loss value
    """
    shift_logits = logits[:, :-1] if not logits_already_shifted else logits
    shift_labels = labels[:, 1:]
    shift_attention_mask = attention_mask[:, 1:]
    
    if logit_mask is not None:
        shift_logits = shift_logits + logit_mask[None, None, :]
    
    # Use CrossEntropyLoss with token-level weights from attention mask
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    # Reshape for CrossEntropyLoss: [batch*seq, vocab_size]
    flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))
    flat_labels = shift_labels.reshape(-1)
    flat_attn_mask = shift_attention_mask.reshape(-1)
    
    # Compute token-level losses
    token_losses = loss_fn(flat_logits, flat_labels)
    
    # Apply attention mask and normalize
    masked_loss = (token_losses * flat_attn_mask).mean()
    
    # Normalize by the requested denominator or by the mean of the attention mask
    if denom is not None:
        masked_loss = masked_loss / denom
    else:
        masked_loss = masked_loss / (shift_attention_mask.mean())
    
    return masked_loss


@dataclass
class LossArgs:
    """Arguments for loss computation functions."""
    state: Any
    params: Any
    batch: Dict[str, Tensor]
    global_batch: Dict[str, Tensor]
    teacher_config: Any
    new_config: Any
    teacher_out: Any
    student_out: Any
    tokenizer_teacher: Any
    tokenizer_new: Any
    teacher_probs: Tensor
    teacher_logprobs: Tensor
    teacher_logits: Tensor
    student_probs: Tensor
    student_logprobs: Tensor
    student_logits: Tensor
    predicted_embeddings: Any
    scalar_report: Dict[str, float]


def log1mexp(x: Tensor) -> Tensor:
    """Computes log(1 - exp(x)) in a numerically stable way for x < 0."""
    # For x < log(0.5), use log1p(-exp(x)) directly
    # For x >= log(0.5), use log(-expm1(x)) to avoid precision issues
    log_half = -math.log(2)
    return torch.where(
        x < log_half, 
        torch.log1p(-torch.exp(x)), 
        torch.log(-torch.expm1(x))
    )


def compute_clm_loss(args: Any, loss_args: LossArgs) -> Tensor:
    """Compute standard causal language modeling loss."""
    clm_loss = cross_entropy(
        loss_args.student_logits,
        loss_args.batch["input_ids_new"],
        loss_args.batch["loss_mask_new"],
        denom=loss_args.global_batch["loss_mask_new"][:, 1:].mean(),
    )
    return clm_loss


def compute_alm_loss(chunk_kind: str, args: Any, loss_args: LossArgs, epsilon: float = 1e-6) -> Tensor:
    """Compute Approximate Likelihood Matching loss (PyTorch version).

    Args:
        chunk_kind: Type of chunking ('unconstrained', 'unbiased', or 'space').
        args: Configuration arguments object containing fields like:
              alm_diff_fn, bce_temp, renyi_alpha, alm_mode,
              tokenizer_pair_bias_threshold, distill_chunk_sizes,
              distill_main_path_numerator, distill_main_path_denominator.
        loss_args: LossArgs dataclass instance with required tensors and objects.
        epsilon: Small constant for numerical stability.

    Returns:
        ALM loss value (scalar tensor).
    """

    # this might be redundant, but it's a sanity check
    device = loss_args.student_logits.device
    dtype_float = loss_args.student_logits.dtype # Use student's float type






    original_shift_labels = loss_args.batch["input_ids_original"][..., 1:] # Shape: [batch, seq_orig-1]

    # --- 1. Select Alignment Matrices ---
    # Shape: [batch, seq_student, seq_original] or similar mapping structure
    # The exact shape depends on how alignment matrices are defined, assuming
    # [batch, seq_new, chunk_len] for A, [batch, seq_orig, chunk_len] for B
    # JAX code implies shape [batch, seq, chunk_len] where seq matches the model
    # Let's assume: A is [b, seq_new, chunk], B is [b, seq_orig, chunk]
    if chunk_kind == "unconstrained":
        alignment_matrix_a = loss_args.batch["alignment_matrix_a_unconstrained"]
        alignment_matrix_b = loss_args.batch["alignment_matrix_b_unconstrained"]
        global_alignment_matrix_a = loss_args.global_batch["alignment_matrix_a_unconstrained"]
        global_alignment_matrix_b = loss_args.global_batch["alignment_matrix_b_unconstrained"]
    elif chunk_kind == "unbiased":
        alignment_matrix_a = loss_args.batch["alignment_matrix_a_unbiased"]
        alignment_matrix_b = loss_args.batch["alignment_matrix_b_unbiased"]
        global_alignment_matrix_a = loss_args.global_batch["alignment_matrix_a_unbiased"]
        global_alignment_matrix_b = loss_args.global_batch["alignment_matrix_b_unbiased"]
    elif chunk_kind == "space":
        alignment_matrix_a = loss_args.batch["alignment_matrix_a_space"]
        alignment_matrix_b = loss_args.batch["alignment_matrix_b_space"]
        global_alignment_matrix_a = loss_args.global_batch["alignment_matrix_a_space"]
        global_alignment_matrix_b = loss_args.global_batch["alignment_matrix_b_space"]
    else:
        raise ValueError(f"Unknown chunk kind: {chunk_kind}")

    # Ensure boolean type for masking operations 
    # this might be redundant, but it's a sanity check
    # alignment_matrix_a = alignment_matrix_a.bool()
    # alignment_matrix_b = alignment_matrix_b.bool()
    # global_alignment_matrix_a = global_alignment_matrix_a.bool()
    # global_alignment_matrix_b = global_alignment_matrix_b.bool()

    # --- 2. Define Difference Function (`diff_fn`) ---
    # Inputs log_y_true, log_y_pred assumed to be log-probabilities (<= 0)
    if args.alm_diff_fn == "abs":
        diff_fn = lambda log_y_true, log_y_pred: torch.abs(log_y_true - log_y_pred)
    elif args.alm_diff_fn == "binary_ce":

        def binary_ce(log_y_true, log_y_pred):
            log_y_true = (log_y_true.to(torch.float32) / args.bce_temp) - epsilon
            log_y_pred = (log_y_pred.to(torch.float32) / args.bce_temp) - epsilon

            term1 = torch.exp(log_y_true) * log_y_pred
            term2 = (-torch.expm1(log_y_true)) * log1mexp(log_y_pred)
            return -(term1 + term2)

        diff_fn = binary_ce
    elif args.alm_diff_fn == "reverse_binary_kl":

        def reverse_binary_kl(log_y_true, log_y_pred):
            log_y_true = (log_y_true.to(torch.float32) / args.bce_temp) - epsilon
            log_y_pred = (log_y_pred.to(torch.float32) / args.bce_temp) - epsilon

            term1 = torch.exp(log_y_pred) * (log_y_pred - log_y_true)
            term2 = (-torch.expm1(log_y_pred)) * (log1mexp(log_y_pred) - log1mexp(log_y_true))
            return term1 + term2

        diff_fn = reverse_binary_kl
    elif args.alm_diff_fn == "binary_kl_temp_limit":

        def binary_kl_temp_limit(log_y_true, log_y_pred):
            log_y_true = log_y_true - epsilon
            log_y_pred = log_y_pred - epsilon

            # Ensure inputs log_y_true, log_y_pred are strictly negative after subtracting epsilon
            # for torch.log(-log_y...) to be valid. Clamping might be needed depending on input range.
            term1 = log_y_true - log_y_pred
            term2 = log_y_true * torch.log(-log_y_pred)
            term3 = log_y_true * torch.log(-log_y_true)
            return term1 + term2 - term3

        diff_fn = binary_kl_temp_limit
    elif args.alm_diff_fn == "abs_exp":

        def abs_exp(log_y_true, log_y_pred):
            log_y_true = (log_y_true.to(torch.float32) / args.bce_temp) - epsilon
            log_y_pred = (log_y_pred.to(torch.float32) / args.bce_temp) - epsilon

            return torch.abs(torch.exp(log_y_true) - torch.exp(log_y_pred))

        diff_fn = abs_exp
    elif args.alm_diff_fn == "renyi":

        def renyi(log_y_true, log_y_pred):
            log_y_true = log_y_true.to(torch.float32) - epsilon
            log_y_pred = log_y_pred.to(torch.float32) - epsilon

            log_one_minus_y_true = log1mexp(log_y_true)
            log_one_minus_y_pred = log1mexp(log_y_pred)

            term1 = args.renyi_alpha * log_y_true + (1.0 - args.renyi_alpha) * log_y_pred
            term2 = (
                args.renyi_alpha * log_one_minus_y_true
                + (1.0 - args.renyi_alpha) * log_one_minus_y_pred
            )

            log_sum = torch.logaddexp(term1, term2)
            return log_sum / (args.renyi_alpha - 1.0)

        diff_fn = renyi
    elif args.alm_diff_fn == "joschu_k2":
        def joschu_k2(log_y_true, log_y_pred):
            logr = (log_y_true - log_y_pred) / args.bce_temp
            return (logr ** 2) / 2.0

        diff_fn = joschu_k2
    elif args.alm_diff_fn == "joschu_k3":
        def joschu_k3(log_y_true, log_y_pred):
            logr = (log_y_true - log_y_pred) / args.bce_temp
            return (torch.exp(logr) - 1.0) - logr

        diff_fn = joschu_k3
    else:
        raise NotImplementedError(f"Unknown diff function: {args.alm_diff_fn}")

    # --- 3. Apply Loss Masks ---
    # Masks: [batch, seq_len] -> Unsqueeze -> [batch, seq_len, 1]
    loss_mask_new = loss_args.batch["loss_mask_new"].unsqueeze(-1)
    loss_mask_original = loss_args.batch["loss_mask_original"].unsqueeze(-1)
    global_loss_mask_new = loss_args.global_batch["loss_mask_new"].unsqueeze(-1)
    global_loss_mask_original = loss_args.global_batch["loss_mask_original"].unsqueeze(-1)

    alignment_matrix_a = alignment_matrix_a * loss_mask_new
    alignment_matrix_b = alignment_matrix_b * loss_mask_original
    global_alignment_matrix_a = global_alignment_matrix_a * global_loss_mask_new
    global_alignment_matrix_b = global_alignment_matrix_b * global_loss_mask_original

    # --- Get Last Indices & Chunk Stats ---
    # Assuming _get_last_index_per_column is defined elsewhere and mirrors JAX functionality
    alignment_matrix_b_last_only_index, _ = _get_last_index_per_column(
        alignment_matrix_b
    )
    alignment_matrix_a_last_only_index, mask = _get_last_index_per_column(
        alignment_matrix_a
    )

    student_chunk_sums = alignment_matrix_a.sum(dim=-2)
    teacher_chunk_sums = alignment_matrix_b.sum(dim=-2)

    # Ensure float division and handle potential division by zero if mean is zero
    epsilon = torch.finfo(student_chunk_sums.dtype).eps if student_chunk_sums.is_floating_point() else 1e-8

    student_denominator = (student_chunk_sums > 0).float().mean()
    teacher_denominator = (teacher_chunk_sums > 0).float().mean()

    student_avg_chunk_lengths = student_chunk_sums.mean() / (student_denominator + epsilon)
    teacher_avg_chunk_lengths = teacher_chunk_sums.mean() / (teacher_denominator + epsilon)

    loss_args.scalar_report["student_avg_chunk_lengths"] = student_avg_chunk_lengths.item()
    loss_args.scalar_report["teacher_avg_chunk_lengths"] = teacher_avg_chunk_lengths.item()

    # --- 5. Calculate Teacher Main Path Logprobs ---
    # teacher_logprobs: [b, seq_orig, vocab] -> select target tokens -> [b, seq_orig-1]
    teacher_main_path_logprobs = torch.gather(
        loss_args.teacher_logprobs[:, :-1],
        dim=-1,
        index=original_shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    t_aligned_main_logp = torch.matmul(
        teacher_main_path_logprobs.unsqueeze(1),
        alignment_matrix_b[:, 1:] # Assuming alignment_matrix_b is shape [b, seq_orig, chunk_len]
                                # PyTorch slicing [:, 1:] on 3D+ tensors slices the second dimension
    ).squeeze(1)

    t_aligned_main_logp = torch.clamp(t_aligned_main_logp, max=0.0)

    # --- 6. Calculate Teacher Space Logprobs ---
    if "eos_as_space" in args.alm_mode:
        eos_id = loss_args.tokenizer_teacher.eos_token_id
        if eos_id is None: raise ValueError("eos_token_id needed for eos_as_space mode")
        # teacher_logprobs: [b, seq_orig, vocab] -> index EOS -> [b, seq_orig]
        t_space_logp = loss_args.teacher_logprobs[:, :, eos_id]
    else:
        # Dot product: [b, seq_orig, vocab] @ [vocab] -> [b, seq_orig]
        # Ensure space_mask is tensor, correct device/dtype
        space_mask_teacher = loss_args.state.space_mask_teacher.to(
            device=loss_args.teacher_probs.device, dtype=loss_args.teacher_probs.dtype
        ) # Shape [vocab_teacher]
        t_space_prob = torch.matmul(loss_args.teacher_probs, space_mask_teacher) # Shape [b, seq_orig]
        t_space_logp = torch.log(torch.clamp(t_space_prob, min=epsilon))
        t_space_logp = torch.clamp(t_space_logp, max=0.0)

    # Gather space logp at the end of each teacher chunk sequence
    # Input: [b, seq_orig], Index: [b, chunk_len] -> Output: [b, chunk_len]
    t_aligned_space_logp = torch.gather(
        t_space_logp,
        dim=1, # Gather along seq_orig dimension
        index=alignment_matrix_b_last_only_index # Indices are positions in seq_orig
    )

    # --- 7. Calculate Student Main Path Logprobs ---
    new_shift_labels = loss_args.batch["input_ids_new"][:, 1:] # Shape: [b, seq_new-1]
    student_main_path_logprobs = torch.gather(
        loss_args.student_logprobs[:, :-1], # Shift logits
        dim=-1,
        index=new_shift_labels.unsqueeze(-1)
    ).squeeze(-1) # Shape: [b, seq_new-1]

    # Matmul: [b, 1, seq_new-1] x [b, seq_new-1, chunk] -> [b, 1, chunk]
    alignment_matrix_a_shifted = alignment_matrix_a[:, 1:, :].to(dtype_float) # Shape [b, seq_new-1, chunk]
    s_aligned_main_logp = torch.matmul(
        student_main_path_logprobs.unsqueeze(1),
        alignment_matrix_a_shifted
    ).squeeze(1) # Result shape [batch, chunk_len]
    s_aligned_main_logp = torch.clamp(s_aligned_main_logp, max=0.0)

    # --- 8. Calculate Student Space Logprobs ---
    if "eos_as_space" in args.alm_mode:
        eos_id = loss_args.tokenizer_new.eos_token_id
        if eos_id is None: raise ValueError("eos_token_id needed for eos_as_space mode")
        s_space_logp = loss_args.student_logprobs[:, :, eos_id] # Shape [b, seq_new]
    else:
        space_mask_new = loss_args.state.space_mask_new.to(
             device=loss_args.student_probs.device, dtype=loss_args.student_probs.dtype
        ) # Shape [vocab_new]
        s_space_prob = torch.matmul(loss_args.student_probs, space_mask_new) # Shape [b, seq_new]
        s_space_logp = torch.log(torch.clamp(s_space_prob, min=epsilon))
        s_space_logp = torch.clamp(s_space_logp, max=0.0)

    # Gather space logp at the end of each student chunk sequence
    s_aligned_space_logp = torch.gather(
        s_space_logp,
        dim=1, # Gather along seq_new dimension
        index=alignment_matrix_a_last_only_index # Indices are positions in seq_new
    )
    alignment_matrix_b_shifted = alignment_matrix_b[:, 1:, :].to(dtype_float)
    # --- 9. Get Aligned Counts ---
    # Number of original tokens contributing to each chunk column. Sum over seq_orig dim.
    # Use alignment_matrix_b_shifted to match shifted logprobs [b, seq_orig-1, chunk]
    aligned_count = alignment_matrix_b_shifted.sum(dim=-2).to(dtype_float) # Shape [batch, chunk_len]

    # Global counts (for denominator scaling later)
    global_alignment_matrix_b_shifted = global_alignment_matrix_b[:, 1:, :].to(dtype_float)
    global_aligned_count = global_alignment_matrix_b_shifted.sum(dim=-2) # Shape [global_batch, chunk_len]

    # --- 10. Handle `merge_by_space_prob` (Optional) ---
    if "merge_by_space_prob" in args.alm_mode:
        batch_size = t_aligned_space_logp.shape[0]
        chunk_len_old = t_aligned_space_logp.shape[-1] # Original number of chunk columns

        # Identify chunks ending with high-probability space (using teacher's space prob)
        t_aligned_space_chunk_mask = (torch.exp(t_aligned_space_logp) > args.tokenizer_pair_bias_threshold) # [b, chunk_old]

        # Assign merge index based on cumsum of space masks (JAX trick)
        cumsum_rev = torch.cumsum(t_aligned_space_chunk_mask.int()[:, ::-1], dim=-1) # [b, chunk_old]
        chunk_merging_indices = cumsum_rev[:, ::-1] # [b, chunk_old]
        # Normalize indices to start from 0 for each sample
        chunk_merging_indices = chunk_merging_indices.max(dim=-1, keepdim=True)[0] - chunk_merging_indices # [b, chunk_old]

        # Determine the number of new chunks after merging
        num_chunks_new = chunk_merging_indices.max().item() + 1

        # Create the merge matrix [batch, chunk_new, chunk_old] where M[b, n, o] = 1 if old chunk o belongs to new chunk n
        chunk_merging_matrix = torch.zeros(
            (batch_size, num_chunks_new, chunk_len_old),
            dtype=dtype_float, device=device
        )
        # Batch indices for assignment
        batch_idx_mg = torch.arange(batch_size, device=device).unsqueeze(-1).expand(-1, chunk_len_old).flatten()
        old_chunk_idx_mg = torch.arange(chunk_len_old, device=device).unsqueeze(0).expand(batch_size, -1).flatten()
        new_chunk_idx_mg = chunk_merging_indices.flatten()

        # Values to assign (1.0 only if the original chunk was valid)
        chunk_merging_values = (aligned_count > 0).to(dtype_float).flatten() # Use original counts mask

        # Assign using flat indices
        chunk_merging_matrix[batch_idx_mg, new_chunk_idx_mg, old_chunk_idx_mg] = chunk_merging_values

        # Transpose for matmul: [batch, chunk_old, chunk_new]
        chunk_merging_matrix_t = chunk_merging_matrix.transpose(1, 2) # [b, chunk_old, chunk_new]

        # Merge logprobs: [b, 1, chunk_old] @ [b, chunk_old, chunk_new] -> [b, 1, chunk_new]
        t_aligned_main_logp = torch.matmul(t_aligned_main_logp.unsqueeze(1), chunk_merging_matrix_t).squeeze(1) # [b, chunk_new]
        s_aligned_main_logp = torch.matmul(s_aligned_main_logp.unsqueeze(1), chunk_merging_matrix_t).squeeze(1) # [b, chunk_new]

        # Find the index of the *last original chunk* contributing to each *new chunk*
        # This tells us which original space probability to use for the merged chunk.
        # Use argmax on the merge matrix [b, new, old] along the 'old' dimension
        merge_matrix_last_indices = torch.argmax(chunk_merging_matrix.float(), dim=-1) # [b, new_chunk] -> indices are old_chunk indices

        # Gather the space logps using these indices
        t_aligned_space_logp = torch.gather(
            t_aligned_space_logp, # [b, chunk_old]
            dim=1,                # Gather along old_chunk dim
            index=merge_matrix_last_indices # [b, chunk_new] containing old_chunk indices
        )
        s_aligned_space_logp = torch.gather(
            s_aligned_space_logp, # [b, chunk_old]
            dim=1,
            index=merge_matrix_last_indices
        )

        # Merge counts: [b, 1, chunk_old] @ [b, chunk_old, chunk_new] -> [b, 1, chunk_new]
        aligned_count = torch.matmul(aligned_count.unsqueeze(1), chunk_merging_matrix_t).squeeze(1) # [b, chunk_new]

        # Update mask based on merged counts
        mask = (aligned_count > 0) # New mask shape [b, chunk_new]

        # Report merged stats
        teacher_chunk_sums_after_merge = aligned_count
        teacher_avg_chunk_lengths_after_merge = teacher_chunk_sums_after_merge.sum() / (mask.sum() + epsilon)
        loss_args.scalar_report["teacher_avg_chunk_lengths_after_merge"] = teacher_avg_chunk_lengths_after_merge.item()
        # Find min space logp where merged chunk is valid
        if mask.any():
            min_space_logp = torch.min(t_aligned_space_logp[mask])
        else:
            min_space_logp = torch.tensor(0.0, device=device)
        loss_args.scalar_report["t_min_aligned_space_logp"] = min_space_logp.item()
        # Note: Global counts remain unmerged as per JAX logic.

    # --- 11. Mask Invalid Aligned Logprobs After Potential Merge ---
    # Ensure logprobs are only considered where original tokens existed (reflected in merged count)
    valid_chunk_mask = (aligned_count > 0) # Shape [b, chunk_len (possibly merged)]
    # Use where to avoid NaN gradients from 0 * inf if logp was -inf
    large_neg_val = _get_large_negative_number(dtype_float)
    t_aligned_main_logp = torch.where(valid_chunk_mask, t_aligned_main_logp, large_neg_val)
    s_aligned_main_logp = torch.where(valid_chunk_mask, s_aligned_main_logp, large_neg_val)
    t_aligned_space_logp = torch.where(valid_chunk_mask, t_aligned_space_logp, large_neg_val)
    s_aligned_space_logp = torch.where(valid_chunk_mask, s_aligned_space_logp, large_neg_val)


    # --- 12. Distillation Over Multiple Chunk Sizes ---
    all_aligned_s_logps = []
    all_aligned_t_logps = []
    all_aligned_counts = []
    global_all_aligned_counts = []

    current_chunk_len = aligned_count.shape[-1]
    global_current_chunk_len = global_aligned_count.shape[-1]
    batch_size = aligned_count.shape[0]
    global_batch_size = global_aligned_count.shape[0]

    processed_distill_sizes = []

    for size in args.distill_chunk_sizes:
        if size <= 0: continue
        if current_chunk_len % size != 0:
            print(f"Warning: Local chunk length {current_chunk_len} not divisible by distill size {size}. Padding or skipping needed. Skipping size {size}.")
            continue
        # Check global divisibility too, if global counts are used for denominator
        if args.distill_main_path_denominator != "skip_global_check" and global_current_chunk_len % size != 0:
             print(f"Warning: Global chunk length {global_current_chunk_len} not divisible by distill size {size}. Skipping size {size}.")
             continue

        processed_distill_sizes.append(size) # Keep track of sizes actually used
        num_chunks_new = current_chunk_len // size
        global_num_chunks_new = global_current_chunk_len // size

        # Reshape and sum: [b, chunk_len] -> [b, num_new, size] -> sum over size -> [b, num_new]
        size_s_logp = s_aligned_main_logp.reshape(batch_size, num_chunks_new, size).sum(dim=-1)
        size_t_logp = t_aligned_main_logp.reshape(batch_size, num_chunks_new, size).sum(dim=-1)
        size_count = aligned_count.reshape(batch_size, num_chunks_new, size) # Keep size dim for now
        # Reshape global counts only if they were divisible
        if global_current_chunk_len % size == 0:
            global_size_count = global_aligned_count.reshape(global_batch_size, global_num_chunks_new, size)
        else:
            # This case should be skipped by the check above, but as fallback:
            global_size_count = torch.zeros((global_batch_size, global_num_chunks_new, size), device=device, dtype=dtype_float)


        if "append_space" in args.alm_mode:
            # Find last valid *original* chunk within the new *size* chunk
            count_mask = (size_count > 0) # [b, num_new, size]
            # Cumsum from right to left along size dim
            cumsum_rev_size = torch.cumsum(count_mask.int()[:, :, ::-1], dim=-1)
            last_pos_mask = (cumsum_rev_size == 1)[:, :, ::-1] # [b, num_new, size]

            # Reshape space logps [b, chunk_len] -> [b, num_new, size]
            s_space_logp_reshaped = s_aligned_space_logp.reshape(batch_size, num_chunks_new, size)
            t_space_logp_reshaped = t_aligned_space_logp.reshape(batch_size, num_chunks_new, size)

            # Add space logp at the last valid position (use where to avoid adding large negatives)
            size_s_logp = size_s_logp + torch.where(last_pos_mask, s_space_logp_reshaped, 0.0).sum(dim=-1)
            size_t_logp = size_t_logp + torch.where(last_pos_mask, t_space_logp_reshaped, 0.0).sum(dim=-1)

        all_aligned_s_logps.append(size_s_logp)
        all_aligned_t_logps.append(size_t_logp)
        # Sum counts over the size dimension now -> [b, num_new]
        all_aligned_counts.append(size_count.sum(dim=-1))
        # Sum global counts -> [g, num_new]
        global_all_aligned_counts.append(global_size_count.sum(dim=-1))

    # --- 13. Concatenate Results Across Chunk Sizes ---
    if not all_aligned_s_logps:
         # Handle case where no valid sizes were found
         print("Warning: No valid distill_chunk_sizes processed for ALM loss.")
         return torch.tensor(0.0, device=device, requires_grad=True)

    s_full_aligned_main_logp = torch.cat(all_aligned_s_logps, dim=-1) # [b, total_distill_chunks]
    t_full_aligned_main_logp = torch.cat(all_aligned_t_logps, dim=-1) # [b, total_distill_chunks]
    full_aligned_counts = torch.cat(all_aligned_counts, dim=-1)       # [b, total_distill_chunks]
    global_full_aligned_counts = torch.cat(global_all_aligned_counts, dim=-1) # [g, total_distill_chunks]

    # --- 14. Mask Final Logprobs Based on Counts ---
    final_valid_mask = (full_aligned_counts > 0) # [b, total_distill_chunks]
    global_final_valid_mask = (global_full_aligned_counts > 0) #[g, total_distill_chunks]

    t_full_aligned_main_logp = torch.where(
        final_valid_mask,
        t_full_aligned_main_logp,
        torch.tensor(large_neg_val, device=device, dtype=dtype_float)
    )
    s_full_aligned_main_logp = torch.where(
        final_valid_mask,
        s_full_aligned_main_logp,
         torch.tensor(large_neg_val, device=device, dtype=dtype_float)
    )

    # --- 15. Report Statistics ---
    num_valid_local_distill_chunks = final_valid_mask.sum()
    if num_valid_local_distill_chunks > 0:
        valid_t_logps = t_full_aligned_main_logp[final_valid_mask]
        valid_s_logps = s_full_aligned_main_logp[final_valid_mask]
        loss_args.scalar_report["t_min_p"] = torch.min(valid_t_logps).item()
        loss_args.scalar_report["t_mean_p"] = torch.mean(valid_t_logps).item()
        loss_args.scalar_report["t_max_p"] = torch.max(valid_t_logps).item()
        loss_args.scalar_report["s_min_p"] = torch.min(valid_s_logps).item()
        loss_args.scalar_report["s_mean_p"] = torch.mean(valid_s_logps).item()
        loss_args.scalar_report["s_max_p"] = torch.max(valid_s_logps).item()
    else:
        loss_args.scalar_report.update({k: 0.0 for k in ["t_min_p", "t_mean_p", "t_max_p", "s_min_p", "s_mean_p", "s_max_p"]})

    # Report legacy loss (using pre-distillation values, after merge if applied)
    legacy_mask = (aligned_count > 0) # Mask before multi-size distillation
    num_legacy_valid = legacy_mask.sum()
    if num_legacy_valid > 0:
        loss_args.scalar_report["legacy_loss"] = (
             torch.abs(s_aligned_main_logp - t_aligned_main_logp)[legacy_mask].mean().item()
         )
    else:
         loss_args.scalar_report["legacy_loss"] = 0.0


    # --- 16. Calculate Numerator/Denominator for Loss Weighting ---
    if args.distill_main_path_numerator == "token_count":
        numerator = full_aligned_counts # Weight by number of original tokens
    elif args.distill_main_path_numerator == "chunk_count":
        numerator = final_valid_mask.to(dtype_float) # Weight each valid distilled chunk equally (1.0)
    elif args.distill_main_path_numerator == "log1p_token_count":
        numerator = torch.log1p(full_aligned_counts) # Weight by log(1 + token_count)
    else:
        raise ValueError(f"Unknown numerator type: {args.distill_main_path_numerator}")


    if args.distill_main_path_denominator == "token_count":
        # Mean token count per distilled chunk over the global batch
        denominator = global_full_aligned_counts.sum() / (global_final_valid_mask.sum() + epsilon)
        # Alternative: Mean token count per *sample* in global batch? JAX mean flattens.
        # JAX .mean() computes sum / total_elements. Let's match that.
        # denominator = global_full_aligned_counts.float().mean() # Mean over global_batch * total_distill_chunks
    elif args.distill_main_path_denominator == "chunk_count":
        # Mean number of valid distilled chunks per sample over the global batch
        # denominator = global_final_valid_mask.float().sum() / global_batch_size
        # JAX mean flattens: proportion of valid chunks globally
        denominator = global_final_valid_mask.float().mean() # Mean over global_batch * total_distill_chunks
    else:
        # Allow skipping global check if needed, use local count maybe?
        # Let's stick to the specified options.
        raise ValueError(f"Unknown denominator type: {args.distill_main_path_denominator}")

    denominator = denominator + epsilon # Avoid division by zero

    # --- 17. Calculate Elementwise Loss ---
    # Apply diff function only where the chunk is valid
    elementwise_loss = torch.zeros_like(t_full_aligned_main_logp)
    elementwise_loss[final_valid_mask] = diff_fn(
        t_full_aligned_main_logp[final_valid_mask],
        s_full_aligned_main_logp[final_valid_mask]
    )

    # Apply weighting and normalization
    elementwise_loss = (elementwise_loss * numerator) / denominator

    # --- 18. Final Loss Calculation ---
    # Mean of the valid, weighted, normalized elementwise losses.
    # JAX .mean() equivalent: sum over valid elements / count of valid elements
    # The JAX code divides this mean by len(args.distill_chunk_sizes) at the end.

    if num_valid_local_distill_chunks > 0:
         # Sum only the valid elements of the already weighted/normalized loss
         loss_sum = elementwise_loss[final_valid_mask].sum()
         # Average by the number of valid chunks in the local batch
         mean_loss_local = loss_sum / num_valid_local_distill_chunks
    else:
         mean_loss_local = torch.tensor(0.0, device=device, dtype=dtype_float)

    # Final division by number of *processed* sizes matches JAX structure
    if not processed_distill_sizes:
        final_loss = torch.tensor(0.0, device=device, dtype=dtype_float) # Should already be handled
    else:
        final_loss = mean_loss_local / len(processed_distill_sizes)

    # Ensure the loss requires gradients if inputs did
    if loss_args.student_logits.requires_grad:
        final_loss.requires_grad_(True)

    return final_loss


def compute_alm_latents_loss(args: Any, loss_args: LossArgs) -> Tensor:
    """Compute loss between hidden states of teacher and student models.
    
    Args:
        args: Configuration arguments
        loss_args: Loss function arguments
        
    Returns:
        Latent loss value
    """
    # Get indices from alignment matrices based on chunks type
    if args.latents_chunks == "naive":
        alignment_matrix_b_last_only_index, _ = get_last_index_per_column(
            loss_args.batch["alignment_matrix_b_unconstrained"]
        )
        alignment_matrix_a_last_only_index, mask = get_last_index_per_column(
            loss_args.batch["alignment_matrix_a_unconstrained"]
        )
        _, global_mask = get_last_index_per_column(
            loss_args.global_batch["alignment_matrix_a_unconstrained"]
        )
    elif args.latents_chunks == "space":
        alignment_matrix_b_last_only_index, _ = get_last_index_per_column(
            loss_args.batch["alignment_matrix_b_space"]
        )
        alignment_matrix_a_last_only_index, mask = get_last_index_per_column(
            loss_args.batch["alignment_matrix_a_space"]
        )
        _, global_mask = get_last_index_per_column(
            loss_args.global_batch["alignment_matrix_a_space"]
        )
    
    # Determine which layers to align
    if "last_hidden_state" in args.latents_to_align:
        layer_indices = [(-1, -1)]  # Just align the last layer
    elif "all_hidden_states" in args.latents_to_align:
        student_layers = loss_args.new_config.num_hidden_layers
        layer_indices = [
            (i, i + args.n_prefix_layers)
            for i in range(student_layers + 1)  # +1 for embeddings
        ]
    else:
        layer_indices = []
    
    # Initialize loss components
    hidden_state_latent_loss = torch.tensor(0.0, device=mask.device)
    attention_latent_loss = torch.tensor(0.0, device=mask.device)
    
    # Calculate hidden state loss across specified layers
    for teacher_idx, student_idx in layer_indices:
        # Get aligned hidden states
        t_hidden_state = loss_args.teacher_out.hidden_states[teacher_idx]
        s_hidden_state = loss_args.student_out.hidden_states[student_idx]
        
        # Use torch.gather to extract aligned hidden states
        # Expand index to match hidden state dimensions for gather
        gather_index_b = alignment_matrix_b_last_only_index.unsqueeze(-1).expand(-1, -1, t_hidden_state.size(-1))
        gather_index_a = alignment_matrix_a_last_only_index.unsqueeze(-1).expand(-1, -1, s_hidden_state.size(-1))
        
        t_aligned_last_hidden_state = torch.gather(t_hidden_state, 1, gather_index_b)
        s_aligned_last_hidden_state = torch.gather(s_hidden_state, 1, gather_index_a)
        
        # Apply projection if needed
        if args.latents_do_project:
            projector = loss_args.params["model"]["projector_latents"]
            s_aligned_last_hidden_state = torch.matmul(
                s_aligned_last_hidden_state, projector["kernel"]
            ) + projector["bias"]
        
        # Calculate loss based on normalization type
        if args.latents_normalization.startswith("l2"):
            elementwise_layer_latent_loss = torch.square(
                s_aligned_last_hidden_state - t_aligned_last_hidden_state
            )
        elif args.latents_normalization.startswith("l1"):
            elementwise_layer_latent_loss = torch.abs(
                s_aligned_last_hidden_state - t_aligned_last_hidden_state
            )
        
        # Apply mask
        layer_latent_loss = elementwise_layer_latent_loss * mask.unsqueeze(-1)
        
        # Apply normalization based on configuration
        if args.latents_normalization == "l2":
            layer_latent_loss = layer_latent_loss / (
                (torch.square(t_aligned_last_hidden_state * mask.unsqueeze(-1)).mean() / 
                 mask.mean()) + EPSILON
            )
        elif args.latents_normalization == "l2_channelwise":
            layer_latent_loss = layer_latent_loss / (
                (torch.square(t_aligned_last_hidden_state * mask.unsqueeze(-1))
                 .mean(dim=[0, 1], keepdim=True) / mask.mean()) + EPSILON
            )
        elif args.latents_normalization == "l1":
            layer_latent_loss = layer_latent_loss / (
                (torch.abs(t_aligned_last_hidden_state * mask.unsqueeze(-1)).mean() / 
                 mask.mean()) + EPSILON
            )
        elif args.latents_normalization == "l1_channelwise":
            layer_latent_loss = layer_latent_loss / (
                (torch.abs(t_aligned_last_hidden_state * mask.unsqueeze(-1))
                 .mean(dim=[0, 1], keepdim=True) / mask.mean()) + EPSILON
            )
        
        # Calculate final layer loss and add to total
        layer_latent_loss = layer_latent_loss.mean() / global_mask.mean()
        hidden_state_latent_loss += layer_latent_loss / len(layer_indices)
    
    # Add attention alignment if requested
    if "qkv" in args.latents_to_align and hasattr(loss_args.teacher_out, "attentions"):
        for layer_idx in range(len(loss_args.teacher_out.attentions)):
            teacher_idx = layer_idx
            student_idx = layer_idx + args.n_prefix_layers
            
            # Concatenate QKV attention states
            teacher_qkv = torch.cat(loss_args.teacher_out.attentions[teacher_idx], dim=-1)
            student_qkv = torch.cat(loss_args.student_out.attentions[student_idx], dim=-1)
            
            # Extract aligned attention states using torch.gather
            # Expand index to match QKV dimensions for gather
            gather_index_b_qkv = alignment_matrix_b_last_only_index.unsqueeze(-1).expand(-1, -1, teacher_qkv.size(-1))
            gather_index_a_qkv = alignment_matrix_a_last_only_index.unsqueeze(-1).expand(-1, -1, student_qkv.size(-1))

            t_aligned_qkv = torch.gather(teacher_qkv, 1, gather_index_b_qkv)
            s_aligned_qkv = torch.gather(student_qkv, 1, gather_index_a_qkv)
            
            # Calculate loss
            elementwise_layer_latent_loss = torch.square(s_aligned_qkv - t_aligned_qkv)
            layer_latent_loss = elementwise_layer_latent_loss * mask.unsqueeze(-1)
            
            # Apply normalization
            if args.latents_normalization == "l2":
                layer_latent_loss = layer_latent_loss / (
                    torch.square(t_aligned_qkv * mask.unsqueeze(-1)).mean() / mask.mean()
                )
            elif args.latents_normalization == "l2_channelwise":
                layer_latent_loss = layer_latent_loss / (
                    torch.square(t_aligned_qkv * mask.unsqueeze(-1))
                    .mean(dim=[0, 1], keepdim=True) / mask.mean()
                )
            
            # Calculate final layer loss and add to total
            layer_latent_loss = layer_latent_loss.mean() / global_mask.mean()
            attention_latent_loss += layer_latent_loss / len(loss_args.teacher_out.attentions)
    
    # Combine losses
    loss = hidden_state_latent_loss + attention_latent_loss
    
    # Record in report
    loss_args.scalar_report["hidden_state_latent_loss"] = hidden_state_latent_loss.item()
    loss_args.scalar_report["attention_latent_loss"] = attention_latent_loss.item()
    
    return loss


# Implementation for other distillation methods

def compute_minED_loss(args: Any, loss_args: LossArgs) -> Tensor:
    """Minimum Edit Distance loss as described in the MinED paper."""
    # Get student/teacher logits
    student_logits = loss_args.student_logits
    teacher_logits = loss_args.teacher_logits
    
    # Get alignment matrices
    alignment_matrix_a = loss_args.batch["alignment_matrix_a_space"]
    alignment_matrix_b = loss_args.batch["alignment_matrix_b_space"]
    
    # Apply loss mask
    loss_mask_new = loss_args.batch["loss_mask_new"]
    loss_mask_original = loss_args.batch["loss_mask_original"]
    
    # Create soft mapping matrix from alignments
    a_sum = alignment_matrix_a.sum(dim=-1, keepdim=True) + EPSILON
    b_sum = alignment_matrix_b.sum(dim=-1, keepdim=True) + EPSILON
    
    # Normalize alignments to get conditional probabilities
    p_a_given_b = alignment_matrix_a / a_sum
    p_b_given_a = alignment_matrix_b / b_sum
    
    # Map teacher logits to student space
    mapped_teacher_logits = torch.matmul(
        p_a_given_b.transpose(1, 2), 
        teacher_logits[:, :-1]
    )
    
    # Calculate MSE loss between aligned logits
    loss_fn = nn.MSELoss(reduction='none')
    token_losses = loss_fn(student_logits[:, :-1], mapped_teacher_logits)
    
    # Apply mask and calculate mean
    masked_loss = (token_losses * loss_mask_new[:, :-1].unsqueeze(-1)).mean()
    
    return masked_loss


def compute_baseline_mined_loss(mined_mapping: List[int], args: Any, loss_args: LossArgs) -> Tensor:
    """Compute loss using a mined token mapping.
    
    This loss function uses a pre-computed mapping between teacher and student tokens,
    typically generated by a token mining process to identify matching tokens across vocabularies.
    
    Args:
        mined_mapping: A list where each index corresponds to a teacher token ID
                      and the value is the corresponding student token ID
        args: Configuration arguments
        loss_args: Loss function arguments
        
    Returns:
        Mined mapping KL loss value
    """
    # Ensure max teacher and student lengths match
    assert args.max_teacher_length == args.max_student_length, \
        "Teacher and student max lengths must be equal for mined loss"
    
    # Apply loss masks to alignment matrices
    alignment_matrix_a = loss_args.batch["alignment_matrix_a_unconstrained"][:, :-1] & \
                         loss_args.batch["loss_mask_new"][:, 1:, None]
    alignment_matrix_b = loss_args.batch["alignment_matrix_b_unconstrained"][:, :-1] & \
                         loss_args.batch["loss_mask_original"][:, 1:, None]
    global_alignment_matrix_a = loss_args.global_batch["alignment_matrix_a_unconstrained"][:, :-1] & \
                               loss_args.global_batch["loss_mask_new"][:, 1:, None]
    global_alignment_matrix_b = loss_args.global_batch["alignment_matrix_b_unconstrained"][:, :-1] & \
                               loss_args.global_batch["loss_mask_original"][:, 1:, None]
    
    # Get last indices for aligned tokens
    alignment_matrix_b_last_only_index, _ = get_last_index_per_column(alignment_matrix_b)
    alignment_matrix_a_last_only_index, _ = get_last_index_per_column(alignment_matrix_a)
    
    # Identify one-to-one alignments
    one_to_one_mask = (alignment_matrix_b[:, 1:].sum(-2) == 1) & \
                      (alignment_matrix_a[:, 1:].sum(-2) == 1)
    global_one_to_one_mask = (global_alignment_matrix_b[:, 1:].sum(-2) == 1) & \
                            (global_alignment_matrix_a[:, 1:].sum(-2) == 1)
    
    # Create mined teacher logits by mapping teacher token logits to student token space
    mined_mapping_tensor = torch.tensor(
        mined_mapping + [0] * (loss_args.new_config.vocab_size - len(mined_mapping)),
        device=loss_args.teacher_logits.device
    )
    
    # Add logit mask to prevent invalid token predictions
    mined_teacher_logits = torch.index_select(
        loss_args.teacher_logits, 
        dim=-1, 
        index=mined_mapping_tensor
    ) + (loss_args.logit_mask_new if hasattr(loss_args, 'logit_mask_new') else 0)
    
    # Create one-hot probability distribution for ground truth tokens
    # Use high/low logit values as in the original implementation
    onehot_probs = F.one_hot(
        loss_args.batch["input_ids_new"][:, 1:], 
        loss_args.new_config.vocab_size
    )
    onehot_logits = (
        onehot_probs * 100.0 + 
        (1.0 - onehot_probs) * -100000.0 + 
        (loss_args.logit_mask_new if hasattr(loss_args, 'logit_mask_new') else 0)
    )
    
    # Get aligned logits
    aligned_teacher_logits = torch.gather(
        mined_teacher_logits,
        dim=1,
        index=alignment_matrix_b_last_only_index.unsqueeze(-1).expand(
            -1, -1, mined_teacher_logits.size(-1)
        )
    )
    
    aligned_student_kl_logits = torch.gather(
        loss_args.student_logits,
        dim=1,
        index=alignment_matrix_a_last_only_index.unsqueeze(-1).expand(
            -1, -1, loss_args.student_logits.size(-1)
        )
    )
    
    # Handle non-one-to-one alignments
    not_one_to_one_alignments = alignment_matrix_a[:, 1:] * ~one_to_one_mask[:, None, :]
    global_not_one_to_one_alignments = global_alignment_matrix_a[:, 1:] * ~global_one_to_one_mask[:, None, :]
    
    # Get indices for onehot alignments
    onehot_index = (
        not_one_to_one_alignments * 
        torch.arange(alignment_matrix_a.size(1) - 1, device=alignment_matrix_a.device)[None, :, None]
    ).max(-1)[0]
    
    onehot_mask = not_one_to_one_alignments.sum(-1) != 0
    global_onehot_mask = global_not_one_to_one_alignments.sum(-1) != 0
    
    # Extract aligned onehot logits and student logits
    aligned_onehot_logits = torch.gather(
        onehot_logits,
        dim=1,
        index=onehot_index.unsqueeze(-1).expand(-1, -1, onehot_logits.size(-1))
    )
    
    aligned_student_onehot_logits = torch.gather(
        loss_args.student_logits,
        dim=1,
        index=onehot_index.unsqueeze(-1).expand(-1, -1, loss_args.student_logits.size(-1))
    )
    kd_temp = args.baseline.kd_temp

    # Compute KL divergence for mined aligned tokens
    elementwise_mined_teacher_kl_loss = F.kl_div(
        F.log_softmax(aligned_student_kl_logits/ kd_temp, dim=-1),
        F.softmax(aligned_teacher_logits/ kd_temp, dim=-1),
        reduction='none'
    ).sum(-1)
    
    # Compute KL divergence for non-aligned tokens using onehot targets
    elementwise_onehot_kl_loss = F.kl_div(
        F.log_softmax(aligned_student_onehot_logits/ kd_temp, dim=-1),
        F.softmax(aligned_onehot_logits/ kd_temp, dim=-1),
        reduction='none'
    ).sum(-1)
    
    # Compute final loss, normalizing by global mask counts
    global_normalizer = (global_one_to_one_mask.sum() + global_onehot_mask.sum()).to(torch.float32) + EPSILON
    
    mined_kl_loss = (
        (elementwise_mined_teacher_kl_loss * one_to_one_mask).sum() +
        (elementwise_onehot_kl_loss * onehot_mask).sum()
    ) / global_normalizer
    
    return mined_kl_loss


def compute_uld_loss(args: Any, loss_args: LossArgs) -> Tensor:
    """Universal Logit Distillation loss."""
    # Extract student/teacher logits
    student_probs = loss_args.student_probs  
    teacher_probs = loss_args.teacher_probs  

    sorted_student_probs = torch.sort(student_probs, dim=-1, descending=True).values
    sorted_teacher_probs= torch.sort(teacher_probs, dim=-1, descending=True).values
    
    # Get masking
    vocab_gap = loss_args.new_config.vocab_size - loss_args.teacher_config.vocab_size
    if vocab_gap > 0:
        # Teacher vocab is smaller; pad teacher probabilities on the last dimension.
        sorted_teacher_probs = F.pad(sorted_teacher_probs, (0, vocab_gap), value=0)
    elif vocab_gap < 0:
        # Student vocab is smaller; pad student probabilities.
        sorted_student_probs = F.pad(sorted_student_probs, (0, -vocab_gap), value=0)
    
    token_loss = torch.abs(sorted_student_probs - sorted_teacher_probs).sum(-1)

    # Apply the attention mask and normalize.
    # The mask is expected to have shape [batch, seq_length] to match token_loss.
    attention_mask = loss_args.batch["attention_mask_new"]
    loss = (token_loss * attention_mask).mean() / attention_mask.mean()

    return loss

def compute_dskd_loss(args: Any, loss_args: LossArgs) -> Tensor:
    """Dual Space Knowledge Distillation loss."""
    # Extract probabilities
    student_probs = loss_args.student_probs[:, :-1]
    teacher_probs = loss_args.teacher_probs[:, :-1]
    
    # Get masked logprobs
    student_logprobs = loss_args.student_logprobs[:, :-1]
    teacher_logprobs = loss_args.teacher_logprobs[:, :-1]
    
    # Get masking
    loss_mask = loss_args.batch["loss_mask_new"][:, :-1]
    
    # Calculate dual space losses
    # Probability space (KL divergence)
    prob_loss = F.kl_div(
        torch.log(student_probs + EPSILON),
        teacher_probs,
        reduction='none'
    )
    
    # Log probability space (MSE)
    logprob_loss = F.mse_loss(
        student_logprobs,
        teacher_logprobs,
        reduction='none'
    )
    
    # Apply masking and weighted combination
    alpha = getattr(args, 'dskd_alpha', 0.5)
    masked_prob_loss = (prob_loss * loss_mask.unsqueeze(-1)).mean()
    masked_logprob_loss = (logprob_loss * loss_mask.unsqueeze(-1)).mean()
    
    combined_loss = alpha * masked_prob_loss + (1 - alpha) * masked_logprob_loss
    
    # Record individual losses
    loss_args.scalar_report["dskd_prob_loss"] = masked_prob_loss.item()
    loss_args.scalar_report["dskd_logprob_loss"] = masked_logprob_loss.item()
    
    return combined_loss


def compute_alm_side_path_loss(
    chunk_kind: str, 
    student_mapping: List[int], 
    teacher_mapping: List[int], 
    args: Any, 
    loss_args: LossArgs
) -> Tensor:
    """Compute Approximate Likelihood Matching for side paths.
    
    This performs ALM loss calculation on specific mappings of tokens
    between teacher and student models, allowing for targeted knowledge transfer.
    
    Args:
        chunk_kind: Type of chunking to use ('unconstrained', 'unbiased', or 'space')
        student_mapping: List of student token indices to align
        teacher_mapping: List of teacher token indices to align
        args: Configuration arguments
        loss_args: Loss function arguments
        
    Returns:
        Side path loss value
    """
    # Get the appropriate alignment matrices based on chunk kind
    if chunk_kind == "unconstrained":
        alignment_matrix_a = loss_args.batch["alignment_matrix_a_unconstrained"]
        alignment_matrix_b = loss_args.batch["alignment_matrix_b_unconstrained"]
        global_alignment_matrix_a = loss_args.global_batch["alignment_matrix_a_unconstrained"]
        global_alignment_matrix_b = loss_args.global_batch["alignment_matrix_b_unconstrained"]
    elif chunk_kind == "unbiased":
        alignment_matrix_a = loss_args.batch["alignment_matrix_a_unbiased"]
        alignment_matrix_b = loss_args.batch["alignment_matrix_b_unbiased"]
        global_alignment_matrix_a = loss_args.global_batch["alignment_matrix_a_unbiased"]
        global_alignment_matrix_b = loss_args.global_batch["alignment_matrix_b_unbiased"]
    elif chunk_kind == "space":
        alignment_matrix_a = loss_args.batch["alignment_matrix_a_space"]
        alignment_matrix_b = loss_args.batch["alignment_matrix_b_space"]
        global_alignment_matrix_a = loss_args.global_batch["alignment_matrix_a_space"]
        global_alignment_matrix_b = loss_args.global_batch["alignment_matrix_b_space"]
    else:
        raise ValueError(f"Unknown chunk kind: {chunk_kind}")
    
    # Apply loss masks to alignment matrices
    alignment_matrix_a = alignment_matrix_a[:, :-1] & loss_args.batch["loss_mask_new"][:, 1:, None]
    alignment_matrix_b = alignment_matrix_b[:, :-1] & loss_args.batch["loss_mask_original"][:, 1:, None]
    global_alignment_matrix_a = global_alignment_matrix_a[:, :-1] & loss_args.global_batch["loss_mask_new"][:, 1:, None]
    global_alignment_matrix_b = global_alignment_matrix_b[:, :-1] & loss_args.global_batch["loss_mask_original"][:, 1:, None]
    
    # Get last indices and masks
    alignment_matrix_b_last_only_index, _ = get_last_index_per_column(alignment_matrix_b)
    alignment_matrix_a_last_only_index, mask = get_last_index_per_column(alignment_matrix_a)
    _, global_mask = get_last_index_per_column(global_alignment_matrix_a)
    
    # Extract logprobs and logits for the specific token mappings
    batch_indices = torch.arange(alignment_matrix_a.size(0), device=alignment_matrix_a.device)[:, None]
    
    s_aligned_logprobs = torch.take_along_dim(
        loss_args.student_logprobs,
        alignment_matrix_a_last_only_index.unsqueeze(-1),
        dim=1
    )[:, :, student_mapping]
    
    t_aligned_logprobs = torch.take_along_dim(
        loss_args.teacher_logprobs,
        alignment_matrix_b_last_only_index.unsqueeze(-1),
        dim=1
    )[:, :, teacher_mapping]
    
    s_aligned_logits = torch.take_along_dim(
        loss_args.student_logits,
        alignment_matrix_a_last_only_index.unsqueeze(-1),
        dim=1
    )[:, :, student_mapping]
    
    t_aligned_logits = torch.take_along_dim(
        loss_args.teacher_logits,
        alignment_matrix_b_last_only_index.unsqueeze(-1),
        dim=1
    )[:, :, teacher_mapping]
    
    # Calculate probabilities from logprobs
    s_aligned_probs = torch.exp(s_aligned_logprobs)
    t_aligned_probs = torch.exp(t_aligned_logprobs)
    
    # Record probability mass statistics
    loss_args.scalar_report["student_side_path_aligned_pmass"] = (
        (s_aligned_probs.sum(dim=-1) * mask).mean() / (mask.mean() + EPSILON)
    )
    loss_args.scalar_report["teacher_side_path_aligned_pmass"] = (
        (t_aligned_probs.sum(dim=-1) * mask).mean() / (mask.mean() + EPSILON)
    )
    
    # Apply different distance functions based on configuration
    if args.side_path_distance_fn == "kl":
        # Calculate remainder probabilities
        s_remainder_probs = torch.clamp(1 - s_aligned_probs.sum(-1), min=EPSILON)
        t_remainder_probs = torch.clamp(1 - t_aligned_probs.sum(-1), min=EPSILON)
        
        # KL divergence calculation
        elementwise_loss = (
            (t_aligned_probs * (t_aligned_logprobs - s_aligned_logprobs)).sum(-1) + 
            (t_remainder_probs * (torch.log(t_remainder_probs) - torch.log(s_remainder_probs)))
        ) * mask
        
        side_path_loss = elementwise_loss.mean() / (global_mask.mean() + EPSILON)
        
    elif args.side_path_distance_fn == "reverse_kl":
        # Calculate remainder probabilities
        s_remainder_probs = torch.clamp(1 - s_aligned_probs.sum(-1), min=EPSILON)
        t_remainder_probs = torch.clamp(1 - t_aligned_probs.sum(-1), min=EPSILON)
        
        # Reverse KL divergence calculation
        elementwise_loss = (
            (s_aligned_probs * (s_aligned_logprobs - t_aligned_logprobs)).sum(-1) + 
            (s_remainder_probs * (torch.log(s_remainder_probs) - torch.log(t_remainder_probs)))
        ) * mask
        
        side_path_loss = elementwise_loss.mean() / (global_mask.mean() + EPSILON)
        
    elif args.side_path_distance_fn == "kl_subset":
        # Calculate normalized log probabilities within subset
        s_aligned_logprobs_subset = F.log_softmax(s_aligned_logits, dim=-1)
        t_aligned_logprobs_subset = F.log_softmax(t_aligned_logits, dim=-1)
        
        # KL divergence on subset
        elementwise_loss = (
            (torch.exp(t_aligned_logprobs_subset) * 
             (t_aligned_logprobs_subset - s_aligned_logprobs_subset)).sum(-1)
        ) * mask
        
        side_path_loss = elementwise_loss.mean() / (global_mask.mean() + EPSILON)
        
    elif args.side_path_distance_fn == "reverse_kl_subset":
        # Calculate normalized log probabilities within subset
        s_aligned_logprobs_subset = F.log_softmax(s_aligned_logits, dim=-1)
        t_aligned_logprobs_subset = F.log_softmax(t_aligned_logits, dim=-1)
        
        # Reverse KL divergence on subset
        elementwise_loss = (
            (torch.exp(s_aligned_logprobs_subset) * 
             (s_aligned_logprobs_subset - t_aligned_logprobs_subset)).sum(-1)
        ) * mask
        
        side_path_loss = elementwise_loss.mean() / (global_mask.mean() + EPSILON)
        
    elif args.side_path_distance_fn == "log_abs":
        # Mean absolute difference in log probability space
        elementwise_loss = (torch.abs(s_aligned_logprobs - t_aligned_logprobs).mean(-1) * mask)
        side_path_loss = elementwise_loss.mean() / (global_mask.mean() + EPSILON)
        
    elif args.side_path_distance_fn == "abs":
        # Mean absolute difference in probability space
        elementwise_loss = (torch.abs(s_aligned_probs - t_aligned_probs).sum(-1) * mask)
        side_path_loss = elementwise_loss.mean() / (global_mask.mean() + EPSILON)
        
    else:
        raise ValueError(f"Unknown side path distance function: {args.side_path_distance_fn}")
    
    return side_path_loss