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
    indices = torch.argmax(matrix_last_only.float(), dim=-1)
    mask = torch.max(matrix_last_only, dim=-1)[0]
    
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
        masked_loss = masked_loss / (shift_attention_mask.mean() + EPSILON)
    
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
    """Compute Approximate Likelihood Matching loss.
    
    Args:
        chunk_kind: Type of chunking to use ('unconstrained', 'unbiased', or 'space')
        args: Configuration arguments
        loss_args: Loss function arguments
        epsilon: Small constant for numerical stability
        
    Returns:
        ALM loss value
    """
    original_shift_labels = loss_args.batch["input_ids_original"][:, 1:]
    
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
    
    # Define the difference function to use
    if args.alm_diff_fn == "abs":
        diff_fn = lambda log_y_true, log_y_pred: torch.abs(log_y_true - log_y_pred)
    elif args.alm_diff_fn == "binary_ce":
        def binary_ce(log_y_true, log_y_pred):
            log_y_true = (log_y_true.float() / args.bce_temp) - epsilon
            log_y_pred = (log_y_pred.float() / args.bce_temp) - epsilon
            
            return -(
                torch.exp(log_y_true) * log_y_pred
                + (torch.expm1(log_y_true) * -1 * log1mexp(log_y_pred))
            )
        diff_fn = binary_ce
    elif args.alm_diff_fn == "reverse_binary_kl":
        def reverse_binary_kl(log_y_true, log_y_pred):
            log_y_true = (log_y_true.float() / args.bce_temp) - epsilon
            log_y_pred = (log_y_pred.float() / args.bce_temp) - epsilon
            
            return torch.exp(log_y_pred) * (log_y_pred - log_y_true) + (
                torch.expm1(log_y_pred) * -1 * (log1mexp(log_y_pred) - log1mexp(log_y_true))
            )
        diff_fn = reverse_binary_kl
    elif args.alm_diff_fn == "binary_kl_temp_limit":
        def binary_kl_temp_limit(log_y_true, log_y_pred):
            log_y_true = log_y_true - epsilon
            log_y_pred = log_y_pred - epsilon
            
            return (log_y_true - log_y_pred) + (
                log_y_true * torch.log(-log_y_pred) - log_y_true * torch.log(-log_y_true)
            )
        diff_fn = binary_kl_temp_limit
    elif args.alm_diff_fn == "abs_exp":
        def abs_exp(log_y_true, log_y_pred):
            log_y_true = (log_y_true.float() / args.bce_temp) - epsilon
            log_y_pred = (log_y_pred.float() / args.bce_temp) - epsilon
            
            return torch.abs(torch.exp(log_y_true) - torch.exp(log_y_pred))
        diff_fn = abs_exp
    else:
        raise NotImplementedError(f"Unknown diff function: {args.alm_diff_fn}")
    
    # Apply loss masks to alignment matrices
    alignment_matrix_a = alignment_matrix_a * loss_args.batch["loss_mask_new"][:, :, None]
    alignment_matrix_b = alignment_matrix_b * loss_args.batch["loss_mask_original"][:, :, None]
    global_alignment_matrix_a = global_alignment_matrix_a * loss_args.global_batch["loss_mask_new"][:, :, None]
    global_alignment_matrix_b = global_alignment_matrix_b * loss_args.global_batch["loss_mask_original"][:, :, None]
    
    # Get last indices and masks
    alignment_matrix_b_last_only_index, _ = get_last_index_per_column(alignment_matrix_b)
    alignment_matrix_a_last_only_index, mask = get_last_index_per_column(alignment_matrix_a)
    
    # Calculate chunk sums and alignment statistics
    student_chunk_sums = alignment_matrix_a.sum(dim=-1)
    teacher_chunk_sums = alignment_matrix_b.sum(dim=-1)
    
    # For global normalization
    _, global_mask = get_last_index_per_column(global_alignment_matrix_a)
    
    # Calculate average chunk lengths
    global_student_chunk_sums = global_alignment_matrix_a.sum(dim=-1)
    global_teacher_chunk_sums = global_alignment_matrix_b.sum(dim=-1)
    global_student_avg_chunk_lengths = (
        (global_student_chunk_sums * global_mask).sum() /
        (global_mask.sum() + EPSILON)
    )
    global_teacher_avg_chunk_lengths = (
        (global_teacher_chunk_sums * global_mask).sum() /
        (global_mask.sum() + EPSILON)
    )
    
    # Extract probabilities for each chunk
    batch_size, seq_len, vocab_size = loss_args.student_logprobs.shape
    
    # Gather teacher and student probabilities at chunk boundaries
    # Use fancy indexing to gather the last probability in each chunk
    batch_indices = torch.arange(batch_size, device=alignment_matrix_a.device)[:, None]
    seq_indices = torch.arange(seq_len, device=alignment_matrix_a.device)[None, :]
    
    # Log-probs for student and teacher per token in chunk
    gathered_student_logprobs = loss_args.student_logprobs[
        batch_indices, seq_indices, loss_args.batch["input_ids_new"][:, 1:]
    ]
    
    # Gather teacher log-probs per token in chunk
    gathered_teacher_original_logprobs = loss_args.teacher_logprobs[
        batch_indices, alignment_matrix_b_last_only_index, original_shift_labels
    ]
    
    # Apply the difference function to calculate token-level loss
    token_level_loss = diff_fn(
        gathered_teacher_original_logprobs,
        gathered_student_logprobs
    )
    
    # Apply masking and compute mean loss
    token_level_loss = token_level_loss * mask
    
    # Normalize by global mask mean for stability across different batch sizes
    alm_loss = token_level_loss.sum() / (global_mask.sum() + EPSILON)
    
    # Store statistics in scalar report
    loss_args.scalar_report[f"alm_loss_{chunk_kind}"] = alm_loss.item()
    loss_args.scalar_report[f"student_avg_chunk_lengths_{chunk_kind}"] = global_student_avg_chunk_lengths.item()
    loss_args.scalar_report[f"teacher_avg_chunk_lengths_{chunk_kind}"] = global_teacher_avg_chunk_lengths.item()
    
    return alm_loss


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
        batch_indices = torch.arange(t_hidden_state.size(0), device=t_hidden_state.device)[:, None]
        t_aligned_last_hidden_state = t_hidden_state[
            batch_indices, alignment_matrix_b_last_only_index
        ]
        s_aligned_last_hidden_state = s_hidden_state[
            batch_indices, alignment_matrix_a_last_only_index
        ]
        
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
            
            # Extract aligned attention states
            batch_indices = torch.arange(teacher_qkv.size(0), device=teacher_qkv.device)[:, None]
            t_aligned_qkv = teacher_qkv[
                batch_indices, alignment_matrix_b_last_only_index
            ]
            s_aligned_qkv = student_qkv[
                batch_indices, alignment_matrix_a_last_only_index
            ]
            
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
    
    # Compute KL divergence for mined aligned tokens
    elementwise_mined_teacher_kl_loss = F.kl_div(
        F.log_softmax(aligned_student_kl_logits, dim=-1),
        F.softmax(aligned_teacher_logits, dim=-1),
        reduction='none'
    ).sum(-1)
    
    # Compute KL divergence for non-aligned tokens using onehot targets
    elementwise_onehot_kl_loss = F.kl_div(
        F.log_softmax(aligned_student_onehot_logits, dim=-1),
        F.softmax(aligned_onehot_logits, dim=-1),
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
        sorted_student_probs = F.pad(sorted_student_probs, (0, abs(vocab_gap)), value=0)
    
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