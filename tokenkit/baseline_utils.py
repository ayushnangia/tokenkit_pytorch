"""
Baseline utilities for tokenkit PyTorch implementation.

This module provides utility functions for baseline methods like
MinED (Minimum Edit Distance) mapping and knowledge distillation.
"""

import math
import multiprocessing
from functools import partial
from typing import Dict, List, Tuple, Union, Optional, Any

import editdistance
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def _compute_edit_distance(token: str, sorted_original_vocab: List[str]) -> Tuple[str, str, int]:
    """Compute minimum edit distance for a token against a vocabulary.
    
    Args:
        token: Token to compute edit distance for
        sorted_original_vocab: Sorted list of original tokens
        
    Returns:
        Tuple of (token, best_match, min_edit_distance)
    """
    min_edit_distance = math.inf
    best_match = None

    # Optimization: Start search from beginning or end based on token length
    closer_to_start = len(token) < len(
        sorted_original_vocab[int(len(sorted_original_vocab) / 2)]
    )

    if closer_to_start:
        candidates = sorted_original_vocab
    else:
        candidates = reversed(sorted_original_vocab)

    for original_token in candidates:
        # Early stopping based on length difference
        if closer_to_start:
            # Tokens only get longer as we go through the list
            if len(original_token) - len(token) >= min_edit_distance:
                break
            if len(token) - len(original_token) >= min_edit_distance:
                continue
        else:
            # Tokens only get shorter as we go through the list in reverse
            if len(token) - len(original_token) >= min_edit_distance:
                break
            if len(original_token) - len(token) >= min_edit_distance:
                continue

        # Compute edit distance
        edit_distance = editdistance.eval(token, original_token)
        if edit_distance < min_edit_distance:
            min_edit_distance = edit_distance
            best_match = original_token

    return token, best_match, min_edit_distance


def compute_mined_mapping(
    tokenizer_original: Any,
    tokenizer_new: Any,
    num_workers: int = 1,
    chunksize: int = 500
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Compute a minimum edit distance mapping between tokenizers.
    
    This function creates a mapping from new tokenizer IDs to original tokenizer IDs
    by finding the closest match for each token based on edit distance.
    
    Args:
        tokenizer_original: Original tokenizer object
        tokenizer_new: New tokenizer object
        num_workers: Number of parallel workers to use
        chunksize: Chunk size for multiprocessing
        
    Returns:
        Tuple of (mapping array, edit distances dictionary)
    """
    original_vocab = tokenizer_original.get_vocab()
    new_vocab = tokenizer_new.get_vocab()

    # Initialize mapping and edit distances
    mapping = np.zeros(len(tokenizer_new), dtype=np.int32)
    edit_distances = {}

    # Find intersection and completion (tokens in new but not in original)
    intersection = [token for token in new_vocab.keys() if token in original_vocab]
    completion = [token for token in new_vocab.keys() if token not in original_vocab]
    sorted_completion = sorted(completion, key=lambda x: len(x))
    sorted_original_vocab = sorted(original_vocab.keys(), key=lambda x: len(x))

    # For tokens in both vocabularies, create identity mapping
    for token in intersection:
        mapping[new_vocab[token]] = original_vocab[token]
        edit_distances[token] = 0

    # For remaining tokens, compute edit distance mappings in parallel
    with multiprocessing.Pool(max(num_workers, 1)) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(
                    partial(
                        _compute_edit_distance,
                        sorted_original_vocab=sorted_original_vocab,
                    ),
                    sorted_completion,
                    chunksize=chunksize,
                ),
                desc="Computing MinED mapping",
                total=len(sorted_completion),
            )
        )

    # Process results
    for token, best_match, min_edit_distance in results:
        mapping[new_vocab[token]] = original_vocab[best_match]
        edit_distances[token] = min_edit_distance

    return mapping, edit_distances


def compute_forward_kl_divergence(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    kd_temp: float = 1.0,
    padding_id: Optional[int] = None,
    tea_temp: Optional[float] = None,
    reduction: str = "sum",
    log: Optional[Dict[str, Any]] = None,
    use_tea_temp: bool = False,
) -> torch.Tensor:
    """Compute forward KL divergence for knowledge distillation.
    
    Args:
        logits: Student model logits
        teacher_logits: Teacher model logits
        target: Target indices for padding masking
        kd_temp: Temperature for knowledge distillation
        padding_id: Padding token ID for masking
        tea_temp: Teacher temperature (if different from kd_temp)
        reduction: Reduction method ('sum' or 'none')
        log: Dictionary to log metrics
        use_tea_temp: Whether to use tea_temp for teacher logits
        
    Returns:
        KL divergence loss
    """
    # Apply temperature scaling
    logits = logits / kd_temp
    teacher_logits = teacher_logits / kd_temp
    if use_tea_temp and tea_temp is not None:
        teacher_logits = teacher_logits / tea_temp

    # Compute log probabilities and probabilities
    lprobs = F.log_softmax(logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    teacher_lprobs = F.log_softmax(teacher_logits, dim=-1)
    
    # Compute KL divergence
    kld = teacher_probs * (teacher_lprobs - lprobs)
    inf_mask = torch.isinf(logits)
    kld = torch.where(inf_mask, torch.zeros_like(kld), kld).sum(-1)

    # Apply reduction if requested
    if reduction == "sum" and target is not None and padding_id is not None:
        pad_mask = target == padding_id
        kld = torch.where(pad_mask, torch.zeros_like(kld), kld)
        kld = kld.sum()

        if log is not None:
            log["forward_kl"] = kld.item()

    return kld


def compute_reverse_kl_divergence(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    kd_temp: float = 1.0,
    padding_id: Optional[int] = None,
    tea_temp: Optional[float] = None,
    reduction: str = "sum",
    log: Optional[Dict[str, Any]] = None,
    use_tea_temp: bool = False,
) -> torch.Tensor:
    """Compute reverse KL divergence for knowledge distillation.
    
    Args:
        logits: Student model logits
        teacher_logits: Teacher model logits
        target: Target indices for padding masking
        kd_temp: Temperature for knowledge distillation
        padding_id: Padding token ID for masking
        tea_temp: Teacher temperature (if different from kd_temp)
        reduction: Reduction method ('sum' or 'none')
        log: Dictionary to log metrics
        use_tea_temp: Whether to use tea_temp for teacher logits
        
    Returns:
        Reverse KL divergence loss
    """
    # Apply temperature scaling
    logits = logits / kd_temp
    teacher_logits = teacher_logits / kd_temp
    if use_tea_temp and tea_temp is not None:
        teacher_logits = teacher_logits / tea_temp

    # Compute probabilities and log probabilities
    probs = F.softmax(logits, dim=-1)
    lprobs = F.log_softmax(logits, dim=-1)
    teacher_lprobs = F.log_softmax(teacher_logits, dim=-1)
    
    # Compute KL divergence
    kld = probs * (lprobs - teacher_lprobs)
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    kld = torch.where(inf_mask, torch.zeros_like(kld), kld).sum(-1)

    # Apply reduction if requested
    if reduction == "sum" and target is not None and padding_id is not None:
        pad_mask = target == padding_id
        kld = torch.where(pad_mask, torch.zeros_like(kld), kld)
        kld = kld.sum()

        if log is not None:
            log["reverse_kl"] = kld.item()

    return kld


def compute_adaptive_kl_divergence(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    kd_temp: float = 1.0,
    padding_id: Optional[int] = None,
    alpha: float = 0.5,
    tea_temp: Optional[float] = None,
    reduction: str = "sum",
    log: Optional[Dict[str, Any]] = None,
    use_tea_temp: bool = False,
) -> torch.Tensor:
    """Compute adaptive KL divergence for knowledge distillation.
    
    This adaptively combines forward and reverse KL divergence based
    on the probability gap between teacher and student distributions.
    
    Args:
        logits: Student model logits
        teacher_logits: Teacher model logits
        target: Target indices for padding masking
        kd_temp: Temperature for knowledge distillation
        padding_id: Padding token ID for masking
        alpha: Threshold for adaptive weighting
        tea_temp: Teacher temperature (if different from kd_temp)
        reduction: Reduction method ('sum' or 'none')
        log: Dictionary to log metrics
        use_tea_temp: Whether to use tea_temp for teacher logits
        
    Returns:
        Adaptive KL divergence loss
    """
    # Compute probabilities
    probs = F.softmax(logits / kd_temp, dim=-1).float()
    if use_tea_temp and tea_temp is not None:
        teacher_probs = F.softmax(teacher_logits / tea_temp / kd_temp, dim=-1).float()
    else:
        teacher_probs = F.softmax(teacher_logits / kd_temp, dim=-1).float()

    # Sort probabilities and compute gaps
    sorted_teacher_probs, sorted_indices = torch.sort(teacher_probs, dim=-1)
    sorted_probs = torch.gather(probs, dim=-1, index=sorted_indices)
    
    gap = torch.abs(sorted_teacher_probs - sorted_probs)
    cum_teacher_probs = torch.cumsum(sorted_teacher_probs, dim=-1)
    tail_mask = (cum_teacher_probs < alpha).float()
    
    # Compute head and tail gap weights with gradient stopping
    with torch.no_grad():
        g_head = torch.sum(gap * (1 - tail_mask), dim=-1)
        g_tail = torch.sum(gap * tail_mask, dim=-1)

    # Compute forward and reverse KL divergence
    fkl = compute_forward_kl_divergence(
        logits,
        teacher_logits,
        target,
        kd_temp,
        padding_id,
        tea_temp=tea_temp,
        reduction="none",
        use_tea_temp=use_tea_temp,
    )
    
    rkl = compute_reverse_kl_divergence(
        logits,
        teacher_logits,
        target,
        kd_temp,
        padding_id,
        tea_temp=tea_temp,
        reduction="none",
        use_tea_temp=use_tea_temp,
    )

    # Combine adaptively
    g_sum = g_head + g_tail
    g_sum = torch.clamp(g_sum, min=1e-10)  # Avoid division by zero
    akl = (g_head / g_sum) * fkl + (g_tail / g_sum) * rkl

    # Apply reduction if requested
    if reduction == "sum" and target is not None and padding_id is not None:
        pad_mask = target == padding_id
        akl = torch.where(pad_mask, torch.zeros_like(akl), akl)
        akl = akl.sum()

        if log is not None:
            log["adaptive_kl"] = akl.item()

    return akl


def compute_skewed_kl_divergence(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    kd_temp: float = 1.0,
    padding_id: Optional[int] = None,
    skew_lambda: float = 0.5,
    tea_temp: Optional[float] = None,
    reduction: str = "sum",
    log: Optional[Dict[str, Any]] = None,
    use_tea_temp: bool = False,
    reverse: bool = False,
    epsilon: float = 1e-9,
) -> torch.Tensor:
    """Compute skewed KL divergence for knowledge distillation.
    
    This uses a mixture of student and teacher probabilities to create
    a skewed target distribution for KL divergence.
    
    Args:
        logits: Student model logits
        teacher_logits: Teacher model logits
        target: Target indices for padding masking
        kd_temp: Temperature for knowledge distillation
        padding_id: Padding token ID for masking
        skew_lambda: Skew weight (0-1)
        tea_temp: Teacher temperature (if different from kd_temp)
        reduction: Reduction method ('sum' or 'none')
        log: Dictionary to log metrics
        use_tea_temp: Whether to use tea_temp for teacher logits
        reverse: Whether to compute reverse skewed KL divergence
        epsilon: Small constant for numerical stability
        
    Returns:
        Skewed KL divergence loss
    """
    # Apply temperature scaling
    logits = logits / kd_temp
    teacher_logits = teacher_logits / kd_temp
    if use_tea_temp and tea_temp is not None:
        teacher_logits = teacher_logits / tea_temp

    # Compute probabilities
    student_probs = F.softmax(logits, dim=-1).float()
    teacher_probs = F.softmax(teacher_logits, dim=-1).float()
    
    if reverse:
        # Reverse skewed KL: student vs mixed (mostly student)
        mixed_probs = (1 - skew_lambda) * teacher_probs + skew_lambda * student_probs
        mixed_lprobs = torch.log(mixed_probs + epsilon)
        student_lprobs = F.log_softmax(logits, dim=-1).float()
        kld = student_probs * (student_lprobs - mixed_lprobs)
        
        log_key = "skewed_reverse_kl"
    else:
        # Forward skewed KL: teacher vs mixed (mostly teacher)
        mixed_probs = skew_lambda * teacher_probs + (1 - skew_lambda) * student_probs
        mixed_lprobs = torch.log(mixed_probs + epsilon)
        teacher_lprobs = F.log_softmax(teacher_logits, dim=-1).float()
        kld = teacher_probs * (teacher_lprobs - mixed_lprobs)
        
        log_key = "skewed_forward_kl"
    
    # Handle infinity and sum across vocabulary
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    kld = torch.where(inf_mask, torch.zeros_like(kld), kld).sum(-1)

    # Apply reduction if requested
    if reduction == "sum" and target is not None and padding_id is not None:
        pad_mask = target == padding_id
        kld = torch.where(pad_mask, torch.zeros_like(kld), kld)
        kld = kld.sum()

        if log is not None:
            log[log_key] = kld.item()

    return kld


def compute_js_divergence(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    kd_temp: float = 1.0,
    tea_temp: Optional[float] = None,
    padding_id: Optional[int] = None,
    reduction: str = "sum",
    log: Optional[Dict[str, Any]] = None,
    use_tea_temp: bool = False,
    epsilon: float = 1e-9,
) -> torch.Tensor:
    """Compute Jensen-Shannon divergence for knowledge distillation.
    
    The JS divergence is a symmetric measure based on the average of
    KL divergences between distributions and their mixture.
    
    Args:
        logits: Student model logits
        teacher_logits: Teacher model logits
        target: Target indices for padding masking
        kd_temp: Temperature for knowledge distillation
        tea_temp: Teacher temperature (if different from kd_temp)
        padding_id: Padding token ID for masking
        reduction: Reduction method ('sum' or 'none')
        log: Dictionary to log metrics
        use_tea_temp: Whether to use tea_temp for teacher logits
        epsilon: Small constant for numerical stability
        
    Returns:
        Jensen-Shannon divergence loss
    """
    # Apply temperature scaling
    logits = logits / kd_temp
    teacher_logits = teacher_logits / kd_temp
    if use_tea_temp and tea_temp is not None:
        teacher_logits = teacher_logits / tea_temp

    # Compute probabilities
    probs = F.softmax(logits, dim=-1).float()
    teacher_probs = F.softmax(teacher_logits, dim=-1).float()
    
    # Compute mixture
    m_probs = (probs + teacher_probs) / 2
    
    # Compute log probabilities
    lprobs = torch.log(probs + epsilon)
    teacher_lprobs = torch.log(teacher_probs + epsilon)
    m_lprobs = torch.log(m_probs + epsilon)
    
    # Compute KL divergences
    kld1 = teacher_probs * (teacher_lprobs - m_lprobs)
    kld2 = probs * (lprobs - m_lprobs)
    kld = (kld1 + kld2) / 2
    
    # Apply reduction if requested
    if reduction == "sum" and target is not None and padding_id is not None:
        pad_mask = target == padding_id
        kld = torch.where(pad_mask, torch.zeros_like(kld), kld)
        kld = kld.sum()

        if log is not None:
            log["js_div"] = kld.item()

    return kld