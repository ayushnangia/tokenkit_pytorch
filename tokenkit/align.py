import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from tokenkit.byteify import ByteifyTokenizer, load_byteify_tokenizer

logger = logging.getLogger(__name__)


def get_alignment_indices(
    tokens_teacher: List[str],
    tokens_student: List[str],
    attention_mask_teacher: Union[np.ndarray, torch.Tensor],
    attention_mask_student: Union[np.ndarray, torch.Tensor],
    tokenizer_teacher: ByteifyTokenizer,
    tokenizer_student: ByteifyTokenizer,
    check: bool = True,
) -> Tuple[List[Tuple[int, int, int, int]], List[str], List[str], List[bool], List[bool]]:
    """Get alignment indices between teacher and student tokens.
    
    This function computes alignment indices between teacher and student tokens,
    which represent chunks of text that should be matched between the two tokenizations.
    
    Args:
        tokens_teacher: List of teacher tokens
        tokens_student: List of student tokens
        attention_mask_teacher: Attention mask for teacher tokens
        attention_mask_student: Attention mask for student tokens
        tokenizer_teacher: Teacher tokenizer
        tokenizer_student: Student tokenizer
        check: Whether to check that aligned tokens are equal
        
    Returns:
        Tuple of:
        - List of alignment indices (start_i, end_i, start_j, end_j)
        - Normalized teacher tokens
        - Normalized student tokens
        - Special tokens mask for teacher tokens
        - Special tokens mask for student tokens
    """
    # Convert torch tensors to numpy arrays if needed
    if isinstance(attention_mask_teacher, torch.Tensor):
        attention_mask_teacher = attention_mask_teacher.cpu().numpy()
    if isinstance(attention_mask_student, torch.Tensor):
        attention_mask_student = attention_mask_student.cpu().numpy()
    
    # Create maps for token replacements
    replacements_teacher = {}
    for k, v in tokenizer_teacher.model_kind_cls.replacements.items():
        if v is None:
            continue
        
        if tokenizer_student.model_kind_cls.replacements[k] is not None:
            replacements_teacher[tuple(v)] = [
                "S" * len(tokenizer_student.model_kind_cls.replacements[k])
            ] * len(v)
        else:
            replacements_teacher[tuple(v)] = ""
    
    replacements_student = {}
    for k, v in tokenizer_student.model_kind_cls.replacements.items():
        if v is None:
            continue
        
        if tokenizer_teacher.model_kind_cls.replacements[k] is not None:
            replacements_student[tuple(v)] = [
                "S" * len(tokenizer_teacher.model_kind_cls.replacements[k])
            ] * len(v)
        else:
            replacements_student[tuple(v)] = ""
    
    # Helper function to get replacement for a token sequence
    def get_replacement(tokens, index, replacements):
        for k, v in replacements.items():
            if index + len(k) <= len(tokens) and tuple(tokens[index:index+len(k)]) == k:
                return k, v
        
        return None
    
    # Initialize lists for normalized tokens and masks
    normalized_tokens_teacher = []
    normalized_tokens_student = []
    special_tokens_mask_teacher = []
    special_tokens_mask_student = []
    
    # Initialize indices
    start_i = 0
    start_j = 0
    i = 0
    j = 0
    
    # Initialize cumulative token lengths
    cum_length_teacher = 0
    cum_length_student = 0
    cum_lengths_teacher_dict = {}
    cum_lengths_student_dict = {}
    
    # Initialize alignment indices list
    alignment_indices = []
    
    # Main alignment loop
    while i < len(tokens_teacher) or j < len(tokens_student):
        # Skip padded tokens
        if i < len(tokens_teacher) and not attention_mask_teacher[i]:
            i += 1
            continue
        if j < len(tokens_student) and not attention_mask_student[j]:
            j += 1
            continue
        
        # Break if we've reached the end of either sequence
        if i == len(tokens_teacher) or j == len(tokens_student):
            break
        
        # Get replacements for special tokens
        r_teacher = get_replacement(tokens_teacher, i, replacements_teacher)
        r_student = get_replacement(tokens_student, j, replacements_student)
        
        skipped_align = False
        
        # Handle cases where special tokens need special handling
        if r_teacher is not None and r_teacher[1] == "":
            normalized_tokens_teacher.append("")
            special_tokens_mask_teacher.append(True)
            i += 1
            skipped_align = True
        if r_student is not None and r_student[1] == "":
            normalized_tokens_student.append("")
            special_tokens_mask_student.append(True)
            j += 1
            skipped_align = True
        if r_teacher is not None and r_student is not None and not skipped_align:
            normalized_tokens_teacher.extend(r_teacher[1])
            normalized_tokens_student.extend(r_student[1])
            special_tokens_mask_teacher.extend([True] * len(r_teacher[1]))
            special_tokens_mask_student.extend([True] * len(r_student[1]))
            i += len(r_teacher[0])
            j += len(r_student[0])
            skipped_align = True
            alignment_indices.append((start_i, i, start_j, j))
            start_i = i
            start_j = j
        
        # Handle regular tokens by aligning based on cumulative character lengths
        if not skipped_align:
            cum_length_teacher = cum_lengths_teacher_dict.get(i - 1, 0) + len(tokens_teacher[i])
            cum_lengths_teacher_dict[i] = cum_length_teacher
            cum_length_student = cum_lengths_student_dict.get(j - 1, 0) + len(tokens_student[j])
            cum_lengths_student_dict[j] = cum_length_student
            
            if cum_length_teacher == cum_length_student:
                # Cumulative lengths are equal, add both tokens
                normalized_tokens_teacher.append(tokens_teacher[i])
                normalized_tokens_student.append(tokens_student[j])
                special_tokens_mask_teacher.append(False)
                special_tokens_mask_student.append(False)
                i += 1
                j += 1
                alignment_indices.append((start_i, i, start_j, j))
                start_i = i
                start_j = j
            elif cum_length_teacher < cum_length_student:
                # Teacher token is shorter, add it and continue
                normalized_tokens_teacher.append(tokens_teacher[i])
                special_tokens_mask_teacher.append(False)
                i += 1
            elif cum_length_teacher > cum_length_student:
                # Student token is shorter, add it and continue
                normalized_tokens_student.append(tokens_student[j])
                special_tokens_mask_student.append(False)
                j += 1
    
    # Pad normalized tokens to match original lengths
    while len(normalized_tokens_teacher) < len(tokens_teacher):
        normalized_tokens_teacher.append("")
        special_tokens_mask_teacher.append(True)
    
    while len(normalized_tokens_student) < len(tokens_student):
        normalized_tokens_student.append("")
        special_tokens_mask_student.append(True)
    
    # Verify that aligned chunks are equal if requested
    if check:
        for start_i, end_i, start_j, end_j in alignment_indices:
            try:
                assert "".join(normalized_tokens_teacher[start_i:end_i]) == "".join(
                    normalized_tokens_student[start_j:end_j]
                )
            except AssertionError:
                logger.warning(
                    f"Alignment failed: {normalized_tokens_teacher[start_i:end_i]} != "
                    f"{normalized_tokens_student[start_j:end_j]}"
                )
                raise
    
    return (
        alignment_indices,
        normalized_tokens_teacher,
        normalized_tokens_student,
        special_tokens_mask_teacher,
        special_tokens_mask_student,
    )


def get_unconstrained_alignments(
    input_ids_teacher: Union[np.ndarray, torch.Tensor],
    input_ids_student: Union[np.ndarray, torch.Tensor],
    attention_mask_teacher: Union[np.ndarray, torch.Tensor],
    attention_mask_student: Union[np.ndarray, torch.Tensor],
    tokenizer_teacher: ByteifyTokenizer,
    tokenizer_student: ByteifyTokenizer,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """Get unconstrained alignment matrices between teacher and student tokens.
    
    Unconstrained alignments are based solely on character matches,
    without any constraints on token boundaries.
    
    Args:
        input_ids_teacher: Teacher input token IDs
        input_ids_student: Student input token IDs
        attention_mask_teacher: Teacher attention mask
        attention_mask_student: Student attention mask
        tokenizer_teacher: Teacher tokenizer
        tokenizer_student: Student tokenizer
        
    Returns:
        Tuple of (student alignment matrix, teacher alignment matrix)
    """
    # Convert to numpy arrays for processing if needed
    is_torch_input = isinstance(input_ids_teacher, torch.Tensor)
    if is_torch_input:
        input_ids_teacher_np = input_ids_teacher.cpu().numpy()
        input_ids_student_np = input_ids_student.cpu().numpy()
        attention_mask_teacher_np = attention_mask_teacher.cpu().numpy()
        attention_mask_student_np = attention_mask_student.cpu().numpy()
    else:
        input_ids_teacher_np = input_ids_teacher
        input_ids_student_np = input_ids_student
        attention_mask_teacher_np = attention_mask_teacher
        attention_mask_student_np = attention_mask_student
    
    # Convert attention masks to boolean
    attention_mask_teacher_np = attention_mask_teacher_np.astype(bool)
    attention_mask_student_np = attention_mask_student_np.astype(bool)
    
    # Get batch dimensions
    batch_size = input_ids_teacher_np.shape[0]
    shared_length = min(input_ids_teacher_np.shape[1], input_ids_student_np.shape[1])
    
    # Initialize alignment matrices
    alignment_matrix_teacher = np.zeros(
        (batch_size, input_ids_teacher_np.shape[1], shared_length), dtype=bool
    )
    alignment_matrix_student = np.zeros(
        (batch_size, input_ids_student_np.shape[1], shared_length), dtype=bool
    )
    
    # Process each example in the batch
    for example_index in range(batch_size):
        # Convert token IDs to tokens
        tokens_teacher = tokenizer_teacher.convert_ids_to_tokens(
            input_ids_teacher_np[example_index]
        )
        tokens_student = tokenizer_student.convert_ids_to_tokens(
            input_ids_student_np[example_index]
        )
        
        # Get alignment indices
        alignment_indices, normalized_tokens_teacher, normalized_tokens_student, _, _ = get_alignment_indices(
            tokens_teacher,
            tokens_student,
            attention_mask_teacher_np[example_index],
            attention_mask_student_np[example_index],
            tokenizer_teacher,
            tokenizer_student,
        )
        
        # Create masks for valid tokens
        teacher_mask = np.array([len(token) > 0 for token in normalized_tokens_teacher])
        student_mask = np.array([len(token) > 0 for token in normalized_tokens_student])
        
        # Fill alignment matrices
        chunk_idx = 0
        for start_i, end_i, start_j, end_j in alignment_indices:
            alignment_matrix_teacher[example_index, start_i:end_i, chunk_idx] = True
            alignment_matrix_student[example_index, start_j:end_j, chunk_idx] = True
            chunk_idx += 1
        
        # Apply token masks to alignment matrices
        alignment_matrix_teacher[example_index, ~teacher_mask, :] = False
        alignment_matrix_student[example_index, ~student_mask, :] = False
    
    # Convert back to PyTorch tensors if input was PyTorch tensors
    if is_torch_input:
        device = input_ids_teacher.device
        return (
            torch.tensor(alignment_matrix_student, dtype=torch.bool, device=device),
            torch.tensor(alignment_matrix_teacher, dtype=torch.bool, device=device)
        )
    else:
        return alignment_matrix_student, alignment_matrix_teacher


def get_space_alignments(
    input_ids_teacher: Union[np.ndarray, torch.Tensor],
    input_ids_student: Union[np.ndarray, torch.Tensor],
    attention_mask_teacher: Union[np.ndarray, torch.Tensor],
    attention_mask_student: Union[np.ndarray, torch.Tensor],
    tokenizer_teacher: ByteifyTokenizer,
    tokenizer_student: ByteifyTokenizer,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """Get space-constrained alignment matrices between teacher and student tokens.
    
    Space alignments are additionally constrained by whitespace boundaries,
    aligning chunks between spaces or special tokens.
    
    Args:
        input_ids_teacher: Teacher input token IDs
        input_ids_student: Student input token IDs
        attention_mask_teacher: Teacher attention mask
        attention_mask_student: Student attention mask
        tokenizer_teacher: Teacher tokenizer
        tokenizer_student: Student tokenizer
        
    Returns:
        Tuple of (student alignment matrix, teacher alignment matrix)
    """
    # Convert to numpy arrays for processing if needed
    is_torch_input = isinstance(input_ids_teacher, torch.Tensor)
    if is_torch_input:
        input_ids_teacher_np = input_ids_teacher.cpu().numpy()
        input_ids_student_np = input_ids_student.cpu().numpy()
        attention_mask_teacher_np = attention_mask_teacher.cpu().numpy()
        attention_mask_student_np = attention_mask_student.cpu().numpy()
    else:
        input_ids_teacher_np = input_ids_teacher
        input_ids_student_np = input_ids_student
        attention_mask_teacher_np = attention_mask_teacher
        attention_mask_student_np = attention_mask_student
    
    # Convert attention masks to boolean
    attention_mask_teacher_np = attention_mask_teacher_np.astype(bool)
    attention_mask_student_np = attention_mask_student_np.astype(bool)
    
    # Get batch dimensions
    batch_size = input_ids_teacher_np.shape[0]
    shared_length = min(input_ids_teacher_np.shape[1], input_ids_student_np.shape[1])
    
    # Initialize alignment matrices
    alignment_matrix_teacher = np.zeros(
        (batch_size, input_ids_teacher_np.shape[1], shared_length), dtype=bool
    )
    alignment_matrix_student = np.zeros(
        (batch_size, input_ids_student_np.shape[1], shared_length), dtype=bool
    )
    
    # Process each example in the batch
    for example_index in range(batch_size):
        # Convert token IDs to tokens
        tokens_teacher = tokenizer_teacher.convert_ids_to_tokens(
            input_ids_teacher_np[example_index]
        )
        tokens_student = tokenizer_student.convert_ids_to_tokens(
            input_ids_student_np[example_index]
        )
        
        # Get alignment indices
        alignment_indices, normalized_tokens_teacher, normalized_tokens_student, special_tokens_mask_teacher, special_tokens_mask_student = get_alignment_indices(
            tokens_teacher,
            tokens_student,
            attention_mask_teacher_np[example_index],
            attention_mask_student_np[example_index],
            tokenizer_teacher,
            tokenizer_student,
        )
        
        # Create masks for valid tokens
        teacher_mask = np.array([len(token) > 0 for token in normalized_tokens_teacher])
        student_mask = np.array([len(token) > 0 for token in normalized_tokens_student])
        
        # Create masks for tokens that start with a space
        teacher_starts_with_space = np.array(
            [len(token) > 0 and token[0] == "Ġ" for token in normalized_tokens_teacher]
        )
        student_starts_with_space = np.array(
            [len(token) > 0 and token[0] == "Ġ" for token in normalized_tokens_student]
        )
        
        # Fill alignment matrices with space-constrained chunks
        chunk_idx = 0
        for start_i, end_i, start_j, end_j in alignment_indices:
            alignment_matrix_teacher[example_index, start_i:end_i, chunk_idx] = True
            alignment_matrix_student[example_index, start_j:end_j, chunk_idx] = True
            
            # Start a new chunk if:
            # 1. Both next tokens start with a space, or
            # 2. The current chunk ends with a special token
            if (
                end_i < len(normalized_tokens_teacher)
                and teacher_starts_with_space[end_i]
                and end_j < len(normalized_tokens_student)
                and student_starts_with_space[end_j]
            ) or (
                special_tokens_mask_teacher[end_i - 1]
                or special_tokens_mask_student[end_j - 1]
            ):
                # Verify that special tokens match across tokenizers
                if special_tokens_mask_teacher[end_i - 1] or special_tokens_mask_student[end_j - 1]:
                    assert (
                        special_tokens_mask_student[end_j - 1]
                        == special_tokens_mask_teacher[end_i - 1]
                    )
                chunk_idx += 1
        
        # Apply token masks to alignment matrices
        alignment_matrix_teacher[example_index, ~teacher_mask, :] = False
        alignment_matrix_student[example_index, ~student_mask, :] = False
    
    # Convert back to PyTorch tensors if input was PyTorch tensors
    if is_torch_input:
        device = input_ids_teacher.device
        return (
            torch.tensor(alignment_matrix_student, dtype=torch.bool, device=device),
            torch.tensor(alignment_matrix_teacher, dtype=torch.bool, device=device)
        )
    else:
        return alignment_matrix_student, alignment_matrix_teacher


def get_unbiased_alignments(
    input_ids_teacher: Union[np.ndarray, torch.Tensor],
    input_ids_student: Union[np.ndarray, torch.Tensor],
    attention_mask_teacher: Union[np.ndarray, torch.Tensor],
    attention_mask_student: Union[np.ndarray, torch.Tensor],
    tokenizer_teacher: ByteifyTokenizer,
    tokenizer_student: ByteifyTokenizer,
    pair_data: Tuple,
    bias_threshold: float,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """Get unbiased alignment matrices between teacher and student tokens.
    
    Unbiased alignments are constrained by token-pair bias matrices,
    which indicate how biased certain token pairs are.
    
    Args:
        input_ids_teacher: Teacher input token IDs
        input_ids_student: Student input token IDs
        attention_mask_teacher: Teacher attention mask
        attention_mask_student: Student attention mask
        tokenizer_teacher: Teacher tokenizer
        tokenizer_student: Student tokenizer
        pair_data: Tuple of (bias1_matrix, bias2_matrix, teacher_token_probs, student_token_probs)
        bias_threshold: Threshold for considering a token pair biased
        
    Returns:
        Tuple of (student alignment matrix, teacher alignment matrix)
    """
    bias1_matrix, bias2_matrix, _, _ = pair_data
    
    # Convert to numpy arrays for processing if needed
    is_torch_input = isinstance(input_ids_teacher, torch.Tensor)
    if is_torch_input:
        input_ids_teacher_np = input_ids_teacher.cpu().numpy()
        input_ids_student_np = input_ids_student.cpu().numpy()
        attention_mask_teacher_np = attention_mask_teacher.cpu().numpy()
        attention_mask_student_np = attention_mask_student.cpu().numpy()
    else:
        input_ids_teacher_np = input_ids_teacher
        input_ids_student_np = input_ids_student
        attention_mask_teacher_np = attention_mask_teacher
        attention_mask_student_np = attention_mask_student
    
    # Convert attention masks to boolean
    attention_mask_teacher_np = attention_mask_teacher_np.astype(bool)
    attention_mask_student_np = attention_mask_student_np.astype(bool)
    
    # Get batch dimensions
    batch_size = input_ids_teacher_np.shape[0]
    shared_length = min(input_ids_teacher_np.shape[1], input_ids_student_np.shape[1])
    
    # Initialize alignment matrices
    alignment_matrix_teacher = np.zeros(
        (batch_size, input_ids_teacher_np.shape[1], shared_length), dtype=bool
    )
    alignment_matrix_student = np.zeros(
        (batch_size, input_ids_student_np.shape[1], shared_length), dtype=bool
    )
    
    # Get bias matrix dimensions
    teacher_length, student_length = bias1_matrix.shape
    
    # Define function to check if a token pair is unbiased
    def is_unbiased(original_token_id, new_token_id):
        # Check if token IDs are out of bounds for bias matrices
        if original_token_id >= teacher_length or new_token_id >= student_length:
            return True
        
        # Check if token pair has bias below threshold
        return (
            bias1_matrix[original_token_id, new_token_id] <= bias_threshold
            and bias2_matrix[original_token_id, new_token_id] <= bias_threshold
        )
    
    # Process each example in the batch
    for example_index in range(batch_size):
        # Convert token IDs to tokens
        tokens_teacher = tokenizer_teacher.convert_ids_to_tokens(
            input_ids_teacher_np[example_index]
        )
        tokens_student = tokenizer_student.convert_ids_to_tokens(
            input_ids_student_np[example_index]
        )
        
        # Get alignment indices
        alignment_indices, normalized_tokens_teacher, normalized_tokens_student, special_tokens_mask_teacher, special_tokens_mask_student = get_alignment_indices(
            tokens_teacher,
            tokens_student,
            attention_mask_teacher_np[example_index],
            attention_mask_student_np[example_index],
            tokenizer_teacher,
            tokenizer_student,
        )
        
        # Create masks for valid tokens
        teacher_mask = np.array([len(token) > 0 for token in normalized_tokens_teacher])
        student_mask = np.array([len(token) > 0 for token in normalized_tokens_student])
        
        # Fill alignment matrices with unbiased chunks
        chunk_idx = 0
        for start_i, end_i, start_j, end_j in alignment_indices:
            alignment_matrix_teacher[example_index, start_i:end_i, chunk_idx] = True
            alignment_matrix_student[example_index, start_j:end_j, chunk_idx] = True
            
            # Start a new chunk if:
            # 1. The current chunk ends with a special token, or
            # 2. The token pair is unbiased
            if (
                special_tokens_mask_teacher[end_i - 1]
                or special_tokens_mask_student[end_j - 1]
            ) or is_unbiased(
                input_ids_teacher_np[example_index][end_i - 1],
                input_ids_student_np[example_index][end_j - 1],
            ):
                chunk_idx += 1
        
        # Apply token masks to alignment matrices
        alignment_matrix_teacher[example_index, ~teacher_mask, :] = False
        alignment_matrix_student[example_index, ~student_mask, :] = False
    
    # Convert back to PyTorch tensors if input was PyTorch tensors
    if is_torch_input:
        device = input_ids_teacher.device
        return (
            torch.tensor(alignment_matrix_student, dtype=torch.bool, device=device),
            torch.tensor(alignment_matrix_teacher, dtype=torch.bool, device=device)
        )
    else:
        return alignment_matrix_student, alignment_matrix_teacher


def test_get_alignment_indices():
    """Test function for get_alignment_indices."""
    teacher_tokenizer = load_byteify_tokenizer("Qwen/Qwen2.5-1.5B:source=Qwen2")
    student_tokenizer = load_byteify_tokenizer(
        "meta-llama/Meta-Llama-3-8B-Instruct:source=Llama3"
    )
    
    tokens_teacher = ["<|im_start|>", "Hel", "lo", "?", "Ċ", "Ċ"]
    attention_mask_teacher = np.ones(len(tokens_teacher), dtype=bool)
    tokens_student = [
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "Hello",
        "?",
        "<|end_header_id|>",
        "ĊĊ",
        "Ċ",
    ]
    attention_mask_student = np.ones(len(tokens_student), dtype=bool)
    idx, _, _, _, _ = get_alignment_indices(
        tokens_teacher,
        tokens_student,
        attention_mask_teacher,
        attention_mask_student,
        teacher_tokenizer,
        student_tokenizer,
    )
    
    assert tokens_teacher[idx[0][0]:idx[0][1]] == ["<|im_start|>"]
    assert tokens_student[idx[0][2]:idx[0][3]] == [
        "<|begin_of_text|>",
        "<|start_header_id|>",
    ]
    
    assert tokens_teacher[idx[1][0]:idx[1][1]] == ["Hel", "lo"]
    assert tokens_student[idx[1][2]:idx[1][3]] == ["Hello"]
    
    assert tokens_teacher[idx[2][0]:idx[2][1]] == ["?"]
    assert tokens_student[idx[2][2]:idx[2][3]] == ["?"]
    
    assert tokens_teacher[idx[3][0]:idx[3][1]] == ["Ċ"]
    assert tokens_student[idx[3][2]:idx[3][3]] == ["<|end_header_id|>", "ĊĊ"]
    
    assert tokens_teacher[idx[4][0]:idx[4][1]] == ["Ċ"]
    assert tokens_student[idx[4][2]:idx[4][3]] == ["Ċ"]
    
    # Test that alignment works with torch tensors as well
    torch_attention_mask_teacher = torch.tensor(attention_mask_teacher)
    torch_attention_mask_student = torch.tensor(attention_mask_student)
    
    idx2, _, _, _, _ = get_alignment_indices(
        tokens_teacher,
        tokens_student,
        torch_attention_mask_teacher,
        torch_attention_mask_student,
        teacher_tokenizer,
        student_tokenizer,
    )
    
    # Verify that the indices are the same
    assert idx == idx2