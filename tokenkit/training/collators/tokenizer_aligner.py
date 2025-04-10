import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset

from tokenkit import align, utils

logger = logging.getLogger(__name__)


class TokenizerAlignerCollator:
    """Collator for aligning tokens between two tokenizers.
    
    This collator handles the creation of alignment matrices between teacher and student
    tokenizers, which is essential for cross-tokenizer distillation.
    """
    
    def __init__(
        self,
        tokenizer_original,
        tokenizer_new,
        max_teacher_length: int,
        max_student_length: int,
        special_tokens_mode: str,
        with_expanded_input_ids: bool = False,
        use_chat_template: bool = False,
        chat_template_mode: str = "direct_encode",
        loss_mask_mode: Optional[str] = None,
        tokenizer_pair_data_path: Optional[str] = None,
        tokenizer_pair_bias_threshold: float = 0.0,
        require_bias_matrices: bool = False,
    ):
        """Initialize the tokenizer aligner collator.
        
        Args:
            tokenizer_original: Teacher tokenizer
            tokenizer_new: Student tokenizer
            max_teacher_length: Maximum sequence length for teacher model
            max_student_length: Maximum sequence length for student model
            special_tokens_mode: How to handle special tokens
            with_expanded_input_ids: Whether to include expanded input IDs
            use_chat_template: Whether to use chat template for encoding
            chat_template_mode: Mode to use for chat template
            loss_mask_mode: Mode to use for loss masking
            tokenizer_pair_data_path: Path to tokenizer pair data
            tokenizer_pair_bias_threshold: Threshold for tokenizer pair bias
            require_bias_matrices: Whether bias matrices are required
        """
        self.tokenizer_original = tokenizer_original
        self.tokenizer_original_vocab = tokenizer_original.get_vocab()
        self.tokenizer_new = tokenizer_new
        self.max_teacher_length = max_teacher_length
        self.max_student_length = max_student_length
        self.special_tokens_mode = special_tokens_mode
        self.with_expanded_input_ids = with_expanded_input_ids
        self.use_chat_template = use_chat_template
        self.chat_template_mode = chat_template_mode
        
        # Set up loss mask tokens
        if loss_mask_mode is None:
            loss_mask_string = None
        elif loss_mask_mode == "dolly":
            loss_mask_string = "### Response:\n"
        elif loss_mask_mode == "openmath2":
            loss_mask_string = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            raise ValueError(f"Unknown loss mask mode: {loss_mask_mode}")
        
        self.loss_mask_tokens_original = (
            self.tokenizer_original.encode(loss_mask_string, add_special_tokens=False)
            if loss_mask_string is not None
            else None
        )
        self.loss_mask_tokens_new = (
            self.tokenizer_new.encode(loss_mask_string, add_special_tokens=False)
            if loss_mask_string is not None
            else None
        )
        
        # Load bias matrices if available
        if tokenizer_pair_data_path:
            bias1_matrix_path = Path(tokenizer_pair_data_path) / "bias1_matrix.npz"
            bias2_matrix_path = Path(tokenizer_pair_data_path) / "bias2_matrix.npz"
            teacher_token_counts_path = Path(tokenizer_pair_data_path) / "teacher_counts.json"
            student_token_counts_path = Path(tokenizer_pair_data_path) / "student_counts.json"
            
            # Load bias matrices if they exist
            if bias1_matrix_path.exists():
                self.tokenizer_pair_bias1_matrix = sparse.load_npz(bias1_matrix_path).todok()
            else:
                self.tokenizer_pair_bias1_matrix = None
                
            if bias2_matrix_path.exists():
                self.tokenizer_pair_bias2_matrix = sparse.load_npz(bias2_matrix_path).todok()
            else:
                self.tokenizer_pair_bias2_matrix = None
                
            # Load token counts if they exist
            if teacher_token_counts_path.exists():
                with open(teacher_token_counts_path, 'r') as f:
                    self.teacher_token_probs = utils.compute_unigram_probabilities(
                        tokenizer_original, json.load(f)
                    )
            else:
                self.teacher_token_probs = None
                
            if student_token_counts_path.exists():
                with open(student_token_counts_path, 'r') as f:
                    self.student_token_probs = utils.compute_unigram_probabilities(
                        tokenizer_new, json.load(f)
                    )
            else:
                self.student_token_probs = None
        else:
            self.tokenizer_pair_bias1_matrix = None
            self.tokenizer_pair_bias2_matrix = None
            self.teacher_token_probs = None
            self.student_token_probs = None
        
        # Check if bias matrices are required but not found
        if require_bias_matrices and (
            self.tokenizer_pair_bias1_matrix is None
            or self.tokenizer_pair_bias2_matrix is None
        ):
            raise ValueError("Bias matrices are required but not found in the given path.")
        
        self.tokenizer_pair_bias_threshold = tokenizer_pair_bias_threshold
        
        # Compute prefix maps for tokenizers
        self.prefix_map_original = self._compute_prefix_map(tokenizer_original)
        self.prefix_map_new = self._compute_prefix_map(tokenizer_new)
    
    def _compute_loss_mask(self, input_ids: np.ndarray, attention_mask: np.ndarray, loss_mask_tokens: Optional[List[int]]) -> np.ndarray:
        """Compute loss mask based on attention mask and loss mask tokens.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            loss_mask_tokens: Loss mask tokens to use
            
        Returns:
            Loss mask array
        """
        loss_mask = attention_mask.astype(bool)
        
        if loss_mask_tokens is not None:
            for i in range(len(input_ids)):
                for j in range(len(input_ids[i])):
                    if input_ids[i][j] != loss_mask_tokens[0]:
                        continue
                    
                    if j + len(loss_mask_tokens) <= len(input_ids[i]) and np.array_equal(
                        input_ids[i][j:j+len(loss_mask_tokens)], loss_mask_tokens
                    ):
                        loss_mask[i, :j+len(loss_mask_tokens)] = False
        
        return loss_mask
    
    def _compute_prefix_map(self, tokenizer) -> Dict[str, List[str]]:
        """Compute prefix map for a tokenizer.
        
        Args:
            tokenizer: The tokenizer to compute prefix map for
            
        Returns:
            Dictionary mapping prefixes to tokens
        """
        prefix_map = {}
        
        for token in tokenizer.get_vocab().keys():
            if isinstance(token, bytes):
                continue  # Skip byte tokens
                
            for i in range(1, len(token) + 1):
                prefix = token[:i]
                if prefix in prefix_map:
                    prefix_map[prefix].append(token)
                else:
                    prefix_map[prefix] = [token]
        
        return prefix_map
    
    def _encode_with_chat_template(self, texts: List[str], tokenizer, max_length: int) -> Dict[str, np.ndarray]:
        """Encode texts using chat template.
        
        Args:
            texts: Texts to encode
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        input_ids = np.full(
            (len(texts), max_length), 
            fill_value=tokenizer.pad_token_id, 
            dtype=np.int32
        )
        attention_mask = np.zeros((len(texts), max_length), dtype=np.int32)
        
        for i in range(len(texts)):
            # Process prompt with chat template
            processed_prompt = utils.preprocess_prompt(texts[i], self.chat_template_mode)
            tokens, _ = utils.encode_prompt(processed_prompt, tokenizer, max_length=max_length)
            current_input_ids = tokenizer.convert_tokens_to_ids(tokens)
            
            # Truncate if needed
            if len(current_input_ids) > max_length:
                current_input_ids = current_input_ids[:max_length]
            
            # Fill arrays
            input_ids[i, :len(current_input_ids)] = current_input_ids
            attention_mask[i, :len(current_input_ids)] = 1
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    
    def __call__(self, examples) -> Dict[str, torch.Tensor]:
        """Process a batch of examples.
        
        Args:
            examples: Batch of examples
            
        Returns:
            Dictionary with processed batch data
        """
        # Handle different input formats
        if isinstance(examples, list) and len(examples) > 0 and isinstance(examples[0], dict):
            # Handle list of dictionaries format
            texts = [ex["text"] for ex in examples]
        elif isinstance(examples, dict) and "text" in examples:
            # Handle direct dictionary format
            texts = examples["text"]
        elif isinstance(examples, list) and len(examples) == 1 and isinstance(examples[0], dict):
            # Handle batch of one example
            texts = examples[0]["text"]
        else:
            raise ValueError(f"Unsupported examples format: {type(examples)}")
        
        # Ensure texts is a list
        if not isinstance(texts, list):
            texts = [texts]
        
        # Encode texts
        if self.use_chat_template:
            encoding_original = self._encode_with_chat_template(
                texts,
                tokenizer=self.tokenizer_original,
                max_length=self.max_teacher_length,
            )
            encoding_new = self._encode_with_chat_template(
                texts,
                tokenizer=self.tokenizer_new,
                max_length=self.max_student_length,
            )
        else:
            encoding_original = self.tokenizer_original(
                texts,
                max_length=self.max_teacher_length,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
            encoding_new = self.tokenizer_new(
                texts,
                max_length=self.max_student_length,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
        
        # Extract encoded data
        input_ids_original = encoding_original["input_ids"]
        attention_mask_original = encoding_original["attention_mask"]
        input_ids_new = encoding_new["input_ids"]
        attention_mask_new = encoding_new["attention_mask"]
        
        # Get unconstrained alignments
        alignment_matrix_a, alignment_matrix_b = align.get_unconstrained_alignments(
            input_ids_original,
            input_ids_new,
            attention_mask_original,
            attention_mask_new,
            tokenizer_teacher=self.tokenizer_original,
            tokenizer_student=self.tokenizer_new,
        )
        
        # Get space alignments
        alignment_matrix_a_space, alignment_matrix_b_space = align.get_space_alignments(
            input_ids_original,
            input_ids_new,
            attention_mask_original,
            attention_mask_new,
            tokenizer_teacher=self.tokenizer_original,
            tokenizer_student=self.tokenizer_new,
        )
        
        # Get unbiased alignments if bias matrices are available
        if (
            self.tokenizer_pair_bias1_matrix is not None
            and self.tokenizer_pair_bias2_matrix is not None
        ):
            alignment_matrix_a_unbiased, alignment_matrix_b_unbiased = align.get_unbiased_alignments(
                input_ids_original,
                input_ids_new,
                attention_mask_original,
                attention_mask_new,
                tokenizer_teacher=self.tokenizer_original,
                tokenizer_student=self.tokenizer_new,
                pair_data=(
                    self.tokenizer_pair_bias1_matrix,
                    self.tokenizer_pair_bias2_matrix,
                    self.teacher_token_probs,
                    self.student_token_probs,
                ),
                bias_threshold=self.tokenizer_pair_bias_threshold,
            )
        else:
            # If bias matrices are not available, use NaN values
            alignment_matrix_a_unbiased = np.full_like(alignment_matrix_a, fill_value=np.nan)
            alignment_matrix_b_unbiased = np.full_like(alignment_matrix_b, fill_value=np.nan)
        
        # Create masks for occurring tokens
        occuring_tokens_mask_original = np.zeros(len(self.tokenizer_original), dtype=bool)
        occuring_tokens_mask_new = np.zeros(len(self.tokenizer_new), dtype=bool)
        
        np.put(occuring_tokens_mask_original, input_ids_original.flatten(), True)
        np.put(occuring_tokens_mask_new, input_ids_new.flatten(), True)
        
        # Compute loss masks
        loss_mask_original = self._compute_loss_mask(
            input_ids_original, attention_mask_original, self.loss_mask_tokens_original
        )
        loss_mask_new = self._compute_loss_mask(
            input_ids_new, attention_mask_new, self.loss_mask_tokens_new
        )
        
        # Create batch dictionary
        batch = {
            "input_ids_new": torch.tensor(input_ids_new, dtype=torch.long),
            "attention_mask_new": torch.tensor(attention_mask_new, dtype=torch.long),
            "occuring_tokens_mask_new": torch.tensor(occuring_tokens_mask_new, dtype=torch.bool),
            "input_ids_original": torch.tensor(input_ids_original, dtype=torch.long),
            "attention_mask_original": torch.tensor(attention_mask_original, dtype=torch.long),
            "occuring_tokens_mask_original": torch.tensor(occuring_tokens_mask_original, dtype=torch.bool),
            "alignment_matrix_a_unconstrained": torch.tensor(alignment_matrix_a, dtype=torch.float32),
            "alignment_matrix_b_unconstrained": torch.tensor(alignment_matrix_b, dtype=torch.float32),
            "alignment_matrix_a_space": torch.tensor(alignment_matrix_a_space, dtype=torch.float32),
            "alignment_matrix_b_space": torch.tensor(alignment_matrix_b_space, dtype=torch.float32),
            "alignment_matrix_a_unbiased": torch.tensor(alignment_matrix_a_unbiased, dtype=torch.float32),
            "alignment_matrix_b_unbiased": torch.tensor(alignment_matrix_b_unbiased, dtype=torch.float32),
            "loss_mask_original": torch.tensor(loss_mask_original, dtype=torch.bool),
            "loss_mask_new": torch.tensor(loss_mask_new, dtype=torch.bool),
        }
        
        # Add expanded input IDs if requested
        if self.with_expanded_input_ids:
            expanded_input_ids = utils.expand_input_ids(
                input_ids_new,
                tokenizer=self.tokenizer_new,
                original_vocab=self.tokenizer_original_vocab,
                use_heuristic=True,
            )
            batch["expanded_input_ids_new"] = torch.tensor(expanded_input_ids, dtype=torch.long)
        
        return batch


class TokenizerAlignerDataset(Dataset):
    """Dataset wrapper that applies tokenizer alignment to texts."""
    
    def __init__(
        self, 
        texts: List[str],
        collator: TokenizerAlignerCollator
    ):
        """Initialize the dataset.
        
        Args:
            texts: List of text samples
            collator: Tokenizer aligner collator to use
        """
        self.texts = texts
        self.collator = collator
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx) -> Dict[str, str]:
        return {"text": self.texts[idx]}
    
    def collate_fn(self, examples) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader.
        
        Args:
            examples: List of examples
            
        Returns:
            Batch dictionary with processed data
        """
        return self.collator(examples)