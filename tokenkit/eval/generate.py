import copy
import logging
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from tokenkit import utils
from tokenkit.constants import EXPAND_INPUT_IDS_MAX_LENGTH
from tokenkit.models import param

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamplingConfig:
    """Configuration for text generation sampling."""
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 0
    no_repeat_ngram_size: Optional[int] = None


class Generator:
    """PyTorch text generator for language models."""
    
    def __init__(
        self,
        model,
        params,
        tokenizer,
        sampling_config,
        batch_size,
        max_new_tokens,
        device=None,
        logit_mask=None,
        until: Optional[List[str]] = None,
        lengths=[1024, 2048, 4096],
        eos_strategy="stop",  # 'forbid', 'stop', 'ignore', or 'restart'
        pad_to_multiple_of=128,
        expand_input_ids=False,
        expand_input_ids_vocab=None,
        expand_input_ids_embeddings=None,
    ):
        """
        Initialize the generator.
        
        Args:
            model: PyTorch language model
            params: Model parameters
            tokenizer: Tokenizer
            sampling_config: Configuration for text generation sampling
            batch_size: Batch size for generation
            max_new_tokens: Maximum number of new tokens to generate
            device: Device to use for generation
            logit_mask: Mask for logits
            until: List of strings to stop generation at
            lengths: List of sequence lengths to use
            eos_strategy: Strategy for handling EOS tokens
            pad_to_multiple_of: Pad embeddings to a multiple of this value
            expand_input_ids: Whether to use expanded input IDs
            expand_input_ids_vocab: Vocabulary for expanded input IDs
            expand_input_ids_embeddings: Embeddings for expanded input IDs
        """
        self.model = model
        self.params = params
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()
        self.sampling_config = sampling_config
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.logit_mask = logit_mask
        self.device = device if device is not None else self._get_device()
        self.config = model.config
        
        # Set up expanded input IDs if needed
        self.expand_input_ids = expand_input_ids
        if expand_input_ids:
            self.expand_input_ids_vocab = expand_input_ids_vocab
            if isinstance(expand_input_ids_embeddings, np.ndarray):
                self.expand_input_ids_embeddings = torch.tensor(
                    expand_input_ids_embeddings, device=self.device)
            else:
                self.expand_input_ids_embeddings = expand_input_ids_embeddings.to(self.device)
        
        # Set up stop tokens
        self.until_tokens = []
        if until is not None:
            for stop_sequence_str in until:
                # with/without prefix space
                self.until_tokens.append(
                    tokenizer.encode(stop_sequence_str, add_special_tokens=False)
                )
                self.until_tokens.append(
                    tokenizer.encode(" " + stop_sequence_str, add_special_tokens=False)
                )
        
        if until is None or tokenizer.eos_token not in until:
            self.until_tokens.append([tokenizer.eos_token_id])
        
        if (
            until is None
            or self.tokenizer.model_kind_cls.replacements["<|<eot>|>"][0] not in until
        ):
            self.until_tokens.append(
                [
                    self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.model_kind_cls.replacements["<|<eot>|>"][0]
                    )
                ]
            )
        
        if (
            until is None
            or self.tokenizer.model_kind_cls.replacements["<|<eos>|>"][0] not in until
        ):
            self.until_tokens.append(
                [
                    self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.model_kind_cls.replacements["<|<eos>|>"][0]
                    )
                ]
            )
        
        self.until_tokens = [
            torch.tensor(x, dtype=torch.long, device=self.device)
            for x in self.until_tokens
        ]
        
        print("Will stop generation at the following stop sequences:")
        for until_sequence in self.until_tokens:
            print(tokenizer.convert_ids_to_tokens(until_sequence.cpu().numpy()))
        
        self.lengths = lengths
        self.eos_strategy = eos_strategy
        assert self.eos_strategy == "stop"
        
        # Prepare special token IDs
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else None
    
    def _get_device(self):
        """Get the device of the model."""
        if hasattr(self.model, 'device'):
            return self.model.device
        elif next(self.model.parameters(), None) is not None:
            return next(self.model.parameters()).device
        else:
            return torch.device('cpu')
    
    def compute_inputs_embeds(self, input_ids, expanded_input_ids=None, last_only=False):
        """
        Compute input embeddings, optionally with expanded input IDs.
        
        Args:
            input_ids: Input token IDs
            expanded_input_ids: Expanded input token IDs
            last_only: Whether to compute embeddings for the last token only
            
        Returns:
            Input embeddings
        """
        # Get the embedding matrix
        embedding_matrix = getattr(self.model, param.get_input_embedding_path(self.config.model_type)).weight
        
        # Compute standard embeddings
        if last_only:
            inputs_embeds = embedding_matrix[input_ids[:, -1:]]
        else:
            inputs_embeds = embedding_matrix[input_ids]
        
        # Add expanded embeddings if requested
        if self.expand_input_ids:
            if expanded_input_ids is None:
                # Compute expanded input IDs
                expanded_input_ids = utils.expand_input_ids(
                    input_ids.cpu().numpy(),
                    self.tokenizer,
                    self.expand_input_ids_vocab,
                    use_heuristic=True,
                    maxlen=EXPAND_INPUT_IDS_MAX_LENGTH,
                )
                expanded_input_ids = torch.tensor(expanded_input_ids, device=input_ids.device)
            
            # Compute expanded embeddings
            expanded_inputs_embeds = self.expand_input_ids_embeddings[expanded_input_ids]
            
            # Add to standard embeddings
            inputs_embeds = inputs_embeds + expanded_inputs_embeds
        
        return inputs_embeds
    
    def generate_step(self, 
                     input_ids, 
                     running_token, 
                     is_sent_finished, 
                     past_key_values,
                     cur_len):
        """
        Generate a single step of tokens.
        
        Args:
            input_ids: Current sequence of input token IDs
            running_token: Current running token sequence
            is_sent_finished: Boolean tensor indicating which sequences are finished
            past_key_values: Past key values for the model
            cur_len: Current length of the sequence
            
        Returns:
            Tuple of (next tokens, updated running token, updated is_sent_finished, updated past_key_values)
        """
        # Compute embeddings for the last token only
        inputs_embeds = self.compute_inputs_embeds(running_token, last_only=True)
        
        # Forward pass through the model
        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
        
        # Get logits for the next token
        raw_logits = outputs.logits[:, -1].float()
        
        # Apply logit mask if provided
        if self.logit_mask is not None:
            raw_logits = torch.where(
                self.logit_mask,
                raw_logits,
                utils.get_large_negative_number(raw_logits.dtype, torch) * torch.ones_like(raw_logits),
            )
        
        # Apply n-gram repetition penalty if requested
        if self.sampling_config.no_repeat_ngram_size is not None:
            # Apply n-gram repetition penalty
            ngram_size = self.sampling_config.no_repeat_ngram_size
            for i in range(input_ids.shape[0]):
                if is_sent_finished[i]:
                    continue
                    
                # Get ngrams from the current sequence
                for j in range(ngram_size, cur_len + 1):
                    ngram = input_ids[i, j-ngram_size:j].tolist()
                    ngram_end_position = j
                    
                    # Check if this ngram appears earlier in the sequence
                    for k in range(ngram_size, ngram_end_position):
                        if ngram == input_ids[i, k-ngram_size:k].tolist():
                            # Prevent generating the next token of this ngram
                            raw_logits[i, input_ids[i, k]] = utils.get_large_negative_number(raw_logits.dtype, torch)
        
        # Sample or greedy selection
        if not self.sampling_config.do_sample:
            # Greedy selection
            next_token = torch.argmax(raw_logits, dim=-1)
        else:
            # Apply temperature
            if self.sampling_config.temperature != 1.0:
                logits = raw_logits / self.sampling_config.temperature
            else:
                logits = raw_logits
            
            # Apply top-k if specified
            if self.sampling_config.top_k > 0:
                top_k = min(self.sampling_config.top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits = logits.masked_fill(indices_to_remove, utils.get_large_negative_number(logits.dtype, torch))
            
            # Apply top-p if specified
            if 0.0 < self.sampling_config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > self.sampling_config.top_p
                
                # Shift the indices to the right to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, utils.get_large_negative_number(logits.dtype, torch))
            
            # Convert to probabilities and sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # Apply finished mask
        next_token = torch.where(
            is_sent_finished,
            torch.tensor(self.pad_token_id, device=next_token.device),
            next_token
        )
        
        # Add next token to sequences
        next_token = next_token.unsqueeze(-1)
        
        # Update running token
        next_running_token = torch.cat([
            running_token[:, 1:],
            next_token,
        ], dim=1)
        
        # Update is_sent_finished
        next_is_sent_finished = is_sent_finished.clone()
        
        # Check for stop sequences
        for until_token in self.until_tokens:
            if cur_len + 1 >= until_token.shape[0]:
                # Get the last N tokens where N is the length of the stop sequence
                seq_end = input_ids[:, cur_len + 1 - until_token.shape[0]:cur_len + 1]
                
                # Add the next token
                seq_end_with_next = torch.cat([seq_end[:, 1:], next_token], dim=1)
                
                # Check if the sequence matches the stop sequence
                is_match = (seq_end_with_next == until_token.unsqueeze(0)).all(dim=1)
                next_is_sent_finished = next_is_sent_finished | is_match
        
        # Mark sequences as finished if EOS token is generated
        next_is_sent_finished = next_is_sent_finished | (next_token.squeeze(-1) == self.eos_token_id)
        
        return next_token, next_running_token, next_is_sent_finished, outputs.past_key_values
    
    def generate(self, prompts, seed=1234):
        """
        Generate text from prompts.
        
        Args:
            prompts: List of prompts to generate from
            seed: Random seed for sampling
            
        Returns:
            List of generated token sequences
        """
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        
        all_prefill_tokens = []
        all_running_tokens = []
        
        # Tokenize all prompts
        for prompt in tqdm(prompts, desc="Encoding prompts..."):
            prompt_tokens = [
                self.vocab[x] for x in utils.encode_prompt(prompt, self.tokenizer)[0]
            ]
            
            all_prefill_tokens.append(prompt_tokens[:-1])
            all_running_tokens.append(prompt_tokens[-EXPAND_INPUT_IDS_MAX_LENGTH:])
        
        # Sort prompts by length (longest first)
        permutation_indices = np.argsort([len(x) for x in all_prefill_tokens])[::-1]
        generations = [None] * len(prompts)
        
        n_batches = (len(prompts) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in tqdm(range(n_batches), desc="Generating..."):
            (start, end) = (
                batch_idx * self.batch_size,
                min((batch_idx + 1) * self.batch_size, len(prompts)),
            )
            
            batch_indices = permutation_indices[start:end]
            batch_size = len(batch_indices)
            
            # Pad batch if necessary
            if batch_size < self.batch_size:
                batch_indices = np.pad(
                    batch_indices, (0, self.batch_size - batch_size), mode="edge"
                )
            
            # Get prefill tokens for this batch
            unpadded_prefill_input_ids = [all_prefill_tokens[i] for i in batch_indices]
            max_new_tokens = min(
                self.max_new_tokens,
                self.lengths[-1] - max(len(x) for x in unpadded_prefill_input_ids),
            )
            
            if max_new_tokens < self.max_new_tokens:
                print(
                    f"Warning: max_new_tokens reduced from {self.max_new_tokens} to {max_new_tokens}"
                )
            
            # Get the smallest length from self.lengths that can accommodate the longest prompt + max_new_tokens
            max_prefill_length = max(len(x) for x in unpadded_prefill_input_ids)
            padded_prefill_length = min([length for length in self.lengths if length >= max_prefill_length + max_new_tokens],
                                       default=self.lengths[-1])
            
            # Prepare input tensors
            prefill_input_ids = torch.full(
                (self.batch_size, padded_prefill_length),
                fill_value=self.pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            
            attention_mask = torch.zeros(
                (self.batch_size, padded_prefill_length),
                dtype=torch.long,
                device=self.device,
            )
            
            running_tokens = torch.tensor(
                [all_running_tokens[i] + [self.pad_token_id] * (EXPAND_INPUT_IDS_MAX_LENGTH - len(all_running_tokens[i])) 
                 for i in batch_indices],
                dtype=torch.long,
                device=self.device,
            )
            
            # Fill in the input IDs and attention mask
            for i, input_ids in enumerate(unpadded_prefill_input_ids):
                prefill_input_ids[i, :len(input_ids)] = torch.tensor(input_ids, device=self.device)
                attention_mask[i, :len(input_ids)] = 1
                attention_mask[i, padded_prefill_length - max_new_tokens:] = 1
            
            # Set the model to evaluation mode
            self.model.eval()
            
            # Initial forward pass to get past key values
            with torch.no_grad():
                if prefill_input_ids.shape[1] > 0:
                    # Get expanded input IDs if needed
                    if self.expand_input_ids:
                        prefill_expanded_input_ids = utils.expand_input_ids(
                            prefill_input_ids.cpu().numpy(),
                            self.tokenizer,
                            self.expand_input_ids_vocab,
                            use_heuristic=True,
                        )
                        prefill_expanded_input_ids = torch.tensor(
                            prefill_expanded_input_ids, device=self.device)
                        inputs_embeds = self.compute_inputs_embeds(
                            prefill_input_ids, prefill_expanded_input_ids)
                        
                        outputs = self.model(
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            use_cache=True,
                        )
                    else:
                        outputs = self.model(
                            input_ids=prefill_input_ids,
                            attention_mask=attention_mask,
                            use_cache=True,
                        )
                    
                    past_key_values = outputs.past_key_values
                else:
                    past_key_values = None
            
            # Initialize generation state
            cur_len = padded_prefill_length - max_new_tokens
            is_sent_finished = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
            
            # Generate tokens one by one
            for _ in range(max_new_tokens):
                if is_sent_finished.all():
                    break
                
                next_token, running_tokens, is_sent_finished, past_key_values = self.generate_step(
                    prefill_input_ids, running_tokens, is_sent_finished, past_key_values, cur_len
                )
                
                # Update the input IDs with the new token
                prefill_input_ids.scatter_(1, torch.full((self.batch_size, 1), cur_len, device=self.device), next_token)
                
                # Increment current length
                cur_len += 1
            
            # Get the generated tokens
            generated_tokens = prefill_input_ids[:, padded_prefill_length - max_new_tokens:].cpu().numpy()
            
            # Filter special tokens
            special_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.model_kind_cls.special_tokens
            )
            
            for i, idx in enumerate(batch_indices[:batch_size]):  # Only process actual batch, not padding
                generations[idx] = [
                    token_id for token_id in generated_tokens[i] 
                    if token_id not in special_ids and token_id != self.pad_token_id
                ]
        
        return generations