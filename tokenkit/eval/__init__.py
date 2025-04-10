import copy
import logging
import math
from functools import partial
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import torch.nn.functional as F
import lm_eval
from lm_eval.loggers import EvaluationTracker
from tqdm.auto import tqdm

from tokenkit import utils
from tokenkit.eval import generate
from tokenkit.models import param

logger = logging.getLogger(__name__)

ATOL = 1e-2


def score(
    model_fn,
    params,
    model_args,
    labels,
    suffix_mask,
    space_mask,
    logit_mask=None,
    atol=ATOL,
):
    """
    Compute scores for a batch of inputs.
    
    Args:
        model_fn: Model function to call
        params: Model parameters
        model_args: Arguments to pass to the model
        labels: Token labels
        suffix_mask: Mask for suffix tokens
        space_mask: Mask for space tokens
        logit_mask: Mask for logits
        atol: Absolute tolerance for checking if a prediction is greedy
        
    Returns:
        Tuple of sequence log probabilities and whether the prediction is greedy
    """
    input_ids, inputs_embeds = model_args
    
    # Forward pass through the model
    if inputs_embeds is not None:
        outputs = model_fn(inputs_embeds=inputs_embeds)
    else:
        outputs = model_fn(input_ids=input_ids)
    
    logits = outputs.logits.float()
    
    # Apply logit mask if provided
    if logit_mask is not None:
        logit_bias = torch.full(
            (len(logit_mask),),
            fill_value=utils.get_large_negative_number(logits.dtype, module=torch),
            dtype=logits.dtype,
            device=logits.device
        )
        logit_bias = logit_bias * ~logit_mask
        logits = logits + logit_bias[None, None, :]
    
    # Compute log probabilities and probabilities
    logprobs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(logprobs)
    
    # Shift for next token prediction
    shift_logprobs = logprobs[:, :-1]
    shift_labels = labels[:, 1:]
    shift_suffix_mask = suffix_mask[:, 1:]
    
    # Get log probabilities for the correct tokens
    sequence_logprobs = torch.gather(
        shift_logprobs, -1, shift_labels[:, :, None]
    ).squeeze(-1)
    
    # Get maximum logprobs for each position
    max_logprobs = torch.max(shift_logprobs, dim=-1)[0]
    
    # Apply the suffix mask
    sequence_logprobs = (sequence_logprobs * shift_suffix_mask).sum(dim=-1)
    max_logprobs = (max_logprobs * shift_suffix_mask).sum(dim=-1)
    
    # Check if predictions are greedy
    is_greedy = torch.isclose(sequence_logprobs, max_logprobs, rtol=0.0, atol=atol)
    
    return sequence_logprobs, is_greedy


class TorchLM(lm_eval.api.model.LM):
    def __init__(
        self,
        model,
        config,
        params,
        tokenizer,
        lengths,
        tokens_per_batch,
        add_bos,
        chat_template_mode,
        logit_mask=None,
        score_fn=score,
        expand_input_ids=False,
        expand_input_ids_vocab=None,
        expand_input_ids_embeddings=None,
    ):
        self.model = model
        self.model_fn = model
        self.config = config
        self.params = params
        self.tokenizer = tokenizer
        self.lengths = lengths
        self.tokens_per_batch = tokens_per_batch
        self.logit_mask = logit_mask
        self.score_fn = score_fn
        self.add_bos = add_bos
        self.chat_template_mode = chat_template_mode
        
        self.expand_input_ids = expand_input_ids
        self.expand_input_ids_vocab = expand_input_ids_vocab
        self.expand_input_ids_embeddings = expand_input_ids_embeddings
        
        # Create the space mask
        space_mask = utils.get_space_mask(self.tokenizer)[:self.config.vocab_size]
        self.space_mask = torch.zeros(self.config.vocab_size, dtype=torch.bool, device=self.get_device())
        self.space_mask[:len(space_mask)] = torch.tensor(space_mask, device=self.get_device())
        
        # Filter lengths that exceed the maximum sequence length
        for length in list(lengths):
            if length > self.max_length:
                logger.warning(
                    "Ignoring length %d as it exceeds maximum sequence length %d",
                    length,
                    self.max_length,
                )
                lengths.remove(length)
        
        # Verify tokens_per_batch is divisible by all lengths
        for length in self.lengths:
            assert self.tokens_per_batch % length == 0
        
        self.max_batch_size = self.tokens_per_batch // self.lengths[0]
        
        super().__init__()
    
    def get_device(self):
        """Get the device of the model"""
        if hasattr(self.model, 'device'):
            return self.model.device
        elif next(self.model.parameters(), None) is not None:
            return next(self.model.parameters()).device
        else:
            return torch.device('cpu')
    
    # Similar to HuggingFace implementation in lm-evaluation-harness
    @property
    def max_length(self):
        seqlen_config_attrs = (
            "n_positions",
            "max_position_embeddings",
            "n_ctx",
        )
        for attr in seqlen_config_attrs:
            if hasattr(self.config, attr):
                return getattr(self.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self.lengths[-1]
            return self.tokenizer.model_max_length
        return self.lengths[-1]
    
    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        prefixes = [x.args[0] for x in requests]
        suffixes = [x.args[1] for x in requests]
        
        prefix_tokens = self.tokenizer(prefixes, add_special_tokens=False)["input_ids"]
        suffix_tokens = self.tokenizer(suffixes, add_special_tokens=False)["input_ids"]
        
        total_lengths = np.array([
            len(prefix) + len(suffix)
            for prefix, suffix in zip(prefix_tokens, suffix_tokens)
        ])
        
        permutation = np.argsort(total_lengths)[::-1]
        
        n_batches = math.ceil(len(prefix_tokens) / self.max_batch_size)
        
        input_ids = torch.full(
            (self.max_batch_size, self.lengths[-1]),
            fill_value=self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.get_device(),
        )
        suffix_mask = torch.zeros(
            (self.max_batch_size, self.lengths[-1]), 
            dtype=torch.bool,
            device=self.get_device(),
        )
        
        output = [None for _ in range(len(prefix_tokens))]
        
        for batch_idx in tqdm(
            range(n_batches), desc="Running loglikelihood requests..."
        ):
            start, end = (
                batch_idx * self.max_batch_size,
                min((batch_idx + 1) * self.max_batch_size, len(prefix_tokens)),
            )
            
            batch_max_length = 0
            for idx, i in enumerate(permutation[start:end]):
                prefix = prefix_tokens[i]
                suffix = suffix_tokens[i]
                
                # Best-effort truncation from the left
                while len(prefix) + len(suffix) > self.lengths[-1] - self.add_bos:
                    del prefix[0]
                
                if len(prefix) == 0:
                    raise ValueError(
                        f"Prefix is empty after truncation to length {self.lengths[-1]}"
                    )
                
                batch_max_length = max(
                    batch_max_length, len(prefix) + len(suffix) + self.add_bos
                )
                
                input_ids[idx] = self.tokenizer.pad_token_id
                suffix_mask[idx] = False
                
                offset = 0
                
                if self.add_bos:
                    assert self.tokenizer.bos_token_id is not None
                    input_ids[idx, 0] = self.tokenizer.bos_token_id
                    offset = 1
                
                input_ids[idx, offset:offset+len(prefix)] = torch.tensor(prefix, device=self.get_device())
                offset = offset + len(prefix)
                input_ids[idx, offset:offset+len(suffix)] = torch.tensor(suffix, device=self.get_device())
                offset = offset + len(suffix)
                
                suffix_mask[idx, offset-len(suffix):offset] = True
            
            length_index = 0
            while self.lengths[length_index] < batch_max_length:
                length_index += 1
            
            ll = []
            is_greedy = []
            
            length = self.lengths[length_index]
            batch_size = self.tokens_per_batch // length
            
            # Save model state
            self.model.eval()
            with torch.no_grad():
                for i in range(0, self.max_batch_size, batch_size):
                    prev_length = self.config.max_length
                    self.config.max_length = length
                    
                    ll_batch, is_greedy_batch = self.score_fn(
                        self.model_fn,
                        self.params,
                        (input_ids[i:i+batch_size, :length],),
                        input_ids[i:i+batch_size, :length],
                        suffix_mask[i:i+batch_size, :length],
                        self.space_mask,
                        self.logit_mask,
                        ATOL,
                    )
                    
                    self.config.max_length = prev_length
                    
                    # Gather results from all processes if using distributed
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        gathered_ll = [torch.zeros_like(ll_batch) for _ in range(torch.distributed.get_world_size())]
                        gathered_is_greedy = [torch.zeros_like(is_greedy_batch) for _ in range(torch.distributed.get_world_size())]
                        
                        torch.distributed.all_gather(gathered_ll, ll_batch)
                        torch.distributed.all_gather(gathered_is_greedy, is_greedy_batch)
                        
                        ll_batch = torch.cat(gathered_ll)
                        is_greedy_batch = torch.cat(gathered_is_greedy)
                    
                    ll.extend(ll_batch.cpu().tolist())
                    is_greedy.extend(is_greedy_batch.cpu().tolist())
            
            for idx, i in enumerate(permutation[start:end]):
                output[i] = (ll[idx], is_greedy[idx])
        
        return output
    
    def loglikelihood_rolling(self, requests):
        raise NotImplementedError()
    
    def generate_until(self, ungrouped_requests):
        request_groups = {}
        generate_kwargs_groups = {}
        
        ungrouped_output_texts = [None for _ in range(len(ungrouped_requests))]
        
        for i, request in enumerate(ungrouped_requests):
            key = utils.make_hashable(request.args[1])
            if key not in request_groups:
                request_groups[key] = []
                generate_kwargs_groups[key] = copy.deepcopy(request.args[1])
            
            assert generate_kwargs_groups[key] == request.args[1]
            request_groups[key].append((i, request))
        
        for key, requests in request_groups.items():
            generate_kwargs = generate_kwargs_groups[key]
            print("Generating with kwargs:")
            pprint(generate_kwargs)
            print(f"Running {len(requests)} generate requests...")
            
            prompts = [
                utils.preprocess_prompt(x.args[0], self.chat_template_mode)
                for _, x in requests
            ]
            
            until = generate_kwargs.pop("until")
            
            max_new_tokens = generate_kwargs.pop("max_gen_toks", self.lengths[-1])
            sampling_config = generate.SamplingConfig(**generate_kwargs)
            
            device = self.get_device()
            
            generator = generate.Generator(
                model=self.model,
                params=self.params,
                tokenizer=self.tokenizer,
                sampling_config=sampling_config,
                logit_mask=self.logit_mask,
                until=until,
                batch_size=self.tokens_per_batch // self.lengths[-1],
                lengths=[self.lengths[-1]],
                max_new_tokens=max_new_tokens,
                expand_input_ids=self.expand_input_ids,
                expand_input_ids_vocab=self.expand_input_ids_vocab,
                expand_input_ids_embeddings=self.expand_input_ids_embeddings,
                device=device,
            )
            
            output_tokens = generator.generate(prompts)
            
            for request_index, example_output_tokens in zip(
                [i for i, _ in requests], output_tokens
            ):
                ungrouped_output_texts[request_index] = self.tokenizer.decode(
                    example_output_tokens
                ).strip()
            
            # Save the model state
            self.params = generator.params
        
        return ungrouped_output_texts


def evaluate(
    model,
    config,
    params,
    tokenizer,
    tasks,
    lengths,
    tokens_per_batch,
    logit_mask=None,
    output=None,
    add_bos=True,
    chat_template_mode="surround_instruct",
    cache_requests=True,
    torch_kwargs=None,
    **kwargs,
):
    """
    Evaluate a model on the given tasks.
    
    Args:
        model: PyTorch model
        config: Model configuration
        params: Model parameters
        tokenizer: Tokenizer
        tasks: List of tasks to evaluate on
        lengths: List of sequence lengths to use
        tokens_per_batch: Number of tokens per batch
        logit_mask: Mask for logits
        output: Output directory
        add_bos: Whether to add BOS token
        chat_template_mode: Chat template mode
        cache_requests: Whether to cache requests
        torch_kwargs: Additional keyword arguments
        **kwargs: Additional keyword arguments
        
    Returns:
        Tuple of results and model parameters
    """
    if output is not None:
        output_dir = Path(output)
        output_dir.mkdir(exist_ok=True, parents=True)
    else:
        output_dir = None
    
    lm_eval_model = TorchLM(
        model=model,
        config=config,
        params=params,
        tokenizer=tokenizer,
        lengths=lengths,
        tokens_per_batch=tokens_per_batch,
        add_bos=add_bos,
        chat_template_mode=chat_template_mode,
        logit_mask=logit_mask,
        **(torch_kwargs or {}),
    )
    
    if output_dir is not None:
        evaluation_tracker = EvaluationTracker(output_path=output_dir)
    else:
        evaluation_tracker = None
    
    results = lm_eval.simple_evaluate(
        lm_eval_model,
        model_args="",
        tasks=tasks,
        evaluation_tracker=evaluation_tracker,
        cache_requests=cache_requests,
        **kwargs,
    )
    
    if evaluation_tracker is not None:
        evaluation_tracker.general_config_tracker.model_source = "torch_lm"
    
    if output_dir is not None:
        evaluation_tracker.save_results_aggregated(
            results=results, samples=results["samples"]
        )
        for task_name, config in results["configs"].items():
            evaluation_tracker.save_results_samples(
                task_name=task_name, samples=results["samples"][task_name]
            )
    
    return results["results"], lm_eval_model.params