"""
Example Usage:

python3 scripts/eval.py \
    +main.pretrained_model_name_or_path='google/gemma-2-2b-it' \
    +main.tokenizer_name='google/gemma-2-2b-it:source=Gemma2'

To evaluate a model with expanded input ids (byte-level models with n-gram embeddings), use:

python3 scripts/eval.py \
    +main.pretrained_model_name_or_path=<model_name> \
    +main.tokenizer_name=<model_name> \
    +expand.pretrained_model_name_or_path=google/gemma-2-2b-it \
    +expand.tokenizer_name=google/gemma-2-2b-it:source=Gemma2
"""

import logging
import os
from functools import partial
from pathlib import Path
from pprint import pformat, pprint

import datasets
import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM

from tokenkit import utils
from tokenkit.byteify import load_byteify_tokenizer
from tokenkit.eval import ATOL, evaluate, score
from tokenkit.models import param, sharding

logger = logging.getLogger(__name__)

datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = (
    True  # careful about this, required for lm_eval
)


def pad_embeddings(embeddings, tokenizer):
    n_embed_diff = len(tokenizer) - len(embeddings)

    if isinstance(embeddings, torch.Tensor):
        embeddings_mean = embeddings.mean(0)
        embeddings_std = embeddings.std(0)
        
        # Create random embeddings with matching statistics
        random_embeddings = torch.randn(
            n_embed_diff, *embeddings.shape[1:],
            device=embeddings.device,
            dtype=embeddings.dtype
        ) * embeddings_std[None] + embeddings_mean[None]
        
        return torch.cat([embeddings, random_embeddings], dim=0)
    else:
        # Handle numpy arrays
        embeddings_mean = embeddings.mean(0)
        embeddings_std = embeddings.std(0)
        
        return np.concatenate(
            [
                embeddings,
                np.random.normal(
                    size=(n_embed_diff, *embeddings.shape[1:]),
                ) * embeddings_std[None] + embeddings_mean[None],
            ]
        )


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def my_app(args: DictConfig) -> None:
    logger.info(pformat(OmegaConf.to_object(args)))

    model_kwargs = OmegaConf.to_object(args.main)
    eval_kwargs = OmegaConf.to_object(args.eval)

    # Initialize distributed training if available
    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")
    
    if args.use_cpu:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            if torch.distributed.is_available() and not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl")
        else:
            device = torch.device("cpu")

    if eval_kwargs["output"] is not None:
        output_dir = Path(eval_kwargs["output"])
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config=args, f=output_dir / "args.yaml", resolve=True)

    config = AutoConfig.from_pretrained(**model_kwargs)
    config.max_length = eval_kwargs["lengths"][-1]

    tokenizer = load_byteify_tokenizer(model_kwargs.pop("tokenizer_name"))

    # Load the model architecture
    model = AutoModelForCausalLM.from_config(config)
    
    # Load model parameters
    params = param.load_params(**model_kwargs)

    if args.expand_input_ids:
        expand_model_kwargs = OmegaConf.to_object(args.expand)

        expand_config = AutoConfig.from_pretrained(**expand_model_kwargs)
        expand_tokenizer = load_byteify_tokenizer(
            expand_model_kwargs.pop("tokenizer_name")
        )
        expand_vocab = expand_tokenizer.get_vocab()

        expand_input_ids_model_params = param.load_params(**expand_model_kwargs)
        expand_input_ids_embeddings = param.get(
            expand_input_ids_model_params,
            f"{param.get_input_embedding_path(expand_config.model_type)}.weight",
        )

        n_overflow = expand_input_ids_embeddings.shape[0] % args.pad_to_multiple_of
        if n_overflow > 0:
            n_pad = args.pad_to_multiple_of - n_overflow
        else:
            n_pad = 0

        if isinstance(expand_input_ids_embeddings, torch.Tensor):
            expand_input_ids_embeddings = torch.nn.functional.pad(
                expand_input_ids_embeddings,
                (0, 0, 0, n_pad),
                mode="constant",
                value=0,
            )
        else:
            expand_input_ids_embeddings = np.pad(
                expand_input_ids_embeddings,
                ((0, n_pad), (0, 0)),
                mode="constant",
                constant_values=0,
            )
    else:
        expand_tokenizer = None
        expand_vocab = None
        expand_input_ids_embeddings = None

    input_embeddings = param.get(
        params, f"{param.get_input_embedding_path(config.model_type)}.weight"
    )
    input_embeddings = input_embeddings[: len(tokenizer)]

    if len(input_embeddings) < len(tokenizer):
        print("Padding input embeddings...")
        input_embeddings = pad_embeddings(input_embeddings, tokenizer)

    if not config.tie_word_embeddings:
        output_embeddings = param.get(
            params, f"{param.get_output_embedding_path(config.model_type)}.weight"
        )
        output_embeddings = output_embeddings[:, : len(tokenizer)]
        print("Padding output embeddings...")
        if isinstance(output_embeddings, torch.Tensor):
            output_embeddings = pad_embeddings(output_embeddings.T, tokenizer).T
        else:
            output_embeddings = pad_embeddings(output_embeddings.T, tokenizer).T
    else:
        output_embeddings = None

    n_overflow = input_embeddings.shape[0] % args.pad_to_multiple_of
    if n_overflow > 0:
        n_pad = args.pad_to_multiple_of - n_overflow
    else:
        n_pad = 0

    if isinstance(input_embeddings, torch.Tensor):
        input_embeddings = torch.nn.functional.pad(
            input_embeddings,
            (0, 0, 0, n_pad),
            mode="constant",
            value=0,
        )
        if output_embeddings is not None:
            output_embeddings = torch.nn.functional.pad(
                output_embeddings,
                (0, n_pad, 0, 0),
                mode="constant",
                value=0,
            )
        logit_mask = torch.zeros((input_embeddings.shape[0],), dtype=torch.bool, device=device)
        logit_mask[:model.config.vocab_size] = True
    else:
        input_embeddings = np.pad(
            input_embeddings,
            ((0, n_pad), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        if output_embeddings is not None:
            output_embeddings = np.pad(
                output_embeddings,
                ((0, 0), (0, n_pad)),
                mode="constant",
                constant_values=0,
            )
        logit_mask = np.zeros((input_embeddings.shape[0],), dtype=bool)
        logit_mask[:model.config.vocab_size] = True
        
    model.config.vocab_size = input_embeddings.shape[0]

    # Set the embeddings in the model parameters
    params = param.put(
        params, f"{param.get_input_embedding_path(config.model_type)}.weight", input_embeddings
    )
    if output_embeddings is not None:
        params = param.put(
            params,
            f"{param.get_output_embedding_path(config.model_type)}.weight",
            output_embeddings,
        )

    # Load parameters into the model
    model.load_state_dict(params)
    model = model.to(device)
    
    # Set up distributed model if available
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model)

    # Synchronize all processes
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    torch_kwargs = {}
    
    if args.expand_input_ids:
        # Set up the inputs_embeds computation for expanded input IDs
        def compute_inputs_embeds(model, input_ids, expanded_input_ids):
            input_embeddings = getattr(model, param.get_input_embedding_path(config.model_type)).weight
            
            standard_inputs_embeds = input_embeddings[input_ids]
            expanded_inputs_embeds = torch.tensor(expand_input_ids_embeddings, 
                                                 device=standard_inputs_embeds.device)[expanded_input_ids]
            
            return standard_inputs_embeds + expanded_inputs_embeds

        def score_fn_wrapper(model_fn, params, model_args, labels, suffix_mask, space_mask, logit_mask, atol=ATOL):
            input_ids = model_args[0]
            
            expanded_input_ids = utils.expand_input_ids(
                input_ids.cpu().numpy() if isinstance(input_ids, torch.Tensor) else input_ids,
                tokenizer=tokenizer,
                original_vocab=expand_vocab,
                use_heuristic=True,
                maxlen=16,
            )
            
            if isinstance(input_ids, torch.Tensor):
                expanded_input_ids = torch.tensor(expanded_input_ids, device=input_ids.device)
            
            inputs_embeds = compute_inputs_embeds(model, input_ids, expanded_input_ids)
            
            return score(
                model_fn,
                params,
                (None, inputs_embeds),
                labels=labels,
                suffix_mask=suffix_mask,
                space_mask=space_mask,
                logit_mask=logit_mask,
                atol=atol,
            )
        
        torch_kwargs["expand_input_ids"] = True
        torch_kwargs["expand_input_ids_vocab"] = expand_vocab
        torch_kwargs["expand_input_ids_embeddings"] = expand_input_ids_embeddings
        torch_kwargs["score_fn"] = score_fn_wrapper

    # Run evaluation
    results, _ = evaluate(
        model=model,
        config=config,
        params=params,
        tokenizer=tokenizer,
        logit_mask=logit_mask,
        **eval_kwargs,
        torch_kwargs=torch_kwargs,
    )

    # Print results on the main process
    if not torch.distributed.is_available() or torch.distributed.get_rank() == 0:
        pprint(results)


if __name__ == "__main__":
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    my_app()