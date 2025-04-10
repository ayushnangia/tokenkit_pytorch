"""
Example Usage:

python3 scripts/eval_lockstep.py models=llama_qwen +eval.limit=100
"""

import logging
import os
from pathlib import Path
from pprint import pformat, pprint

import datasets
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM

from tokenkit.byteify import load_byteify_tokenizer
from tokenkit.eval import evaluate_lockstep
from tokenkit.models import param

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


@hydra.main(version_base=None, config_path="../configs", config_name="eval_lockstep")
def my_app(args: DictConfig) -> None:
    logger.info(pformat(OmegaConf.to_object(args)))

    eval_kwargs = OmegaConf.to_object(args.eval)

    # Initialize distributed training if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if eval_kwargs["output"] is not None:
        output_dir = Path(eval_kwargs["output"])
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config=args, f=output_dir / "args.yaml", resolve=True)

    all_models = []
    all_configs = []
    all_params = []
    all_tokenizers = []
    all_logit_masks = []

    eval_kwargs.pop("add_bos")
    all_add_bos = []

    for model_args in args.models:
        model_kwargs = OmegaConf.to_object(model_args)

        print("Loading model...")
        pprint(model_kwargs)

        config = AutoConfig.from_pretrained(model_args.pretrained_model_name_or_path)

        config.max_length = eval_kwargs["lengths"][-1]

        tokenizer = load_byteify_tokenizer(model_kwargs.pop("tokenizer_name"))

        # Load the model architecture
        model = AutoModelForCausalLM.from_config(config)
        
        # Load model parameters
        params = param.load_params(
            pretrained_model_name_or_path=model_args.pretrained_model_name_or_path
        )

        input_embeddings = param.get(
            params, f"{param.get_input_embedding_path(config.model_type)}.weight"
        )

        if len(input_embeddings) < len(tokenizer):
            print("Padding input embeddings...")
            input_embeddings = pad_embeddings(input_embeddings, tokenizer)

        if not config.tie_word_embeddings:
            output_embeddings = param.get(
                params, f"{param.get_output_embedding_path(config.model_type)}.weight"
            )
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

        all_models.append(model)
        all_configs.append(config)
        all_params.append(params)
        all_tokenizers.append(tokenizer)
        all_logit_masks.append(logit_mask)
        all_add_bos.append(model_args.add_bos)

    # Define combine function for model ensemble
    def combine_fn(hidden_states, logits, combine_params, output_embeddings):
        probs = None
        for model_logits in logits:
            model_probs = torch.nn.functional.softmax(model_logits, dim=-1)
            if probs is None:
                probs = model_probs
            else:
                probs += model_probs

        probs /= len(logits)
        return torch.log(probs)

    results = evaluate_lockstep(
        models=all_models,
        configs=all_configs,
        params=all_params,
        tokenizers=all_tokenizers,
        logit_masks=all_logit_masks,
        add_bos=all_add_bos,
        combine_fn=combine_fn,
        combine_params={},
        **eval_kwargs,
    )

    # Print results on the main process
    if not torch.distributed.is_available() or torch.distributed.get_rank() == 0:
        pprint(results[0])


if __name__ == "__main__":
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    my_app()