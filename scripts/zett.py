import logging
from pprint import pformat

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM

from tokenkit import utils
from tokenkit.byteify import load_byteify_tokenizer
from tokenkit.models import param, sharding

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="zett")
def my_app(args: DictConfig) -> None:
    logger.info(pformat(OmegaConf.to_object(args)))

    # Load the source and target tokenizers
    source_tokenizer = load_byteify_tokenizer(args.source_tokenizer_name)
    target_tokenizer = load_byteify_tokenizer(args.target_tokenizer_name)

    # Initialize distributed if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model config
    config = AutoConfig.from_pretrained(args.source_model_pretrained_name_or_path)
    
    # Load the model parameters
    model_params = param.load_params(
        pretrained_model_name_or_path=args.source_model_pretrained_name_or_path
    )

    # Get the embeddings and model parameters
    embeddings, model_params = param.stack_embeddings(
        model_params,
        config,
        pop_embeddings=True,
    )

    # Convert embeddings to numpy for FVT
    if torch.is_tensor(embeddings):
        embeddings_np = embeddings.cpu().numpy()
    else:
        embeddings_np = embeddings

    # Apply Fast Vocabulary Transfer (FVT)
    diff_embeddings, original_to_new_indices, diff_indices = utils.fvt(
        source_tokenizer,
        target_tokenizer,
        embeddings_np,
    )
    
    # Update embeddings with FVT results
    if torch.is_tensor(embeddings):
        original_to_new_indices_tensor = torch.tensor(original_to_new_indices, dtype=torch.long)
        new_embeddings = embeddings[original_to_new_indices_tensor]
        
        if len(diff_indices) > 0:
            diff_indices_tensor = torch.tensor(diff_indices, dtype=torch.long)
            diff_embeddings_tensor = torch.tensor(diff_embeddings, dtype=embeddings.dtype)
            new_embeddings[diff_indices_tensor] = diff_embeddings_tensor
    else:
        new_embeddings = embeddings[original_to_new_indices]
        if len(diff_indices) > 0:
            new_embeddings[diff_indices] = diff_embeddings

    # Assign the new embeddings to the model parameters
    model_params = param.assign_embeddings(model_params, new_embeddings, config)
    
    # Initialize the model architecture
    model = AutoModelForCausalLM.from_config(config)
    
    # Load the modified parameters into the model
    model.load_state_dict(model_params)

    # Save the model, configuration, and target tokenizer
    model.save_pretrained(args.output)
    config.save_pretrained(args.output)
    target_tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    my_app()