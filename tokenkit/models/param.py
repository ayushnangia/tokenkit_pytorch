import copy
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoConfig
from transformers.utils.hub import cached_file

logger = logging.getLogger(__name__)

# Model-specific embedding paths
def get_input_embedding_path(model_type: str) -> str:
    """Get the parameter path for input embeddings based on model type."""
    paths = {
        "gpt2": "transformer.wte",
        "roberta": "roberta.embeddings.word_embeddings",
        "xlm-roberta": "roberta.embeddings.word_embeddings",
        "xglm": "model.embed_tokens",
        "mistral": "model.embed_tokens", 
        "llama": "model.embed_tokens",
        "gemma": "model.embed_tokens",
        "gemma2": "model.embed_tokens",
    }
    return paths.get(model_type, "model.embed_tokens")


def get_output_embedding_path(model_type: str) -> Optional[str]:
    """Get the parameter path for output embeddings based on model type."""
    paths = {
        "gpt2": "lm_head",
        "roberta": None,
        "xlm-roberta": None,
        "xglm": None,
        "mistral": "lm_head",
        "llama": "lm_head",
        "gemma": "lm_head",
        "gemma2": "lm_head",
    }
    return paths.get(model_type, "lm_head")


def get_layer_path(model_type: str) -> str:
    """Get the parameter path for transformer layers based on model type."""
    paths = {
        "gemma2": "model.layers",
        "gpt2": "transformer.h",
        "llama": "model.layers",
    }
    return paths.get(model_type, "model.layers")


def load_params(**kwargs) -> Dict[str, Any]:
    """Load parameters from a pretrained model in the Hugging Face format.
    
    Args:
        **kwargs: Arguments to pass to the Hugging Face model loading functions
        
    Returns:
        Dict of model parameters
    """
    kwargs = copy.copy(kwargs)
    config = AutoConfig.from_pretrained(**kwargs)
    path = kwargs.pop("pretrained_model_name_or_path")
    embedding_path = kwargs.pop("embedding_path", None)
    
    # Try to load PyTorch format first
    try:
        # Check if we're loading from a local path or from HF
        if os.path.isdir(path):
            model_path = os.path.join(path, "pytorch_model.bin")
            if not os.path.exists(model_path):
                model_path = os.path.join(path, "model.safetensors")
        else:
            try:
                model_path = cached_file(path, "pytorch_model.bin", **kwargs)
            except OSError:
                model_path = cached_file(path, "model.safetensors", **kwargs)
        
        # Load the model weights
        state_dict = torch.load(model_path, map_location="cpu")
        
    except (OSError, FileNotFoundError):
        logger.warning("Could not load PyTorch model, falling back to manual loading")
        # If there's no PyTorch model, try to get the model config and build the architecture
        try:
            index = cached_file(path, "pytorch_model.bin.index.json", **kwargs)
            index = json.load(open(index))
            files = [cached_file(path, x, **kwargs) for x in set(index["weight_map"].values())]
            
            state_dict = {}
            for file_path in files:
                state_dict.update(torch.load(file_path, map_location="cpu"))
        except OSError:
            logger.error("Failed to load model weights")
            raise ValueError(f"Could not load model from {path}")
    
    # Handle embeddings if specified in a separate file
    if embedding_path is not None:
        embeddings = np.load(embedding_path)
        embed_tensor = torch.from_numpy(embeddings[:, 0])
        
        # Insert into state_dict
        input_path = get_input_embedding_path(config.model_type)
        if input_path:
            state_dict[f"{input_path}.weight"] = embed_tensor
            
        # Handle output embeddings if available
        if embeddings.shape[1] > 1 and not config.tie_word_embeddings:
            output_path = get_output_embedding_path(config.model_type)
            if output_path:
                state_dict[f"{output_path}.weight"] = torch.from_numpy(embeddings[:, 1].T)
    
    return state_dict


def put(state_dict: Dict[str, torch.Tensor], path: str, value: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Set a value in a PyTorch state dict at the given path.
    
    Args:
        state_dict: PyTorch state dictionary
        path: Parameter path (e.g., "model.embed_tokens.weight")
        value: Tensor value to set
        
    Returns:
        Updated state dictionary
    """
    # Create a new dictionary to avoid modifying the input
    result = state_dict.copy()
    
    # Convert numpy arrays to torch tensors if needed
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    
    # Set the value
    result[path] = value
    
    return result


def pop(state_dict: Dict[str, torch.Tensor], path: str) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """Remove and return a value from a PyTorch state dict at the given path.
    
    Args:
        state_dict: PyTorch state dictionary
        path: Parameter path to remove
        
    Returns:
        Tuple of (updated state dict, removed value)
    """
    # Create a new dictionary to avoid modifying the input
    result = state_dict.copy()
    
    # Pop the value if it exists
    value = result.pop(path, None)
    
    return result, value


def get(state_dict: Dict[str, torch.Tensor], path: str) -> Optional[torch.Tensor]:
    """Get a value from a PyTorch state dict at the given path.
    
    Args:
        state_dict: PyTorch state dictionary
        path: Parameter path
        
    Returns:
        The value at the path, or None if not found
    """
    return state_dict.get(path)


def keys(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    """Get all keys in a PyTorch state dict.
    
    Args:
        state_dict: PyTorch state dictionary
        
    Returns:
        List of parameter paths
    """
    return list(state_dict.keys())


def assign_embeddings(state_dict: Dict[str, torch.Tensor], embeddings: Union[np.ndarray, torch.Tensor], config: Any) -> Dict[str, torch.Tensor]:
    """Assign embedding weights to a model state dict.
    
    Args:
        state_dict: PyTorch state dictionary
        embeddings: Embedding weights to assign
        config: Model configuration
        
    Returns:
        Updated state dictionary
    """
    # Convert numpy arrays to torch tensors if needed
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)
    
    # Get the input embedding path
    input_path = f"{get_input_embedding_path(config.model_type)}.weight"
    
    # Assign input embeddings
    state_dict = put(state_dict, input_path, embeddings[:, 0])
    
    # Assign output embeddings if not tied
    if not config.tie_word_embeddings:
        output_path = get_output_embedding_path(config.model_type)
        if output_path:
            state_dict = put(state_dict, f"{output_path}.weight", embeddings[:, 1].T)
    
    return state_dict


def unassign_embeddings(state_dict: Dict[str, torch.Tensor], config: Any) -> Dict[str, torch.Tensor]:
    """Remove embedding weights from a model state dict.
    
    Args:
        state_dict: PyTorch state dictionary
        config: Model configuration
        
    Returns:
        Updated state dictionary with embeddings removed
    """
    # Get the input embedding path
    input_path = f"{get_input_embedding_path(config.model_type)}.weight"
    
    # Remove input embeddings
    state_dict, _ = pop(state_dict, input_path)
    
    # Remove output embeddings if not tied
    if not config.tie_word_embeddings:
        output_path = get_output_embedding_path(config.model_type)
        if output_path:
            state_dict, _ = pop(state_dict, f"{output_path}.weight")
    
    return state_dict


def stack_embeddings(state_dict: Dict[str, torch.Tensor], config: Any, pop_embeddings: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Get embeddings from a model state dict and optionally remove them.
    
    Args:
        state_dict: PyTorch state dictionary
        config: Model configuration
        pop_embeddings: Whether to remove the embeddings from the state dict
        
    Returns:
        Tuple of (stacked embeddings, updated state dict)
    """
    # Get the input embedding path
    input_path = f"{get_input_embedding_path(config.model_type)}.weight"
    
    if config.tie_word_embeddings:
        # If embeddings are tied, just use the input embeddings
        input_embeddings = get(state_dict, input_path)
        if input_embeddings is None:
            raise ValueError(f"Could not find input embeddings at {input_path}")
        
        embeddings = input_embeddings.unsqueeze(1)
    else:
        # If embeddings are not tied, get both input and output embeddings
        input_embeddings = get(state_dict, input_path)
        if input_embeddings is None:
            raise ValueError(f"Could not find input embeddings at {input_path}")
        
        output_path = f"{get_output_embedding_path(config.model_type)}.weight"
        output_embeddings = get(state_dict, output_path)
        if output_embeddings is None:
            logger.warning(f"Could not find output embeddings at {output_path}, using input embeddings")
            output_embeddings = input_embeddings.T
        
        embeddings = torch.stack([input_embeddings, output_embeddings.T], dim=1)
    
    # Remove embeddings if requested
    if pop_embeddings:
        state_dict = unassign_embeddings(state_dict, config)
    
    return embeddings, state_dict


def get_num_layers(config: Any) -> int:
    """Get the number of layers in a model configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Number of layers
    """
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    elif hasattr(config, "n_layer"):  # gpt2
        return config.n_layer
    else:
        raise ValueError("Could not determine number of layers from config")


def set_num_layers(config: Any, num_layers: int) -> None:
    """Set the number of layers in a model configuration.
    
    Args:
        config: Model configuration
        num_layers: Number of layers to set
    """
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = num_layers
    elif hasattr(config, "n_layer"):  # gpt2
        config.n_layer = num_layers
    else:
        raise ValueError("Could not determine number of layers from config")


def strip_layers(state_dict: Dict[str, torch.Tensor], config: Any, n_keep: int = 1) -> Dict[str, torch.Tensor]:
    """Remove layers from a model state dict, keeping only the first n_keep layers.
    
    Args:
        state_dict: PyTorch state dictionary
        config: Model configuration
        n_keep: Number of layers to keep
        
    Returns:
        Updated state dictionary with removed layers
    """
    # Get the layer path for this model type
    layer_path = get_layer_path(config.model_type)
    
    # Get the total number of layers
    total_layers = get_num_layers(config)
    
    # Create a new state dict with only the layers we want to keep
    result = {}
    for key, value in state_dict.items():
        # Check if this key corresponds to a layer we want to remove
        keep = True
        for layer_idx in range(n_keep, total_layers):
            if f"{layer_path}.{layer_idx}." in key:
                keep = False
                break
        
        if keep:
            result[key] = value
    
    # Update the config
    set_num_layers(config, n_keep)
    
    return result