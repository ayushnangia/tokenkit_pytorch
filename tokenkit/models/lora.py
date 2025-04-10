import re
import torch
import numpy as np
from typing import Dict, Tuple, List, Any, Optional
import logging

from tokenkit import utils

logger = logging.getLogger(__name__)

# LoRA patterns for different model types
LORA_PATTERNS = {
    "llama": [
        ".*self_attn.(q_proj|k_proj|v_proj).weight",
        ".*self_attn.o_proj.weight",
        ".*mlp.down_proj.weight",
        ".*mlp.up_proj.weight",
        ".*mlp.gate_proj.weight",
    ],
    "gemma2": [
        ".*self_attn.(q_proj|k_proj|v_proj).weight",
        ".*self_attn.o_proj.weight",
        ".*mlp.down_proj.weight",
        ".*mlp.up_proj.weight",
        ".*mlp.gate_proj.weight",
    ],
}


class LoRALayer(torch.nn.Module):
    """Implementation of a LoRA (Low-Rank Adaptation) layer."""
    
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        
        # Initialize A with small random values
        self.lora_A = torch.nn.Parameter(torch.randn(rank, in_features) * 0.02)
        # Initialize B to zeros for zero init of the adapter
        self.lora_B = torch.nn.Parameter(torch.zeros(out_features, rank))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Low-rank update: BA 
        return self.scale * torch.matmul(self.lora_B, torch.matmul(self.lora_A, x.T)).T


def init_lora_layers(model, model_type: str, rank: int, alpha: float = 1.0) -> Dict[str, LoRALayer]:
    """Initialize LoRA layers based on patterns for the given model type.
    
    Args:
        model: The PyTorch model to add LoRA layers to
        model_type: Type of model (llama, gemma2, etc.)
        rank: Rank of LoRA layers
        alpha: Scaling factor
        
    Returns:
        Dictionary mapping parameter names to LoRA layers
    """
    lora_patterns = LORA_PATTERNS.get(model_type, [])
    lora_layers = {}
    
    # Collect params that should have LoRA applied
    for name, param in model.named_parameters():
        for pattern in lora_patterns:
            if re.match(pattern, name) and len(param.shape) == 2:
                out_features, in_features = param.shape
                
                # Create a LoRA layer
                lora_layers[name] = LoRALayer(
                    in_features=in_features,
                    out_features=out_features,
                    rank=rank,
                    alpha=alpha
                )
                
                logger.info(f"Initialized LoRA layer for {name} with shape {param.shape}")
                
                # Move to same device as the model parameter
                lora_layers[name].to(param.device)
                break
    
    return lora_layers


def init_lora_params(args, model, model_type, seed, dtype=torch.float32):
    """Initialize LoRA parameters using the configuration and random seed.
    
    Args:
        args: Config arguments
        model: The PyTorch model
        model_type: Type of model (llama, gemma2, etc.)
        seed: Random seed for initialization
        dtype: Data type for parameters
        
    Returns:
        Dictionary of LoRA parameters
    """
    torch.manual_seed(seed)
    lora_patterns = LORA_PATTERNS.get(model_type, [])
    lora_rank = args.model_lora_rank
    stddev = 1.0 / lora_rank
    
    lora_params = {}
    
    for name, param in model.named_parameters():
        for pattern in lora_patterns:
            if re.match(pattern, name) and len(param.shape) == 2:
                out_features, in_features = param.shape
                
                # Initialize B to zeros
                b = torch.zeros((out_features, lora_rank), dtype=dtype)
                # Initialize A with small random weights
                a = torch.randn((lora_rank, in_features), dtype=dtype) * stddev
                
                lora_params[name] = {"a": a, "b": b}
                break
    
    return lora_params


def materialize_lora(params, lora_params, alpha):
    """Apply LoRA updates to model parameters.
    
    Args:
        params: Model parameters as a state dict
        lora_params: LoRA parameters dictionary
        alpha: Scaling factor for LoRA
        
    Returns:
        Updated parameters with LoRA applied
    """
    result = {}
    for name, param in params.items():
        if name in lora_params:
            a, b = lora_params[name]["a"], lora_params[name]["b"]
            scale = alpha / b.shape[-1]
            
            # Apply LoRA update: original + scale * (B @ A)
            result[name] = param + scale * torch.matmul(b, a)
        else:
            result[name] = param
            
    return result


def dematerialize_lora(params, lora_params, alpha):
    """Remove LoRA updates from model parameters.
    
    Args:
        params: Model parameters with LoRA already applied
        lora_params: LoRA parameters dictionary
        alpha: Scaling factor for LoRA
        
    Returns:
        Original parameters with LoRA removed
    """
    result = {}
    for name, param in params.items():
        if name in lora_params:
            a, b = lora_params[name]["a"], lora_params[name]["b"]
            scale = alpha / b.shape[-1]
            
            # Remove LoRA update: full - scale * (B @ A)
            result[name] = param - scale * torch.matmul(b, a)
        else:
            result[name] = param
            
    return result


class LoRAWrapper(torch.nn.Module):
    """Wrapper around a PyTorch model for applying LoRA to specific layers."""
    
    def __init__(self, model, lora_params, alpha=1.0):
        super().__init__()
        self.model = model
        self.lora_layers = {}
        self.alpha = alpha
        
        # Create LoRA layers for each parameter in lora_params
        for name, param_dict in lora_params.items():
            # Find the parameter in the model
            param = None
            for n, p in model.named_parameters():
                if n == name:
                    param = p
                    break
                    
            if param is not None:
                out_features, in_features = param.shape
                
                # Create LoRA layer and initialize with provided params
                self.lora_layers[name] = LoRALayer(
                    in_features=in_features,
                    out_features=out_features,
                    rank=param_dict["a"].shape[0],
                    alpha=alpha
                )
                
                # Set weights from lora_params
                with torch.no_grad():
                    self.lora_layers[name].lora_A.copy_(param_dict["a"])
                    self.lora_layers[name].lora_B.copy_(param_dict["b"])
                
                # Register LoRA layers as modules
                self.add_module(f"lora_{name}", self.lora_layers[name])
    
    def apply_lora_to_forward(self, module, input_tensors, lora_names):
        """Modifies the forward pass to incorporate LoRA updates."""
        # Store original forward method
        original_forward = module.forward
        
        # Define the hook function for named modules
        def hook_fn(orig_module, name):
            def modified_forward(*args, **kwargs):
                result = original_forward(*args, **kwargs)
                
                # Apply LoRA updates to matching parameters
                for lora_name in lora_names:
                    if lora_name.split(".")[-2:] == [name, "weight"]:
                        # Find the corresponding input tensor
                        input_tensor = args[0]  # Typically the first argument
                        
                        # Apply LoRA update
                        lora_output = self.lora_layers[lora_name](input_tensor)
                        
                        # Add to the result
                        if isinstance(result, tuple):
                            # Some modules return tuples - modify the first element
                            result = (result[0] + lora_output,) + result[1:]
                        else:
                            result = result + lora_output
                        
                return result
            
            return modified_forward
        
        # Apply hooks to all matching modules
        for name, child_module in module.named_modules():
            for lora_name in lora_names:
                if lora_name.startswith(name):
                    child_module.forward = hook_fn(child_module, name)
        
        return module.forward(*input_tensors)
    
    def forward(self, *args, **kwargs):
        """Forward pass with LoRA updates."""
        # Get a list of names where LoRA is applied
        lora_names = list(self.lora_layers.keys())
        
        # Apply LoRA to the model's forward pass
        return self.apply_lora_to_forward(self.model, args, lora_names)