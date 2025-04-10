import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from tokenkit import utils

logger = logging.getLogger(__name__)

# Parameter patterns for model parallel and data parallel configurations
# These will be used to determine how to split parameters across GPUs in model parallel mode
MODEL_PARALLEL_PATTERNS = {
    "hypernet": {
        ".*ffn_layer1.linear": "column",  # split by columns (output dim)
        ".*ffn_layer2.linear": "row",     # split by rows (input dim)
        ".*self_attention.(query|key|value).weight": "column",
        ".*self_attention.post.weight": "row",
        ".*embeddings": "row",
    },
    "llama": {
        ".*embed_tokens.*embedding": "row",
        ".*self_attn.(q_proj|k_proj|v_proj).weight": "column",
        ".*self_attn.o_proj.weight": "row",
        ".*lm_head.weight": "column",
        ".*mlp.down_proj.weight": "row",
        ".*mlp.up_proj.weight": "column",
        ".*mlp.gate_proj.weight": "column",
    },
    "gemma2": {
        ".*embed_tokens.*weight": "row",
        ".*self_attn.(q_proj|k_proj|v_proj).weight": "column",
        ".*self_attn.o_proj.weight": "row",
        ".*lm_head.weight": "column", 
        ".*mlp.down_proj.weight": "row",
        ".*mlp.up_proj.weight": "column",
        ".*mlp.gate_proj.weight": "column",
    },
    "mistral": {
        ".*embed_tokens.*weight": "row",
        ".*self_attn.(q_proj|k_proj|v_proj).weight": "column",
        ".*self_attn.o_proj.weight": "row",
        ".*lm_head.weight": "column",
        ".*mlp.down_proj.weight": "row",
        ".*mlp.up_proj.weight": "column",
        ".*mlp.gate_proj.weight": "column",
    },
    "gpt2": {
        ".*c_attn.weight": "column",
        ".*c_proj.weight": "row",
        ".*c_fc.weight": "column",
    },
}


def is_distributed() -> bool:
    """Check if distributed training is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Get the number of processes in the distributed training."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get the rank of the current process."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    """Get the local rank of the current process."""
    if is_distributed():
        return int(os.environ.get("LOCAL_RANK", "0"))
    return 0


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def get_device() -> torch.device:
    """Get the device for the current process."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{get_local_rank()}")
    return torch.device("cpu")


def initialize_distributed(backend: str = "nccl") -> None:
    """Initialize distributed training if not already initialized.
    
    Args:
        backend: PyTorch distributed backend ('nccl' for GPUs, 'gloo' for CPU)
    """
    if not dist.is_available():
        logger.warning("Distributed training not available")
        return
    
    if not dist.is_initialized():
        # Check if environment variables are set
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            logger.warning("Distributed environment variables not set, skipping distributed initialization")
            return
        
        # Initialize process group
        dist.init_process_group(backend=backend)
        
        # Set device
        if torch.cuda.is_available():
            local_rank = get_local_rank()
            torch.cuda.set_device(local_rank)
            logger.info(f"Initialized distributed training on rank {get_rank()}/{get_world_size()} (local rank: {local_rank})")
        else:
            logger.info(f"Initialized distributed CPU training on rank {get_rank()}/{get_world_size()}")


def get_model_parallel_config(model_type: str) -> Dict[str, str]:
    """Get the model parallel configuration for a specific model type.
    
    Args:
        model_type: Type of model (llama, gemma2, etc.)
        
    Returns:
        Dictionary mapping parameter patterns to their sharding strategy
    """
    return MODEL_PARALLEL_PATTERNS.get(model_type, {})


def wrap_model_for_distributed(
    model: torch.nn.Module,
    model_type: str = None,
    use_model_parallel: bool = False,
    model_parallel_size: int = None,
    find_unused_parameters: bool = False
) -> torch.nn.Module:
    """Wrap a PyTorch model for distributed training.
    
    Args:
        model: PyTorch model to wrap
        model_type: Type of model (llama, gemma2, etc.) for model parallel config
        use_model_parallel: Whether to use model parallelism
        model_parallel_size: Number of GPUs to use for model parallelism
        find_unused_parameters: Whether to find unused parameters in DDP
        
    Returns:
        Wrapped model
    """
    if not is_distributed():
        logger.info("Not running in distributed mode, returning original model")
        return model
    
    # Move model to the correct device
    device = get_device()
    model.to(device)
    
    # If not using model parallelism, simply wrap with DDP
    if not use_model_parallel:
        logger.info("Using data parallel training with DistributedDataParallel")
        return DistributedDataParallel(
            model,
            device_ids=[get_local_rank()] if torch.cuda.is_available() else None,
            output_device=get_local_rank() if torch.cuda.is_available() else None,
            find_unused_parameters=find_unused_parameters
        )
    
    # If using model parallelism, we need a more complex setup
    # This is a simplified implementation that would need to be extended for full model parallelism
    logger.warning("Model parallelism is not fully implemented yet, using data parallelism only")
    return DistributedDataParallel(
        model,
        device_ids=[get_local_rank()] if torch.cuda.is_available() else None,
        output_device=get_local_rank() if torch.cuda.is_available() else None,
        find_unused_parameters=find_unused_parameters
    )


def split_batch_for_data_parallel(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Split a batch across data parallel devices.
    
    Args:
        batch: Dictionary of tensors representing a batch
        
    Returns:
        Batch with tensors split across devices
    """
    if not is_distributed():
        return batch
    
    rank = get_rank()
    world_size = get_world_size()
    
    # Split batch along the first dimension
    result = {}
    for key, tensor in batch.items():
        if tensor.dim() == 0:
            # Scalar tensors shouldn't be split
            result[key] = tensor
        else:
            # Get the size of each split
            batch_size = tensor.size(0)
            split_size = batch_size // world_size
            if batch_size % world_size != 0:
                logger.warning(f"Batch size {batch_size} not divisible by world size {world_size}")
                
            # Calculate start and end indices for this rank
            start_idx = rank * split_size
            end_idx = start_idx + split_size if rank < world_size - 1 else batch_size
            
            # Extract this rank's portion of the batch
            result[key] = tensor[start_idx:end_idx]
    
    return result


def gather_from_data_parallel(tensor: torch.Tensor) -> torch.Tensor:
    """Gather a tensor from all data parallel devices.
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        Gathered tensor
    """
    if not is_distributed():
        return tensor
    
    # Create a list to hold the gathered tensors
    world_size = get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    
    # Gather tensors from all processes
    dist.all_gather(gathered_tensors, tensor)
    
    # Concatenate along the first dimension
    return torch.cat(gathered_tensors, dim=0)


def synchronize_model_parameters(model: torch.nn.Module) -> None:
    """Ensure all processes have the same model parameters.
    
    Args:
        model: Model to synchronize
    """
    if not is_distributed():
        return
    
    # Get parameters from rank 0
    for param in model.parameters():
        # Broadcast from rank 0 to all processes
        dist.broadcast(param.data, src=0)


def all_reduce_tensors(tensors: List[torch.Tensor], op: str = "mean") -> List[torch.Tensor]:
    """Perform all-reduce operation on a list of tensors.
    
    Args:
        tensors: List of tensors to reduce
        op: Reduction operation ('mean', 'sum', 'max', 'min')
        
    Returns:
        List of reduced tensors
    """
    if not is_distributed():
        return tensors
    
    # Map operation string to torch.distributed operation
    op_map = {
        "mean": dist.ReduceOp.SUM,
        "sum": dist.ReduceOp.SUM,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
    }
    
    if op not in op_map:
        raise ValueError(f"Unsupported reduce operation: {op}, must be one of {list(op_map.keys())}")
    
    reduce_op = op_map[op]
    world_size = get_world_size()
    
    # Perform all-reduce on each tensor
    result = []
    for tensor in tensors:
        # Clone to avoid modifying the original tensor
        reduced = tensor.clone()
        dist.all_reduce(reduced, op=reduce_op)
        
        # If operation is 'mean', divide by world size
        if op == "mean":
            reduced = reduced / world_size
            
        result.append(reduced)
    
    return result


def initialize_parallel_training(
    model: torch.nn.Module,
    model_type: str = None,
    use_model_parallel: bool = False,
    use_distributed: bool = True,
    backend: str = "nccl",
    find_unused_parameters: bool = False
) -> torch.nn.Module:
    """Set up a model for parallel training.
    
    Args:
        model: PyTorch model to wrap
        model_type: Type of model (llama, gemma2, etc.) for model parallel config
        use_model_parallel: Whether to use model parallelism
        use_distributed: Whether to use distributed training
        backend: PyTorch distributed backend
        find_unused_parameters: Whether to find unused parameters in DDP
        
    Returns:
        Wrapped model
    """
    # Initialize distributed training if requested
    if use_distributed:
        initialize_distributed(backend=backend)
    
    # Wrap the model for distributed training if initialized
    if is_distributed():
        model = wrap_model_for_distributed(
            model,
            model_type=model_type,
            use_model_parallel=use_model_parallel,
            find_unused_parameters=find_unused_parameters
        )
    else:
        # If not distributed, move to device (typically a single GPU)
        device = get_device()
        model.to(device)
        
    return model