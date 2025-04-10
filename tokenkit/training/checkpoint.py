import os
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import torch
import torch.distributed as dist

from tokenkit.models import sharding

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    step: int = 0,
    epoch: int = 0,
    train_mask: Optional[Dict[str, bool]] = None,
    keys_to_keep: Optional[Set[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Save a PyTorch model checkpoint.
    
    Args:
        path: Path to save the checkpoint
        model: PyTorch model to save
        optimizer: Optional optimizer to save state
        scheduler: Optional learning rate scheduler to save state
        step: Current training step
        epoch: Current epoch
        train_mask: Dictionary indicating which parameters are trainable
        keys_to_keep: Set of parameter prefixes to always save
        metadata: Additional metadata to save with the checkpoint
    """
    # Only save from the main process in distributed training
    if sharding.is_distributed() and sharding.get_rank() != 0:
        # Synchronize to ensure all processes wait for checkpoint to be saved
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        return
    
    # Convert to Path object
    path = Path(path)
    
    # Create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract model state dict
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    
    # Filter parameters based on train_mask and keys_to_keep if provided
    if train_mask is not None and keys_to_keep is not None:
        filtered_state = {}
        for name, param in model_state.items():
            param_path_parts = name.split('.')
            
            # Check if this parameter is trainable or has a prefix we want to keep
            is_trainable = False
            should_keep = False
            
            # Check trainability based on train_mask
            current_mask = train_mask
            for part in param_path_parts:
                if not isinstance(current_mask, dict):
                    break
                if part in current_mask:
                    current_mask = current_mask[part]
                    if isinstance(current_mask, bool) and current_mask:
                        is_trainable = True
                        break
            
            # Check if it has a prefix we want to keep
            for prefix in keys_to_keep:
                if name.startswith(prefix):
                    should_keep = True
                    break
            
            if is_trainable or should_keep:
                filtered_state[name] = param
        
        model_state = filtered_state
    
    # Prepare checkpoint dictionary
    checkpoint = {
        'model_state_dict': model_state,
        'step': step,
        'epoch': epoch,
    }
    
    # Add optimizer state if provided
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # Add scheduler state if provided
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Add metadata if provided
    if metadata is not None:
        checkpoint['metadata'] = metadata
    
    # Save the checkpoint
    logger.info(f"Saving checkpoint to {path}")
    torch.save(checkpoint, path)
    
    # Synchronize to ensure all processes wait for checkpoint to be saved
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a PyTorch model checkpoint.
    
    Args:
        path: Path to the checkpoint
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional learning rate scheduler to load state into
        map_location: Device to map tensors to
        strict: Whether to strictly enforce that the keys in state_dict match
        
    Returns:
        Dictionary containing metadata from the checkpoint
    """
    # Convert to Path object
    path = Path(path)
    
    # Check if path exists
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    
    # Determine map_location if not provided
    if map_location is None:
        map_location = sharding.get_device()
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=map_location)
    
    # Load model state
    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Return metadata
    metadata = {
        'step': checkpoint.get('step', 0),
        'epoch': checkpoint.get('epoch', 0),
    }
    
    # Add additional metadata if available
    if 'metadata' in checkpoint:
        metadata.update(checkpoint['metadata'])
    
    # Synchronize to ensure all processes wait for checkpoint to be loaded
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    
    return metadata


def save_embeddings(path: Union[str, Path], embeddings: torch.Tensor):
    """Save embedding weights to a file.
    
    Args:
        path: Path to save the embeddings
        embeddings: Embedding tensor to save
    """
    # Only save from the main process in distributed training
    if sharding.is_distributed() and sharding.get_rank() != 0:
        return
    
    # Convert to Path object
    path = Path(path)
    
    # Create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the embeddings
    logger.info(f"Saving embeddings to {path}")
    torch.save(embeddings, path)


def load_embeddings(path: Union[str, Path], map_location: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    """Load embedding weights from a file.
    
    Args:
        path: Path to the embeddings file
        map_location: Device to map tensors to
        
    Returns:
        Loaded embedding tensor
    """
    # Convert to Path object
    path = Path(path)
    
    # Check if path exists
    if not path.exists():
        raise FileNotFoundError(f"Embeddings not found at {path}")
    
    # Determine map_location if not provided
    if map_location is None:
        map_location = sharding.get_device()
    
    # Load the embeddings
    logger.info(f"Loading embeddings from {path}")
    return torch.load(path, map_location=map_location)


def find_latest_checkpoint(checkpoint_dir: Union[str, Path], prefix: str = "checkpoint") -> Optional[Path]:
    """Find the latest checkpoint in a directory based on step number.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        prefix: Prefix of checkpoint files
        
    Returns:
        Path to the latest checkpoint, or None if no checkpoints found
    """
    # Convert to Path object
    checkpoint_dir = Path(checkpoint_dir)
    
    # Check if directory exists
    if not checkpoint_dir.exists():
        return None
    
    # Find all checkpoints with the specified prefix
    checkpoints = list(checkpoint_dir.glob(f"{prefix}_*.pt"))
    
    if not checkpoints:
        return None
    
    # Extract step numbers from filenames
    def get_step(path):
        try:
            return int(path.stem.split('_')[-1])
        except (ValueError, IndexError):
            return -1
    
    # Return the checkpoint with the highest step number
    return max(checkpoints, key=get_step)


def get_checkpoint_manager(
    checkpoint_dir: Union[str, Path],
    prefix: str = "checkpoint",
    keep_last_n: int = 5,
):
    """Create a checkpoint manager for saving and loading checkpoints.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        prefix: Prefix for checkpoint filenames
        keep_last_n: Number of recent checkpoints to keep
        
    Returns:
        Dictionary with checkpoint management functions
    """
    # Convert to Path object
    checkpoint_dir = Path(checkpoint_dir)
    
    # Create directory if it doesn't exist
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Track checkpoint paths by step
    checkpoints_by_step = {}
    
    def save(
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        step: int = 0,
        epoch: int = 0,
        train_mask: Optional[Dict[str, bool]] = None,
        keys_to_keep: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Save a checkpoint at the current step."""
        # Determine checkpoint path
        path = checkpoint_dir / f"{prefix}_{step}.pt"
        
        # Save the checkpoint
        save_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            epoch=epoch,
            train_mask=train_mask,
            keys_to_keep=keys_to_keep,
            metadata=metadata,
        )
        
        # Update tracking
        checkpoints_by_step[step] = path
        
        # Remove old checkpoints if needed
        _cleanup_old_checkpoints()
        
        return path
    
    def load(
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        step: Optional[int] = None,
        map_location: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load a checkpoint at the specified step or latest."""
        # Determine which checkpoint to load
        if step is not None:
            path = checkpoint_dir / f"{prefix}_{step}.pt"
        else:
            path = find_latest_checkpoint(checkpoint_dir, prefix)
            if path is None:
                raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        
        # Load the checkpoint
        return load_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=map_location,
            strict=strict,
        )
    
    def _cleanup_old_checkpoints():
        """Remove old checkpoints, keeping only the most recent n."""
        if not sharding.is_main_process():
            return
            
        # Only keep the last n checkpoints
        if keep_last_n > 0 and len(checkpoints_by_step) > keep_last_n:
            # Sort steps in descending order
            sorted_steps = sorted(checkpoints_by_step.keys(), reverse=True)
            
            # Keep only the most recent n steps
            steps_to_keep = sorted_steps[:keep_last_n]
            steps_to_remove = sorted_steps[keep_last_n:]
            
            # Remove old checkpoints
            for step in steps_to_remove:
                path = checkpoints_by_step.pop(step)
                if path.exists():
                    path.unlink()
                    logger.info(f"Removed old checkpoint: {path}")
    
    # Scan existing checkpoints in the directory
    existing_checkpoints = list(checkpoint_dir.glob(f"{prefix}_*.pt"))
    for path in existing_checkpoints:
        try:
            # Extract step number from filename
            step = int(path.stem.split('_')[-1])
            checkpoints_by_step[step] = path
        except (ValueError, IndexError):
            logger.warning(f"Could not extract step number from checkpoint filename: {path}")
    
    # Return checkpoint management functions
    return {
        'save': save,
        'load': load,
        'find_latest': lambda: find_latest_checkpoint(checkpoint_dir, prefix),
        'cleanup': _cleanup_old_checkpoints,
    }