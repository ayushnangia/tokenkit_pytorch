import logging
import re
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import Module, Parameter
from torch.nn.utils import clip_grad_norm_

logger = logging.getLogger(__name__)


def decay_mask_fn(name: str, param: Parameter) -> bool:
    """Return whether to apply weight decay to the parameter.
    
    Args:
        name: Parameter name
        param: The parameter
        
    Returns:
        True if weight decay should be applied, False otherwise
    """
    # Don't decay bias, layernorms, layer norms, or lns
    if "bias" in name or name.endswith(".b"):
        return False
    if "layernorm" in name.lower() or "layer_norm" in name.lower() or ".ln" in name.lower():
        return False
    return True


class ModelOptimizer:
    """PyTorch optimizer wrapper that supports parameter groups and grad accumulation."""
    
    def __init__(
        self,
        model: Module,
        train_mask: Dict[str, bool], 
        learning_rate_fn: Callable[[Optimizer], LambdaLR],
        **optimizer_kwargs
    ):
        """Initialize the optimizer.
        
        Args:
            model: The model to optimize
            train_mask: Dictionary mapping parameter names to whether they should be trained
            learning_rate_fn: Function that creates an LR scheduler when called with an optimizer
            **optimizer_kwargs: Additional optimizer arguments
        """
        self.model = model
        self.opt_type = optimizer_kwargs.pop("type")
        self.grad_acc_steps = optimizer_kwargs.pop("grad_acc_steps", 1)
        self.max_grad_norm = optimizer_kwargs.pop("max_grad_norm", None)
        self.current_step = 0
        
        # Set up parameter groups
        param_groups = []
        special_groups = {}
        configured_groups = optimizer_kwargs.pop("param_groups", [])
        
        # First, separate parameters by trainable/non-trainable and special groups
        for name, param in model.named_parameters():
            # Check if parameter should be trained according to train_mask
            trainable = self._is_trainable(name, train_mask)
            if not trainable:
                param.requires_grad = False
                continue
                
            # Check if parameter belongs to a special group
            group_name = None
            lr_scale = 1.0
            for group in configured_groups:
                if re.match(group["pattern"], name):
                    group_name = group["pattern"]
                    lr_scale = group["lr_scale"]
                    break
                    
            if group_name is not None:
                if group_name not in special_groups:
                    special_groups[group_name] = {
                        "params": [],
                        "names": [],
                        "lr_scale": lr_scale,
                    }
                special_groups[group_name]["params"].append(param)
                special_groups[group_name]["names"].append(name)
            else:
                # Default group
                weight_decay = optimizer_kwargs.get("weight_decay", 0.0)
                param_groups.append({
                    "params": [param],
                    "weight_decay": weight_decay if decay_mask_fn(name, param) else 0.0,
                    "lr_scale": 1.0,
                    "name": name
                })
                
        # Set up special groups
        for group_name, group in special_groups.items():
            weight_decay = optimizer_kwargs.get("weight_decay", 0.0)
            for param, name in zip(group["params"], group["names"]):
                param_groups.append({
                    "params": [param],
                    "weight_decay": weight_decay if decay_mask_fn(name, param) else 0.0,
                    "lr_scale": group["lr_scale"],
                    "name": name
                })

        # Log special parameter groups
        logger.info("Special parameter groups:")
        pprint({
            g["name"]: g["lr_scale"] 
            for g in param_groups 
            if g["lr_scale"] != 1.0
        })
        
        # Create optimizer
        if self.opt_type.lower() == "adamw":
            self.optimizer = AdamW(param_groups, **optimizer_kwargs)
        else:
            raise ValueError(f"Unknown optimizer type: {self.opt_type}")
            
        # Create scheduler
        self.scheduler = learning_rate_fn(self.optimizer)
        
        # Reset gradients
        self.zero_grad()
        
    def _is_trainable(self, name: str, train_mask: Dict[str, bool]) -> bool:
        """Check if a parameter should be trained according to train_mask."""
        # Convert flattened keys like 'a.b.c' to nested structure
        nested_name = name.split('.')
        
        # Navigate through the train_mask following the parameter name hierarchy
        current = train_mask
        for part in nested_name:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                # If we can't navigate down the exact path, check if any pattern matches
                found = False
                if isinstance(current, dict):
                    for k, v in current.items():
                        if re.match(k, part):
                            current = v
                            found = True
                            break
                if not found:
                    # Default to True if we can't find a specific mask entry
                    return True
                    
        # If we navigated to a boolean value, that's our result
        if isinstance(current, bool):
            return current
            
        # Default to trainable
        return True
        
    def step(self):
        """Perform a single optimization step."""
        if self.current_step % self.grad_acc_steps == 0:
            if self.max_grad_norm is not None:
                # Clip gradients
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
        self.current_step += 1
        
    def zero_grad(self):
        """Zero out gradients."""
        self.optimizer.zero_grad()
        
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


def get_optimizer(model: Module, train_mask: Dict[str, bool], learning_rate_fn, **optimizer_kwargs) -> ModelOptimizer:
    """Create an optimizer for the model.
    
    Args:
        model: The model to optimize
        train_mask: Dictionary mapping parameter names to whether they should be trained
        learning_rate_fn: Function that creates an LR scheduler when called with an optimizer
        **optimizer_kwargs: Additional optimizer arguments
        
    Returns:
        ModelOptimizer: The configured optimizer wrapper
    """
    return ModelOptimizer(model, train_mask, learning_rate_fn, **optimizer_kwargs)