import math
from typing import Callable, List, Optional
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class LinearWarmupLinearDecayScheduler(LambdaLR):
    """PyTorch implementation of linear warmup followed by linear decay."""
    
    def __init__(
        self, 
        optimizer: Optimizer,
        lr: float,
        total_steps: int,
        warmup_steps: int,
        prefix_steps: int = 0,
        prefix_lr: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.lr = lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.prefix_steps = prefix_steps
        self.prefix_lr = prefix_lr
        
        def lr_lambda(current_step: int) -> float:
            if current_step < prefix_steps:
                # Linear increase from 0 to prefix_lr during prefix phase
                return float(current_step) * prefix_lr / max(1.0, prefix_steps)
            elif current_step < prefix_steps + warmup_steps:
                # Linear increase from prefix_lr to lr during warmup
                step_in_warmup = current_step - prefix_steps
                base = prefix_lr if prefix_steps > 0 else 0.0
                return base + float(step_in_warmup) * (lr - base) / max(1.0, warmup_steps)
            else:
                # Linear decay from lr to 0
                step_in_decay = current_step - prefix_steps - warmup_steps
                decay_steps = max(1.0, total_steps - warmup_steps - prefix_steps)
                return max(0.0, lr * (1.0 - float(step_in_decay) / decay_steps))
        
        super().__init__(optimizer, lr_lambda, last_epoch, verbose)


class LinearWarmupCosineDecayScheduler(LambdaLR):
    """PyTorch implementation of linear warmup followed by cosine decay."""
    
    def __init__(
        self, 
        optimizer: Optimizer,
        lr: float,
        total_steps: int,
        warmup_steps: int,
        alpha: float = 0.0,
        prefix_steps: int = 0,
        prefix_lr: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.lr = lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.alpha = alpha
        self.prefix_steps = prefix_steps
        self.prefix_lr = prefix_lr
        
        def lr_lambda(current_step: int) -> float:
            if current_step < prefix_steps:
                # Linear increase from 0 to prefix_lr during prefix phase
                return float(current_step) * prefix_lr / max(1.0, prefix_steps)
            elif current_step < prefix_steps + warmup_steps:
                # Linear increase from prefix_lr to lr during warmup
                step_in_warmup = current_step - prefix_steps
                base = prefix_lr if prefix_steps > 0 else 0.0
                return base + float(step_in_warmup) * (lr - base) / max(1.0, warmup_steps)
            else:
                # Cosine decay from lr to alpha*lr
                step_in_decay = current_step - prefix_steps - warmup_steps
                decay_steps = max(1.0, total_steps - warmup_steps - prefix_steps)
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * step_in_decay / decay_steps))
                decayed = (1 - alpha) * cosine_decay + alpha
                return lr * decayed
        
        super().__init__(optimizer, lr_lambda, last_epoch, verbose)


# Convenience functions to match original API
def linear_warmup_linear_decay_with_linear_prefix(
    lr: float, steps: int, warmup_steps: int, prefix_steps: int = 0, prefix_lr: float = 0.0
) -> Callable[[Optimizer], LambdaLR]:
    """Returns a function that creates a linear warmup, linear decay scheduler."""
    
    def _create_scheduler(optimizer: Optimizer) -> LambdaLR:
        return LinearWarmupLinearDecayScheduler(
            optimizer=optimizer,
            lr=lr,
            total_steps=steps,
            warmup_steps=warmup_steps,
            prefix_steps=prefix_steps,
            prefix_lr=prefix_lr
        )
    
    return _create_scheduler


def linear_warmup_cosine_decay_with_linear_prefix(
    lr: float, steps: int, warmup_steps: int, alpha: float = 0.0, 
    prefix_steps: int = 0, prefix_lr: float = 0.0
) -> Callable[[Optimizer], LambdaLR]:
    """Returns a function that creates a linear warmup, cosine decay scheduler."""
    
    def _create_scheduler(optimizer: Optimizer) -> LambdaLR:
        return LinearWarmupCosineDecayScheduler(
            optimizer=optimizer,
            lr=lr,
            total_steps=steps,
            warmup_steps=warmup_steps,
            alpha=alpha,
            prefix_steps=prefix_steps,
            prefix_lr=prefix_lr
        )
    
    return _create_scheduler