#!/usr/bin/env python
"""
Cross-Tokenizer Distillation Script (PyTorch Implementation)

This script implements cross-tokenizer distillation using techniques like 
Approximate Likelihood Matching (ALM) to transfer knowledge between 
language models with different tokenizers.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, get_scheduler

import wandb
from tokenkit import utils
from tokenkit.byteify import load_byteify_tokenizer
from tokenkit.models import lora, param, sharding
from tokenkit.models.hypernet import Hypernet
from tokenkit.training import checkpoint, collators, losses, lr, opt
from tokenkit.training.collators import TokenizerAlignerCollator, TokenizerAlignerDataset

logger = logging.getLogger(__name__)


class CrossTokenizerDistiller:
    """Main class for cross-tokenizer distillation."""
    
    def __init__(self, args):
        """Initialize the distiller.
        
        Args:
            args: Configuration arguments
        """
        self.args = args
        self.device = sharding.get_device()
        self.initialize_tokenizers()
        self.initialize_models()
        self.initialize_embeddings()
        self.initialize_mappings()
        self.initialize_dataloaders()
        self.initialize_training()
        
        # Set up checkpoint directory
        self.checkpoint_dir = Path(args.output_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize checkpoint manager
        self.checkpoint_manager = checkpoint.get_checkpoint_manager(
            checkpoint_dir=self.checkpoint_dir,
            prefix="checkpoint",
            keep_last_n=args.keep_last_n_checkpoints
        )
        
    def initialize_tokenizers(self):
        """Initialize tokenizers for teacher and student models."""
        logger.info("Loading tokenizers...")
        
        # Load teacher tokenizer
        self.tokenizer_teacher = load_byteify_tokenizer(self.args.teacher.tokenizer_name)
        logger.info(f"Loaded teacher tokenizer: {self.args.teacher.tokenizer_name}")
        
        # Load student tokenizer
        self.tokenizer_new = load_byteify_tokenizer(self.args.new.tokenizer_name)
        logger.info(f"Loaded student tokenizer: {self.args.new.tokenizer_name}")
        
        # Load original tokenizer if using expanded input IDs
        if self.args.add_expanded_input_ids:
            self.tokenizer_original = load_byteify_tokenizer(self.args.original.tokenizer_name)
            logger.info(f"Loaded original tokenizer: {self.args.original.tokenizer_name}")
        else:
            self.tokenizer_original = None
        
        # Create space masks for tokenizers
        self.space_mask_teacher = utils.get_space_mask(self.tokenizer_teacher)
        self.space_mask_new = utils.get_space_mask(self.tokenizer_new)
        
    def initialize_mappings(self):
        """Initialize token mappings for side path distillation and mined mappings."""
        logger.info("Initializing token mappings...")
        
        # Load mined mappings if needed for baseline_mined loss
        if hasattr(self.args, 'tokenizer_pair_data_path') and any(
            loss in self.args.losses for loss in ["baseline_mined"]
        ):
            mined_path = Path(self.args.tokenizer_pair_data_path) / "mined_mapping.npy"
            if mined_path.exists():
                self.mined_mapping = np.load(mined_path).tolist()
                logger.info(f"Loaded mined mapping with {len(self.mined_mapping)} token pairs")
                
                # Load mined distances for info
                distances_path = Path(self.args.tokenizer_pair_data_path) / "mined_distances.json"
                if distances_path.exists():
                    with open(distances_path, 'r') as f:
                        mined_distances = json.load(f)
                    logger.info(f"Average MinED distance: {np.mean(list(mined_distances.values())):.4f}")
            else:
                self.mined_mapping = None
                logger.warning(f"Mined mapping file not found at {mined_path}, setting to None")
        else:
            self.mined_mapping = None
            
        # Set up side path mappings if needed
        if any(loss.startswith("alm_side_path") for loss in self.args.losses):
            if hasattr(self.args, 'side_path_mapping_mode'):
                # Get side path mappings based on mode
                self.student_mapping, self.teacher_mapping = self.get_side_path_mappings(
                    mode=self.args.side_path_mapping_mode,
                    bias_threshold=getattr(self.args, 'tokenizer_pair_bias_threshold_side_path', 
                                          getattr(self.args, 'tokenizer_pair_bias_threshold', 0.5))
                )
                logger.info(f"Using {len(self.student_mapping)}/{len(self.tokenizer_new)} student tokens for side path alignment")
            else:
                self.student_mapping = self.teacher_mapping = None
                logger.warning("Side path mapping mode not specified, setting mappings to None")
        else:
            self.student_mapping = self.teacher_mapping = None
            
    def get_side_path_mappings(self, mode, bias_threshold):
        """Generate token mappings for side path loss calculation.
        
        Args:
            mode: Mapping mode ('mined', 'shared', 'biased', etc.)
            bias_threshold: Threshold for token pair bias
            
        Returns:
            Tuple of (student_mapping, teacher_mapping)
        """
        if mode == "mined" and self.mined_mapping is not None:
            # For mined mappings, simply return indices
            teacher_tokens = list(range(len(self.mined_mapping)))
            student_tokens = self.mined_mapping
        elif mode == "shared":
            # For shared mode, find tokens that are exactly the same in both tokenizers
            teacher_tokens = []
            student_tokens = []
            
            # Get token strings from both tokenizers
            teacher_token_strs = [self.tokenizer_teacher.convert_ids_to_tokens(i) 
                                 for i in range(len(self.tokenizer_teacher))]
            student_token_strs = [self.tokenizer_new.convert_ids_to_tokens(i) 
                                 for i in range(len(self.tokenizer_new))]
            
            # Create mapping from token strings to token IDs for student
            student_token_map = {token: i for i, token in enumerate(student_token_strs)}
            
            # Find matching tokens
            for teacher_id, token in enumerate(teacher_token_strs):
                if token in student_token_map:
                    teacher_tokens.append(teacher_id)
                    student_tokens.append(student_token_map[token])
        elif mode == "biased" and hasattr(self.args, 'tokenizer_pair_data_path'):
            # For biased mode, load token pair bias matrices
            bias_path = Path(self.args.tokenizer_pair_data_path)
            
            # Load bias matrices
            bias1_matrix = np.load(bias_path / "bias1_matrix.npy")
            bias2_matrix = np.load(bias_path / "bias2_matrix.npy")
            
            # Find biased token pairs
            teacher_tokens = []
            student_tokens = []
            
            for i in range(bias1_matrix.shape[0]):
                for j in range(bias1_matrix.shape[1]):
                    if (bias1_matrix[i, j] > bias_threshold and
                        bias2_matrix[i, j] > bias_threshold):
                        teacher_tokens.append(i)
                        student_tokens.append(j)
        else:
            # Default to empty mappings
            teacher_tokens = []
            student_tokens = []
            logger.warning(f"Unknown side path mapping mode: {mode}, using empty mappings")
            
        return student_tokens, teacher_tokens
        
    def initialize_models(self):
        """Initialize teacher and student models."""
        logger.info("Loading models...")
        
        # Load teacher model config
        self.teacher_config = AutoConfig.from_pretrained(
            self.args.teacher.pretrained_model_name_or_path
        )
        
        # Load student model config
        self.student_config = AutoConfig.from_pretrained(
            self.args.new.pretrained_model_name_or_path
        )
        
        # Load teacher model
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            self.args.teacher.pretrained_model_name_or_path,
            config=self.teacher_config,
            torch_dtype=getattr(torch, self.args.dtype)
        )
        self.teacher_model.to(self.device)
        self.teacher_model.eval()  # Teacher model is frozen
        
        # Load student model
        self.student_model = AutoModelForCausalLM.from_pretrained(
            self.args.new.pretrained_model_name_or_path,
            config=self.student_config,
            torch_dtype=getattr(torch, self.args.dtype)
        )
        self.student_model.to(self.device)
        
        # Initialize Hypernet model
        self.hypernet = Hypernet(
            hidden_size=self.student_model.config.hidden_size,
            max_seq_length=self.args.max_length_new,
            num_embeddings=2 if not self.student_config.tie_word_embeddings else 1,
            vocab_size=len(self.tokenizer_new),
            residual=self.args.hypernet.residual,
            use_attention=self.args.hypernet.use_attention,
            architecture=self.args.hypernet.architecture,
            num_layers=self.args.hypernet.num_layers,
            num_heads=self.args.hypernet.num_heads,
            hidden_expansion_factor=self.args.hypernet.hidden_expansion_factor,
            embedding_lora_rank=self.args.hypernet.embedding_lora_rank,
            embedding_lora_alpha=self.args.hypernet.embedding_lora_alpha,
            embedding_lora_position=self.args.hypernet.embedding_lora_position,
        ).to(self.device)
        
        # Set up LoRA if requested
        if self.args.train_model_mode == "lora":
            self.student_model_lora = lora.LoRAWrapper(
                self.student_model,
                lora.init_lora_params(
                    self.args,
                    self.student_model,
                    model_type=self.student_config.model_type,
                    seed=self.args.seed
                ),
                alpha=self.args.model_lora_alpha
            )
            self.trainable_model = self.student_model_lora
        else:
            # No LoRA, use base model
            self.trainable_model = self.student_model
            
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        # Set up distributed training if needed
        if sharding.is_distributed():
            self.teacher_model = sharding.wrap_model_for_distributed(
                self.teacher_model,
                model_type=self.teacher_config.model_type,
                use_model_parallel=self.args.use_model_parallel,
                find_unused_parameters=True
            )
            
            self.trainable_model = sharding.wrap_model_for_distributed(
                self.trainable_model,
                model_type=self.student_config.model_type,
                use_model_parallel=self.args.use_model_parallel,
                find_unused_parameters=True
            )
            
            self.hypernet = sharding.wrap_model_for_distributed(
                self.hypernet,
                model_type="hypernet",
                use_model_parallel=self.args.use_model_parallel,
                find_unused_parameters=True
            )
    
    def initialize_embeddings(self):
        """Initialize embeddings for teacher and student models."""
        logger.info("Initializing embeddings...")
        
        # Get teacher embeddings
        self.teacher_embeddings, _ = param.stack_embeddings(
            self.teacher_model.state_dict(),
            self.teacher_config,
            pop_embeddings=False
        )
        
        # Get student embeddings
        self.new_embeddings, _ = param.stack_embeddings(
            self.student_model.state_dict(),
            self.student_config,
            pop_embeddings=False
        )
        
        # Get original embeddings if using expanded input IDs
        if self.args.add_expanded_input_ids:
            self.original_embeddings, _ = param.stack_embeddings(
                AutoModelForCausalLM.from_pretrained(
                    self.args.original.pretrained_model_name_or_path,
                    config=AutoConfig.from_pretrained(self.args.original.pretrained_model_name_or_path)
                ).state_dict(),
                AutoConfig.from_pretrained(self.args.original.pretrained_model_name_or_path),
                pop_embeddings=False
            )
        else:
            self.original_embeddings = None
            
        # Pad embeddings to multiple of pad_to_multiple_of
        n_pad_teacher = utils.get_n_pad(
            self.teacher_embeddings.shape[0], self.args.pad_to_multiple_of
        )
        n_pad_new = utils.get_n_pad(
            self.new_embeddings.shape[0], self.args.pad_to_multiple_of
        )
        
        # Create logit masks for padded embeddings
        self.logit_mask_teacher = torch.ones(
            (self.teacher_embeddings.shape[0] + n_pad_teacher,), 
            dtype=torch.float32, 
            device=self.device
        )
        self.logit_mask_teacher[:self.teacher_embeddings.shape[0]] = 0.0
        self.logit_mask_teacher *= utils.get_large_negative_number(
            self.logit_mask_teacher.dtype, module=torch
        )
        
        self.logit_mask_new = torch.ones(
            (self.new_embeddings.shape[0] + n_pad_new,), 
            dtype=torch.float32, 
            device=self.device
        )
        self.logit_mask_new[:self.new_embeddings.shape[0]] = 0.0
        self.logit_mask_new *= utils.get_large_negative_number(
            self.logit_mask_new.dtype, module=torch
        )
        
        # Pad embeddings
        self.teacher_embeddings_padded = torch.nn.functional.pad(
            torch.tensor(self.teacher_embeddings),
            (0, 0, 0, 0, 0, n_pad_teacher),
            mode="constant",
            value=0
        ).to(self.device)
        
        self.new_embeddings_padded = torch.nn.functional.pad(
            torch.tensor(self.new_embeddings),
            (0, 0, 0, 0, 0, n_pad_new),
            mode="constant",
            value=0
        ).to(self.device)
        
        # Pad space masks
        self.space_mask_teacher_padded = torch.nn.functional.pad(
            torch.tensor(self.space_mask_teacher),
            (0, n_pad_teacher),
            mode="constant",
            value=False
        ).to(self.device)
        
        self.space_mask_new_padded = torch.nn.functional.pad(
            torch.tensor(self.space_mask_new),
            (0, n_pad_new),
            mode="constant",
            value=False
        ).to(self.device)
        
        # Initialize the hypernet with appropriate embeddings
        self.hypernet.init_rescalers(self.new_embeddings_padded[:, None, :, :])
        
        # Create overlapping embeddings mask for prefix training
        self.overlapping_embeddings_mask = torch.zeros(
            (self.new_embeddings.shape[0],),
            dtype=torch.bool,
            device=self.device
        )
        
        # Identify overlapping tokens between teacher and student
        teacher_tokens = self.tokenizer_teacher.convert_ids_to_tokens(range(len(self.tokenizer_teacher)))
        teacher_token_set = set(teacher_tokens)
        
        for i, token in enumerate(self.tokenizer_new.convert_ids_to_tokens(range(len(self.tokenizer_new)))):
            if token in teacher_token_set:
                self.overlapping_embeddings_mask[i] = True
        
        logger.info(f"Found {self.overlapping_embeddings_mask.sum().item()} overlapping tokens")
        
        # Initialize loss weights
        self.loss_weights = torch.full(
            (len(self.args.losses),),
            fill_value=self.args.uncertainty_s_init,
            dtype=torch.float32,
            device=self.device
        )
        
        # Create projectors if needed
        if self.args.latents_do_project:
            assert self.args.latents_to_align == "last_hidden_state", \
                "Latent projectors only implemented for last_hidden_state at the moment"
                
            # Initialize projector for latents
            self.latent_projector = torch.nn.Linear(
                self.new_embeddings.shape[-1], 
                self.teacher_embeddings.shape[-1]
            ).to(self.device)
        else:
            self.latent_projector = None
            
        # Create DSKD projectors if needed
        if "baseline_dskd" in self.args.losses:
            self.projector_t2s = torch.nn.Linear(
                self.teacher_embeddings.shape[-1], 
                self.new_embeddings.shape[-1]
            ).to(self.device)
            
            self.projector_s2t = torch.nn.Linear(
                self.new_embeddings.shape[-1], 
                self.teacher_embeddings.shape[-1]
            ).to(self.device)
            
            self.projector_query = torch.nn.Linear(
                self.new_embeddings.shape[-1] * 2, 
                self.teacher_embeddings.shape[-1] * 2
            ).to(self.device)
        else:
            self.projector_t2s = None
            self.projector_s2t = None
            self.projector_query = None
            
        # Create expanded input IDs projection if needed
        if self.args.add_expanded_input_ids:
            n_pad_original = utils.get_n_pad(
                self.original_embeddings.shape[0], self.args.pad_to_multiple_of
            )
            self.original_embeddings_padded = torch.nn.functional.pad(
                torch.tensor(self.original_embeddings),
                (0, 0, 0, 0, 0, n_pad_original),
                mode="constant",
                value=0
            ).to(self.device)
            
            self.expanded_input_ids_projection = torch.nn.Linear(
                self.original_embeddings.shape[-1], 
                self.original_embeddings.shape[-1],
                bias=False
            ).to(self.device)
            # Initialize to identity
            torch.nn.init.zeros_(self.expanded_input_ids_projection.weight)
        else:
            self.expanded_input_ids_projection = None
            self.original_embeddings_padded = None
        
    def initialize_dataloaders(self):
        """Initialize data loaders for training and evaluation."""
        logger.info("Initializing data loaders...")
        
        # Load dataset
        if self.args.dataset_path:
            dataset = datasets.load_from_disk(self.args.dataset_path)
        else:
            dataset = datasets.load_dataset(
                self.args.dataset_name,
                self.args.dataset_config_name,
                split=self.args.split
            )
            
        # Create training subset
        train_dataset = dataset.select(range(min(len(dataset), self.args.max_train_examples)))
        logger.info(f"Loaded {len(train_dataset)} training examples")
        
        # Create validation subset
        if self.args.val_split:
            val_dataset = dataset.select(range(
                len(train_dataset),
                min(len(dataset), len(train_dataset) + self.args.max_val_examples)
            ))
            logger.info(f"Loaded {len(val_dataset)} validation examples")
        else:
            val_dataset = None
        
        # Create tokenizer aligner collator
        self.collator = TokenizerAlignerCollator(
            tokenizer_original=self.tokenizer_teacher,
            tokenizer_new=self.tokenizer_new,
            max_teacher_length=self.args.max_length_teacher,
            max_student_length=self.args.max_length_new,
            special_tokens_mode=self.args.special_tokens_mode,
            with_expanded_input_ids=self.args.add_expanded_input_ids,
            use_chat_template=self.args.use_chat_template,
            chat_template_mode=self.args.chat_template_mode,
            loss_mask_mode=self.args.loss_mask_mode,
            tokenizer_pair_data_path=self.args.tokenizer_pair_data_path,
            tokenizer_pair_bias_threshold=self.args.tokenizer_pair_bias_threshold,
            require_bias_matrices="baseline_minED" in self.args.losses,
        )
        
        # Create datasets with collator
        train_dataset_with_collator = TokenizerAlignerDataset(
            train_dataset["text"], self.collator
        )
        
        if val_dataset:
            val_dataset_with_collator = TokenizerAlignerDataset(
                val_dataset["text"], self.collator
            )
        else:
            val_dataset_with_collator = None
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset_with_collator,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=train_dataset_with_collator.collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        if val_dataset_with_collator:
            self.val_loader = DataLoader(
                val_dataset_with_collator,
                batch_size=self.args.eval_batch_size,
                shuffle=False,
                collate_fn=val_dataset_with_collator.collate_fn,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
        else:
            self.val_loader = None
            
    def initialize_training(self):
        """Initialize optimizer and learning rate scheduler."""
        logger.info("Initializing optimizer and scheduler...")
        
        # Determine trainable parameters
        trainable_params = []
        
        # Add hypernet parameters
        for name, param in self.hypernet.named_parameters():
            if "in_rescaler" not in name and "out_rescaler" not in name:
                trainable_params.append(param)
                
        # Add student model parameters if not using LoRA
        if self.args.train_model_mode != "lora":
            for name, param in self.trainable_model.named_parameters():
                # Skip embedding parameters as they will be replaced by hypernet
                if not any(x in name for x in ["wte", "embed_tokens"]):
                    trainable_params.append(param)
        
        # Add projector parameters if they exist
        if self.latent_projector is not None:
            trainable_params.extend(self.latent_projector.parameters())
            
        if self.projector_t2s is not None:
            trainable_params.extend(self.projector_t2s.parameters())
            trainable_params.extend(self.projector_s2t.parameters())
            trainable_params.extend(self.projector_query.parameters())
            
        if self.expanded_input_ids_projection is not None:
            trainable_params.extend(self.expanded_input_ids_projection.parameters())
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.optimizer.learning_rate,
            weight_decay=self.args.optimizer.weight_decay,
            betas=(self.args.optimizer.beta1, self.args.optimizer.beta2),
            eps=self.args.optimizer.epsilon
        )
        
        # Create learning rate scheduler
        num_update_steps_per_epoch = len(self.train_loader)
        num_training_steps = self.args.num_epochs * num_update_steps_per_epoch
        
        # Choose scheduler based on configuration
        if self.args.optimizer.lr_scheduler_type == "cosine":
            lr_scheduler_fn = lr.linear_warmup_cosine_decay_with_linear_prefix(
                lr=self.args.optimizer.learning_rate,
                steps=num_training_steps,
                warmup_steps=self.args.optimizer.warmup_steps,
                alpha=self.args.optimizer.lr_min_ratio,
                prefix_steps=self.args.prefix_steps,
                prefix_lr=self.args.optimizer.learning_rate * 0.1
            )
        else:
            lr_scheduler_fn = lr.linear_warmup_linear_decay_with_linear_prefix(
                lr=self.args.optimizer.learning_rate,
                steps=num_training_steps,
                warmup_steps=self.args.optimizer.warmup_steps,
                prefix_steps=self.args.prefix_steps,
                prefix_lr=self.args.optimizer.learning_rate * 0.1
            )
            
        self.lr_scheduler = lr_scheduler_fn(self.optimizer)
        
        # Initialize EMA tracking for loss normalization
        self.loss_ema_stats = torch.zeros(
            (len(self.args.losses), 2),
            dtype=torch.float32,
            device=self.device
        )
        self.loss_ema_stats[:, :] = float('nan')  # Initialize with NaN
        
        # Initialize step counter
        self.step = 0
        
    def predict_embeddings(self):
        """Generate embeddings for the student model using the hypernet."""
        # Use hypernet to predict embeddings
        with torch.no_grad():
            vocab_indices = torch.arange(
                len(self.tokenizer_new), 
                dtype=torch.int32, 
                device=self.device
            )
            
            predicted_embeddings = self.hypernet(
                self.new_embeddings_padded[:, None, :, :],
                torch.ones((self.new_embeddings_padded.shape[0], 1), dtype=torch.bool, device=self.device),
                vocab_indices=vocab_indices
            )
            
        return predicted_embeddings

    def compute_inputs_embeds(self, batch):
        """Compute input embeddings for the model.
        
        Args:
            batch: Batch of data
            
        Returns:
            Input embeddings for the model
        """
        # Get embeddings from hypernet
        predicted_embeddings = self.predict_embeddings()
        
        # Assign embeddings to student model
        # This is a temporary copy of the model state dict
        model_state_dict = dict(self.trainable_model.state_dict())
        model_state_dict = param.assign_embeddings(
            model_state_dict,
            predicted_embeddings.cpu().numpy(),
            config=self.student_config
        )
        
        # Get input embeddings
        input_embeddings = model_state_dict[f"{param.get_input_embedding_path(self.student_config.model_type)}.weight"]
        input_embeddings = torch.tensor(input_embeddings, device=self.device)
        
        # Compute input embeddings from input IDs
        inputs_embeds = torch.nn.functional.embedding(
            batch["input_ids_new"], 
            input_embeddings
        )
        
        # If using expanded input IDs, incorporate them
        if self.args.add_expanded_input_ids and "expanded_input_ids_new" in batch:
            expanded_embeds = torch.nn.functional.embedding(
                batch["expanded_input_ids_new"],
                torch.tensor(
                    self.original_embeddings_padded[:, 0], 
                    device=self.device
                )
            )
            
            # Project expanded embeddings
            expanded_embeds = self.expanded_input_ids_projection(expanded_embeds)
            
            # Add to input embeddings
            inputs_embeds = inputs_embeds + expanded_embeds
            
        return inputs_embeds
    
    def train_step(self, batch):
        """Perform a single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary with loss and metrics
        """
        self.teacher_model.eval()
        self.trainable_model.train()
        self.hypernet.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Compute teacher outputs
        with torch.no_grad():
            teacher_inputs_embeds = torch.nn.functional.embedding(
                batch["input_ids_original"],
                torch.tensor(
                    self.teacher_embeddings_padded[:, 0], 
                    device=self.device
                )
            )
            
            teacher_outputs = self.teacher_model(
                input_ids=None,
                inputs_embeds=teacher_inputs_embeds,
                attention_mask=batch["attention_mask_original"],
                output_hidden_states=True,
                output_attentions="qkv" in self.args.latents_to_align,
                return_dict=True
            )
            
            # Compute teacher probabilities and log-probabilities
            teacher_logits = teacher_outputs.logits
            teacher_logits = teacher_logits + self.logit_mask_teacher[None, None, :]
            teacher_probs = torch.softmax(teacher_logits, dim=-1)
            teacher_logprobs = torch.log_softmax(teacher_logits, dim=-1)
        
        # Compute student outputs
        student_inputs_embeds = self.compute_inputs_embeds(batch)
        
        student_outputs = self.trainable_model(
            input_ids=None,
            inputs_embeds=student_inputs_embeds,
            attention_mask=batch["attention_mask_new"],
            output_hidden_states=True,
            output_attentions="qkv" in self.args.latents_to_align,
            return_dict=True
        )
        
        # Compute student probabilities and log-probabilities
        student_logits = student_outputs.logits
        student_logits = student_logits + self.logit_mask_new[None, None, :]
        student_probs = torch.softmax(student_logits, dim=-1)
        student_logprobs = torch.log_softmax(student_logits, dim=-1)
        
        # Get predicted embeddings
        predicted_embeddings = self.predict_embeddings()
        
        # Prepare loss arguments
        loss_args = losses.LossArgs(
            state=None,  # Not used in PyTorch implementation
            params=None,  # Not used in PyTorch implementation
            batch=batch,
            global_batch=batch,  # Same as batch in PyTorch implementation
            teacher_config=self.teacher_config,
            new_config=self.student_config,
            teacher_out=teacher_outputs,
            student_out=student_outputs,
            tokenizer_teacher=self.tokenizer_teacher,
            tokenizer_new=self.tokenizer_new,
            teacher_probs=teacher_probs,
            teacher_logprobs=teacher_logprobs,
            teacher_logits=teacher_logits,
            student_probs=student_probs,
            student_logprobs=student_logprobs,
            student_logits=student_logits,
            predicted_embeddings=predicted_embeddings,
            scalar_report={}
        )
        
        # Compute losses
        total_loss = 0.0
        scalar_report = {}
        
        for loss_idx, (loss_name, weight) in enumerate(zip(self.args.losses, self.loss_weights)):
            # Skip losses with zero weight
            if weight == 0:
                continue
                
            # Compute the specific loss
            if loss_name == "clm":
                current_loss = losses.compute_clm_loss(self.args, loss_args)
            elif loss_name == "alm_unconstrained":
                current_loss = losses.compute_alm_loss("unconstrained", self.args, loss_args)
            elif loss_name == "alm_space":
                current_loss = losses.compute_alm_loss("space", self.args, loss_args)
            elif loss_name == "alm_unbiased":
                current_loss = losses.compute_alm_loss("unbiased", self.args, loss_args)
            elif loss_name == "alm_latents":
                current_loss = losses.compute_alm_latents_loss(self.args, loss_args)
            elif loss_name.startswith("alm_side_path"):
                # Extract chunk kind from loss name
                chunk_kind = loss_name[len("alm_side_path_"):] if "_" in loss_name else "unbiased"
                current_loss = losses.compute_alm_side_path_loss(
                    chunk_kind,
                    self.student_mapping,
                    self.teacher_mapping,
                    self.args,
                    loss_args
                )
            elif loss_name == "baseline_minED":
                current_loss = losses.compute_minED_loss(self.args, loss_args)
            elif loss_name == "baseline_uld":
                current_loss = losses.compute_uld_loss(self.args, loss_args)
            elif loss_name == "baseline_dskd":
                current_loss = losses.compute_dskd_loss(self.args, loss_args)
            elif loss_name == "baseline_mined":
                current_loss = losses.compute_baseline_mined_loss(self.mined_mapping, self.args, loss_args)
            else:
                raise ValueError(f"Unknown loss: {loss_name}")
                
            # Record loss value
            scalar_report[f"loss/{loss_name}"] = current_loss.item()
            
            # Apply normalization if using uncertainty weighting
            if self.args.uncertainty_weighting:
                # Update EMA statistics
                if torch.isnan(self.loss_ema_stats[loss_idx, 0]):
                    self.loss_ema_stats[loss_idx, 0] = current_loss.detach()
                    self.loss_ema_stats[loss_idx, 1] = 1.0
                else:
                    self.loss_ema_stats[loss_idx, 0] = (
                        self.args.ema_alpha * self.loss_ema_stats[loss_idx, 0]
                        + (1 - self.args.ema_alpha) * current_loss.detach()
                    )
                    self.loss_ema_stats[loss_idx, 1] = (
                        self.args.ema_alpha * self.loss_ema_stats[loss_idx, 1]
                        + (1 - self.args.ema_alpha)
                        * (current_loss.detach() - self.loss_ema_stats[loss_idx, 0]) ** 2
                    )
                    
                # Calculate normalized loss
                running_std = torch.maximum(
                    torch.sqrt(self.loss_ema_stats[loss_idx, 1]),
                    torch.tensor(1e-6, device=self.device)
                )
                normalized_loss = (
                    current_loss - self.loss_ema_stats[loss_idx, 0]
                ) / running_std
                
                # Record normalized loss statistics
                scalar_report[f"loss/{loss_name}_normalized"] = normalized_loss.item()
                scalar_report[f"loss/{loss_name}_ema_mean"] = self.loss_ema_stats[loss_idx, 0].item()
                scalar_report[f"loss/{loss_name}_ema_var"] = self.loss_ema_stats[loss_idx, 1].item()
                
                # Add to total loss
                total_loss += weight * normalized_loss
            else:
                # Add to total loss without normalization
                total_loss += weight * current_loss
        
        # Apply special handling for prefix training
        if self.step < self.args.prefix_steps:
            # In prefix training, only update overlapping embeddings
            for param in self.hypernet.parameters():
                if param.grad is not None:
                    param.grad.zero_()
                    
            # Only allow gradients for overlapping tokens
            for name, param in self.hypernet.named_parameters():
                if "embedding" in name and param.grad is not None:
                    mask = self.overlapping_embeddings_mask.view(-1, 1, 1).expand_as(param.grad)
                    param.grad *= ~mask
        
        # Backpropagate
        total_loss.backward()
        
        # Update weights
        self.optimizer.step()
        self.lr_scheduler.step()
        
        # Increment step counter
        self.step += 1
        
        # Add learning rate to report
        scalar_report["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
        scalar_report["loss"] = total_loss.item()
        
        return scalar_report
    
    def eval_step(self, batch):
        """Perform a single evaluation step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary with loss and metrics
        """
        self.trainable_model.eval()
        self.hypernet.eval()
        
        with torch.no_grad():
            # Compute input embeddings
            student_inputs_embeds = self.compute_inputs_embeds(batch)
            
            # Forward pass through model
            logits = self.trainable_model(
                input_ids=None,
                inputs_embeds=student_inputs_embeds,
                attention_mask=batch["attention_mask_new"],
                return_dict=True
            ).logits
            
            # Compute loss
            loss = losses.cross_entropy(
                logits,
                batch["input_ids_new"],
                batch["attention_mask_new"],
                logit_mask=self.logit_mask_new
            )
            
        return {"loss": loss.item()}
    
    def train(self):
        """Train the model."""
        logger.info("Starting training...")
        
        # Set up wandb if requested
        if self.args.use_wandb:
            wandb.init(
                project=self.args.wandb_project,
                name=f"{Path(self.args.output_dir).name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                config=OmegaConf.to_container(self.args, resolve=True),
            )
        
        # Training loop
        global_step = 0
        for epoch in range(self.args.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.args.num_epochs}")
            
            # Training epoch
            self.trainable_model.train()
            self.hypernet.train()
            
            progress_bar = tqdm(
                self.train_loader, 
                desc=f"Training epoch {epoch+1}", 
                disable=not sharding.is_main_process()
            )
            
            train_losses = []
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Perform training step
                metrics = self.train_step(batch)
                train_losses.append(metrics["loss"])
                
                # Update progress bar
                progress_bar.set_postfix(loss=metrics["loss"])
                
                # Log to wandb
                if self.args.use_wandb and global_step % self.args.logging_steps == 0:
                    wandb.log(metrics, step=global_step)
                
                global_step += 1
                
                # Save checkpoint
                if global_step % self.args.save_steps == 0:
                    self.save_checkpoint(global_step, epoch)
                    
                # Evaluate
                if self.val_loader is not None and global_step % self.args.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    
                    # Log to wandb
                    if self.args.use_wandb:
                        wandb.log(
                            {f"eval/{k}": v for k, v in eval_metrics.items()},
                            step=global_step
                        )
            
            # Log epoch results
            epoch_loss = sum(train_losses) / len(train_losses)
            logger.info(f"Epoch {epoch+1} average loss: {epoch_loss:.4f}")
            
            # Save checkpoint at end of epoch
            self.save_checkpoint(global_step, epoch)
            
            # Evaluate at end of epoch
            if self.val_loader is not None:
                eval_metrics = self.evaluate()
                
                # Log to wandb
                if self.args.use_wandb:
                    wandb.log(
                        {f"eval/{k}": v for k, v in eval_metrics.items()},
                        step=global_step
                    )
        
        # Save final model
        self.save_model()
        
        logger.info("Training complete!")
    
    def evaluate(self):
        """Evaluate the model on the validation set.
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model...")
        
        if self.val_loader is None:
            logger.warning("No validation data loader available, skipping evaluation")
            return {}
        
        self.trainable_model.eval()
        self.hypernet.eval()
        
        eval_losses = []
        
        progress_bar = tqdm(
            self.val_loader, 
            desc="Evaluating", 
            disable=not sharding.is_main_process()
        )
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Perform evaluation step
            metrics = self.eval_step(batch)
            eval_losses.append(metrics["loss"])
            
            # Update progress bar
            progress_bar.set_postfix(loss=metrics["loss"])
        
        # Calculate average metrics
        avg_loss = sum(eval_losses) / len(eval_losses)
        
        logger.info(f"Evaluation loss: {avg_loss:.4f}")
        
        return {"loss": avg_loss}
    
    def save_checkpoint(self, step: int, epoch: int):
        """Save a checkpoint.
        
        Args:
            step: Current global step
            epoch: Current epoch
        """
        if not sharding.is_main_process():
            return
            
        logger.info(f"Saving checkpoint at step {step}...")
        
        # Create state dict with all necessary components
        state_dict = {
            "hypernet": self.hypernet.state_dict(),
            "step": step,
            "epoch": epoch,
            "loss_ema_stats": self.loss_ema_stats,
        }
        
        # Add projectors if they exist
        if self.latent_projector is not None:
            state_dict["latent_projector"] = self.latent_projector.state_dict()
            
        if self.projector_t2s is not None:
            state_dict["projector_t2s"] = self.projector_t2s.state_dict()
            state_dict["projector_s2t"] = self.projector_s2t.state_dict()
            state_dict["projector_query"] = self.projector_query.state_dict()
            
        if self.expanded_input_ids_projection is not None:
            state_dict["expanded_input_ids_projection"] = self.expanded_input_ids_projection.state_dict()
        
        # Add student model if not using LoRA
        if self.args.train_model_mode != "lora":
            state_dict["student_model"] = self.student_model.state_dict()
        else:
            state_dict["student_model_lora"] = self.student_model_lora.state_dict()
        
        # Save checkpoint
        self.checkpoint_manager["save"](
            state_dict,
            step=step,
            epoch=epoch
        )
    
    def save_model(self):
        """Save the final model."""
        if not sharding.is_main_process():
            return
            
        logger.info("Saving final model...")
        
        # Get predicted embeddings
        predicted_embeddings = self.predict_embeddings().cpu().numpy()
        
        # Create output directory
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer
        self.tokenizer_new.save_pretrained(output_dir)
        
        # Create model with predicted embeddings
        if self.args.train_model_mode == "lora":
            # Extract base model from LoRA wrapper
            final_model = self.student_model
        else:
            final_model = self.student_model
            
        # Assign predicted embeddings to model
        model_state_dict = dict(final_model.state_dict())
        model_state_dict = param.assign_embeddings(
            model_state_dict,
            predicted_embeddings,
            config=self.student_config
        )
        
        # Load state dict back into model
        final_model.load_state_dict(model_state_dict)
        
        # Save model
        final_model.save_pretrained(output_dir)
        
        # Save configuration
        with open(output_dir / "config.yaml", "w") as f:
            OmegaConf.save(config=self.args, f=f)
        
        logger.info(f"Model saved to {output_dir}")


@hydra.main(config_path="../configs", config_name="cross_tokenizer_distill")
def main(args: DictConfig):
    """Main entry point.
    
    Args:
        args: Configuration arguments
    """
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set up device
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Initialize distributed training if needed
    if args.use_distributed:
        sharding.initialize_distributed(backend=args.distributed_backend)
    
    # Create distiller and train
    distiller = CrossTokenizerDistiller(args)
    distiller.train()


if __name__ == "__main__":
    main()