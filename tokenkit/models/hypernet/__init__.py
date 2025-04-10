import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-8


class EmbeddingRescaler(nn.Module):
    """Rescales embeddings to a target distribution."""
    
    def __init__(self, shape, axes=(0,)):
        super().__init__()
        expanded_shape = tuple(1 for _ in axes) + shape
        self.weight = nn.Parameter(torch.ones(expanded_shape), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(expanded_shape), requires_grad=False)
        self.axes = axes
        
    def forward(self, x):
        return x * self.weight + self.bias
    
    @staticmethod
    def scale_to(x, target=None, target_means=None, target_stds=None, axes=(0,)):
        """Calculate scaling weights and biases to normalize x to target's distribution."""
        # Compute along specified axes
        if target_stds is None and target is not None:
            if isinstance(axes, int):
                target_stds = torch.std(target, dim=axes)
            else:
                target_stds = torch.std(target, dim=axes[0])
                for axis in axes[1:]:
                    target_stds = torch.std(target_stds, dim=axis - 1)
        
        if target_means is None and target is not None:
            target_means = torch.mean(target, dim=0)
            
        # Calculate source statistics
        if isinstance(axes, int):
            x_stds = torch.std(x, dim=axes)
            x_means = torch.mean(x, dim=axes)
        else:
            x_stds = torch.std(x, dim=axes[0])
            x_means = torch.mean(x, dim=axes[0])
            for axis in axes[1:]:
                x_stds = torch.std(x_stds, dim=axis - 1)
                x_means = torch.mean(x_means, dim=axis - 1)
        
        # Calculate rescaling parameters
        w = (target_stds / (x_stds + EPSILON)).unsqueeze(0)
        b = (target_means - (x_means * w)).unsqueeze(0)
        
        return w, b


class TransformerAttention(nn.Module):
    """Multi-head attention layer."""
    
    def __init__(self, hidden_size, num_heads, dropout_rate=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + (1 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
        
        # Apply softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Final projection
        output = self.out_proj(context)
        
        return output, (q, k, v)


class IdentityAttention(nn.Module):
    """Identity attention layer that does nothing."""
    
    def forward(self, x, attention_mask=None):
        return x, (torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x))


class TransformerFeedForward(nn.Module):
    """Feed-forward network in transformer layer."""
    
    def __init__(self, hidden_size, intermediate_size, activation="silu", dropout_rate=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        if activation == "silu":
            self.activation = F.silu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class HypernetTransformerLayer(nn.Module):
    """Transformer layer for hypernet."""
    
    def __init__(
        self, 
        hidden_size, 
        intermediate_size, 
        num_heads,
        dropout_rate=0.1,
        layer_norm_eps=1e-12,
        use_attention=True,
        activation="silu"
    ):
        super().__init__()
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Attention
        if use_attention:
            self.attention = TransformerAttention(hidden_size, num_heads, dropout_rate)
        else:
            self.attention = IdentityAttention()
            
        # Feed-forward
        self.ffn = TransformerFeedForward(
            hidden_size, 
            intermediate_size, 
            activation=activation, 
            dropout_rate=dropout_rate
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, attention_mask=None):
        # Self-attention with residual connection and layer norm
        attn_output, attentions = self.attention(
            self.ln1(x), 
            attention_mask
        )
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection and layer norm
        ffn_output = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_output)
        
        return x, attentions


class Hypernet(nn.Module):
    """PyTorch implementation of the Hypernet for cross-tokenizer transfer."""
    
    def __init__(
        self,
        hidden_size: int = 0,
        max_seq_length: int = 0,
        num_embeddings: int = 2,
        vocab_size: Optional[int] = None,
        residual: bool = True,
        shared: bool = True,
        use_attention_mask: bool = True,
        pooling: str = "first",  # "first", "mean"
        residual_pooling: str = "first",  # "first", "mean"
        architecture: str = "transformer",  # 'transformer', 'linear', 'identity'
        embedding_lora_rank: int = 0,
        embedding_lora_alpha: float = 8.0,
        embedding_lora_position: str = "post",  # 'pre', 'post'
        
        # Transformer-specific parameters
        use_attention: bool = True,
        multiply_hidden_dim_by_num_embeddings: bool = True,
        hidden_expansion_factor: int = 2,
        num_layers: int = 3,
        num_heads: int = 16,
        dropout_rate: float = 0.1,
        residual_alpha: float = 8.0,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.num_embeddings = num_embeddings
        self.vocab_size = vocab_size
        self.residual = residual
        self.shared = shared
        self.use_attention_mask = use_attention_mask
        self.pooling = pooling
        self.residual_pooling = residual_pooling
        self.architecture = architecture
        self.embedding_lora_rank = embedding_lora_rank
        self.embedding_lora_alpha = embedding_lora_alpha
        self.embedding_lora_position = embedding_lora_position
        
        # Transformer-specific parameters
        self.use_attention = use_attention
        self.multiply_hidden_dim_by_num_embeddings = multiply_hidden_dim_by_num_embeddings
        self.hidden_expansion_factor = hidden_expansion_factor
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.residual_alpha = residual_alpha
        
        # Create rescalers
        self.in_rescaler = EmbeddingRescaler((num_embeddings, hidden_size), axes=(0, 1))
        self.out_rescaler = EmbeddingRescaler((num_embeddings, hidden_size), axes=(0,))
        
        # Create LoRA components if needed
        if self.embedding_lora_rank > 0:
            self.lora_embedding_a = nn.ModuleList([
                nn.Embedding(vocab_size, embedding_lora_rank)
                for _ in range(num_embeddings)
            ])
            self.lora_linear_b = nn.ModuleList([
                nn.Linear(embedding_lora_rank, hidden_size, bias=False)
                for _ in range(num_embeddings)
            ])
            # Initialize B matrices to zero for zero-init of LoRA
            for layer in self.lora_linear_b:
                nn.init.zeros_(layer.weight)
        
        # Create architecture-specific components
        if self.architecture == "transformer":
            hidden_dims = hidden_size
            if self.multiply_hidden_dim_by_num_embeddings:
                hidden_dims *= num_embeddings
                
            if self.shared:
                # Input projection if needed
                if not self.multiply_hidden_dim_by_num_embeddings:
                    self.input_linear = nn.Linear(hidden_size * num_embeddings, hidden_dims)
                
                # Positional embedding
                self.position_emb = nn.Embedding(max_seq_length, hidden_dims)
                
                # Transformer layers
                self.transformer_layers = nn.ModuleList([
                    HypernetTransformerLayer(
                        hidden_dims,
                        hidden_dims * hidden_expansion_factor,
                        num_heads,
                        dropout_rate=dropout_rate,
                        use_attention=use_attention,
                        activation="silu"
                    )
                    for _ in range(num_layers)
                ])
                
                # Output projection
                self.output_linear = nn.Linear(hidden_dims, hidden_size * num_embeddings)
                
                # Initialize output linear to zero for residual connection
                if self.residual:
                    nn.init.zeros_(self.output_linear.weight)
                    nn.init.zeros_(self.output_linear.bias)
            else:
                raise NotImplementedError("Non-shared transformer not implemented")
        
        elif self.architecture == "linear":
            if not self.shared:
                raise NotImplementedError("Non-shared linear not implemented")
                
            self.linear = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size)
                for _ in range(num_embeddings)
            ])
            
            # Initialize to identity for residual connections
            if self.residual:
                for layer in self.linear:
                    nn.init.zeros_(layer.weight)
                    nn.init.zeros_(layer.bias)
                    
        elif self.architecture == "identity":
            # Identity architecture doesn't need additional parameters
            pass
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
    
    def compute_embedding_lora(self, vocab_indices):
        """Compute LoRA embeddings for the given vocabulary indices."""
        lora_embeddings = []
        for i in range(self.num_embeddings):
            x = self.lora_embedding_a[i](vocab_indices)
            x = self.lora_linear_b[i](x)
            lora_embeddings.append(x)
        return torch.stack(lora_embeddings, dim=1)
    
    def forward(self, embeddings, attention_mask, vocab_indices=None):
        """
        Args:
            embeddings: [vocab_size, seq_length, num_embeddings, hidden_size]
            attention_mask: [vocab_size, seq_length]
            vocab_indices: Optional[Tensor] - Indices of vocabulary items
            
        Returns:
            [vocab_size, num_embeddings, hidden_size]
        """
        # Identity architecture just returns the first embeddings
        if self.architecture == "identity":
            return embeddings[:, 0, :, :]
            
        # Apply mask if needed
        if not self.use_attention_mask:
            attention_mask = torch.ones_like(attention_mask)
            
        # Apply input rescaling
        vocab_size, seq_length, _, _ = embeddings.shape
        embeddings = self.in_rescaler(embeddings)
        
        # Apply LoRA if using pre-position
        if self.embedding_lora_rank > 0 and self.embedding_lora_position == "pre":
            assert vocab_indices is not None
            
            lora_embeddings = self.compute_embedding_lora(vocab_indices)
            scaler = self.embedding_lora_alpha / math.sqrt(self.embedding_lora_rank)
            embeddings[:, 0, :, :] = embeddings[:, 0, :, :] + lora_embeddings * scaler
            
        # Process through architecture
        if self.architecture == "transformer":
            if self.shared:
                # Reshape inputs for transformer
                x = embeddings.reshape(
                    vocab_size, seq_length, self.hidden_size * self.num_embeddings
                )
                
                # Apply input projection if needed
                if not self.multiply_hidden_dim_by_num_embeddings:
                    x = self.input_linear(x)
                    
                # Add positional embeddings
                positions = torch.arange(seq_length, device=x.device).unsqueeze(0).expand(vocab_size, -1)
                x = x + self.position_emb(positions)
                
                # Process through transformer layers
                attentions = []
                for layer in self.transformer_layers:
                    x, layer_attentions = layer(x, attention_mask)
                    attentions.append(layer_attentions)
                
                # Apply pooling
                if self.pooling == "first":
                    x = x[:, 0, :]  # Take first token
                elif self.pooling == "mean":
                    # Mean pooling with attention mask
                    x = (x * attention_mask.unsqueeze(-1)).sum(dim=1) / (
                        attention_mask.sum(dim=1) + EPSILON
                    ).unsqueeze(-1)
                else:
                    raise ValueError(f"Unknown pooling method: {self.pooling}")
                    
                # Apply output projection
                x = self.output_linear(x)
                x = x.reshape(vocab_size, self.num_embeddings, self.hidden_size)
                
                # Apply residual connection if needed
                if self.residual:
                    residual_weight = self.residual_alpha / math.sqrt(self.hidden_size)
                    if self.residual_pooling == "first":
                        non_residual = embeddings[:, 0, :, :]
                    elif self.residual_pooling == "mean":
                        # Mean pooling with attention mask
                        non_residual = (embeddings * attention_mask.unsqueeze(-1).unsqueeze(-1)).sum(dim=1) / (
                            attention_mask.sum(dim=1) + EPSILON
                        ).unsqueeze(-1).unsqueeze(-1)
                    else:
                        raise ValueError(f"Unknown pooling method: {self.residual_pooling}")
                        
                    predicted_embeddings = non_residual + residual_weight * x
                else:
                    predicted_embeddings = x
            else:
                raise NotImplementedError("Non-shared transformer not implemented")
        
        elif self.architecture == "linear":
            if self.shared:
                # Process through linear layers for each embedding
                outputs = []
                for i in range(self.num_embeddings):
                    # Extract embeddings for this index
                    emb = embeddings[:, 0, i, :]
                    
                    # Apply linear transformation
                    output = self.linear[i](emb)
                    
                    # Apply residual connection if needed
                    if self.residual:
                        residual_weight = self.residual_alpha / math.sqrt(self.hidden_size)
                        output = emb + residual_weight * output
                        
                    outputs.append(output)
                    
                predicted_embeddings = torch.stack(outputs, dim=1)
            else:
                raise NotImplementedError("Non-shared linear not implemented")
        
        # Apply LoRA if using post-position
        if self.embedding_lora_rank > 0 and self.embedding_lora_position == "post":
            assert vocab_indices is not None
            
            lora_embeddings = self.compute_embedding_lora(vocab_indices)
            scaler = self.embedding_lora_alpha / math.sqrt(self.embedding_lora_rank)
            predicted_embeddings[:, 0, :] = predicted_embeddings[:, 0, :] + lora_embeddings * scaler
            
        # Apply output rescaling
        return self.out_rescaler(predicted_embeddings)
    
    def init_rescalers(self, embeddings):
        """Initialize the rescalers based on the input embeddings."""
        # Calculate target distribution for input rescaler
        in_std = math.sqrt(2.0 / self.hidden_size)
        in_w, in_b = EmbeddingRescaler.scale_to(
            embeddings, target_means=0, target_stds=in_std, axes=(0, 1)
        )
        
        # Update input rescaler parameters
        with torch.no_grad():
            self.in_rescaler.weight.copy_(in_w)
            self.in_rescaler.bias.copy_(in_b)
            
        # Generate predictions to calculate output rescaler
        with torch.no_grad():
            # Create a dummy attention mask
            dummy_mask = torch.ones(embeddings.shape[0], embeddings.shape[1], device=embeddings.device)
            # Generate predictions
            preds = self.forward(embeddings, dummy_mask)
            
            # Calculate target distribution for output rescaler
            out_w, out_b = EmbeddingRescaler.scale_to(
                preds, target=embeddings[:, 0], axes=(0,)
            )
            
            # Update output rescaler parameters
            self.out_rescaler.weight.copy_(out_w)
            self.out_rescaler.bias.copy_(out_b)
            
        return self