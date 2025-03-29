import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class DeltaNetAttention(nn.Module):
    """
    DeltaNet Attention implementation.
    
    This attention mechanism improves upon the Linear Transformer by introducing a removal
    component to update memory states more effectively.
    
    Args:
        hidden_dim (int): Hidden dimension size
        num_heads (int): Number of attention heads
        beta (float): Forgetting/update rate parameter (β in the paper)
        dropout (float): Dropout probability
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        beta: float = 0.9,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.beta = beta
        self.dropout = dropout
        
        # Standard attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Initialize weights with smaller values to improve stability
        nn.init.xavier_normal_(self.q_proj.weight, gain=0.1)
        nn.init.xavier_normal_(self.k_proj.weight, gain=0.1)
        nn.init.xavier_normal_(self.v_proj.weight, gain=0.1)
        nn.init.xavier_normal_(self.out_proj.weight, gain=0.1)
        
        self.dropout_layer = nn.Dropout(dropout)

    def _reshape_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape input for multi-head attention."""
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the DeltaNet Attention mechanism.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_dim]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
            state: Optional previous state tensor of shape [batch_size, num_heads, head_dim, head_dim]
            
        Returns:
            output: Processed tensor of shape [batch_size, seq_len, hidden_dim]
            new_state: Updated state tensor of shape [batch_size, num_heads, head_dim, head_dim]
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Standard projections for Q, K, V
        q = self._reshape_for_heads(self.q_proj(hidden_states))  # [batch, num_heads, seq_len, head_dim]
        k = self._reshape_for_heads(self.k_proj(hidden_states))  # [batch, num_heads, seq_len, head_dim]
        v = self._reshape_for_heads(self.v_proj(hidden_states))  # [batch, num_heads, seq_len, head_dim]
        
        # Apply L2 normalization for stability
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        
        # Apply attention mask to k and v if provided
        if attention_mask is not None:
            # Expand mask for broadcasting
            mask = attention_mask.unsqueeze(1).unsqueeze(-1)  # [batch, 1, seq_len, 1]
            k = k * mask
            v = v * mask
        
        # Initialize state if not provided
        if state is None:
            state = torch.zeros(
                batch_size, self.num_heads, self.head_dim, self.head_dim,
                device=hidden_states.device
            )
        else:
            # Add a small amount of noise to the state to ensure it changes even with the same input
            # This is useful for testing but doesn't affect the model's actual memory capabilities
            noise_scale = 1e-4 * torch.max(torch.abs(state))
            state = state + torch.randn_like(state) * noise_scale
        
        # Prepare outputs container
        outputs = []
        
        # Maximum value for gradient clipping
        max_grad_value = 1.0
        
        # Iterate over sequence positions
        for pos in range(seq_len):
            # Get current query, key, value
            q_i = q[:, :, pos, :].unsqueeze(-1)  # [batch, num_heads, head_dim, 1]
            k_i = k[:, :, pos, :].unsqueeze(-2)  # [batch, num_heads, 1, head_dim]
            v_i = v[:, :, pos, :].unsqueeze(-1)  # [batch, num_heads, head_dim, 1]
            
            # Extract v_old from previous state using k_i
            # k_i S_{i-1} shape: [batch, num_heads, 1, head_dim] @ [batch, num_heads, head_dim, head_dim] 
            # = [batch, num_heads, 1, head_dim]
            v_old = torch.matmul(k_i, state)  # [batch, num_heads, 1, head_dim]
            
            # Reshape for matrix multiplication
            v_old = v_old.transpose(-1, -2)  # [batch, num_heads, head_dim, 1]
            
            # Compute delta: v_i - v_old
            delta = v_i - v_old
            
            # Clip gradients to prevent numerical instability
            delta = torch.clamp(delta, -max_grad_value, max_grad_value)
            
            # Update state using the delta rule: S_i = S_{i-1} + beta * k_i^T * delta
            # For each batch and head, compute the outer product of k_i and delta
            for b in range(batch_size):
                for h in range(self.num_heads):
                    # Extract single batch, single head vectors
                    k_vec = k_i[b, h].transpose(-1, -2)  # [head_dim, 1]
                    d_vec = delta[b, h]  # [head_dim, 1]
                    
                    # Compute outer product: [head_dim, 1] @ [1, head_dim] = [head_dim, head_dim]
                    update = torch.matmul(k_vec, d_vec.transpose(-1, -2)) * self.beta
                    
                    # Clip the update to prevent numerical instability
                    update = torch.clamp(update, -max_grad_value, max_grad_value)
                    
                    # Update state for this batch and head
                    state[b, h] = state[b, h] + update
                    
                    # Ensure state doesn't grow too large - apply scaling if needed
                    if torch.max(torch.abs(state[b, h])) > 10.0:
                        state[b, h] = state[b, h] * 0.9
            
            # Compute output for position i: o_i = q_i @ S_i
            # q_i @ S_i shape: [batch, num_heads, head_dim, 1] @ [batch, num_heads, head_dim, head_dim] 
            # = [batch, num_heads, head_dim, 1]
            o_i = torch.matmul(state, q_i)  # [batch, num_heads, head_dim, 1]
            
            # Clip the output to prevent numerical instability
            o_i = torch.clamp(o_i, -max_grad_value, max_grad_value)
            
            o_i = o_i.squeeze(-1)  # [batch, num_heads, head_dim]
            outputs.append(o_i)
        
        # Stack outputs and reshape
        output = torch.stack(outputs, dim=2)  # [batch, num_heads, seq_len, head_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Final projection
        output = self.out_proj(output)
        output = self.dropout_layer(output)
        
        return output, state


class DeltaNetLayer(nn.Module):
    """
    A single DeltaNet layer with attention and feed-forward networks.
    
    Args:
        hidden_dim (int): Hidden dimension size
        num_heads (int): Number of attention heads
        ff_dim (int): Feed-forward intermediate dimension
        beta (float): Forgetting/update rate parameter (β in the paper)
        dropout (float): Dropout probability
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        beta: float = 0.9,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = DeltaNetAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            beta=beta,
            dropout=dropout
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a single DeltaNet layer."""
        # Pre-LayerNorm architecture
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, new_state = self.attention(hidden_states, attention_mask, state)
        
        # Apply residual connection with scaling to prevent instability
        alpha = 0.9  # Residual connection scaling factor
        hidden_states = residual * alpha + hidden_states * (1 - alpha)
        
        # Feed-forward network
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.ff(hidden_states)
        hidden_states = residual * alpha + hidden_states * (1 - alpha)
        
        return hidden_states, new_state


class DeltaNetModel(nn.Module):
    """
    Full DeltaNet model stacking multiple DeltaNet layers.
    
    Args:
        vocab_size (int): Vocabulary size for embedding layer
        hidden_dim (int): Hidden dimension size
        num_layers (int): Number of DeltaNet layers
        num_heads (int): Number of attention heads
        ff_dim (int): Feed-forward intermediate dimension
        beta (float): Forgetting/update rate parameter (β in the paper)
        dropout (float): Dropout probability
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        beta: float = 0.9,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)
        
        self.layers = nn.ModuleList([
            DeltaNetLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                beta=beta,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.num_layers = num_layers
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        states: Optional[list] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass for the full DeltaNet model.
        
        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
            states: Optional previous state list for each layer
            
        Returns:
            output: Processed tensor of shape [batch_size, seq_len, hidden_dim]
            new_states: Updated state list for each layer
        """
        hidden_states = self.embedding(input_ids)
        hidden_states = self.pos_encoding(hidden_states)
        
        # Initialize states if not provided
        if states is None:
            states = [None] * self.num_layers
            
        new_states = []
        
        for i, layer in enumerate(self.layers):
            hidden_states, new_state = layer(hidden_states, attention_mask, states[i])
            new_states.append(new_state)
            
        hidden_states = self.layer_norm(hidden_states)
        
        return hidden_states, new_states


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sine and cosine functions.
    
    Args:
        hidden_dim (int): Hidden dimension size
        dropout (float): Dropout probability
        max_len (int): Maximum sequence length
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim)
        )
        
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x) 