import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class HopfieldAttention(nn.Module):
    """
    Hopfield Network-based attention mechanism.
    
    This implements an attention mechanism based on modern Hopfield Networks,
    providing associative memory capabilities. It can be viewed as a linear
    transformer with additional pattern completion/retrieval properties.
    
    Args:
        hidden_dim (int): Hidden dimension size
        num_heads (int): Number of attention heads
        temperature (float): Temperature parameter for the attention softmax
        dropout (float): Dropout probability
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        temperature: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.temperature = temperature
        self.dropout = dropout
        
        # Standard attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout_layer = nn.Dropout(dropout)

    def _reshape_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape input for multi-head attention."""
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _energy_function(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Energy function for the Hopfield Network.
        
        This computes the energy between queries and keys, which is used for pattern retrieval.
        """
        # Compute the similarity between queries and keys
        energy = torch.matmul(query, key.transpose(-2, -1))
        return energy / (self.temperature * math.sqrt(self.head_dim))
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Hopfield attention mechanism.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_dim]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
            state: Optional previous state tensor
            
        Returns:
            output: Processed tensor of shape [batch_size, seq_len, hidden_dim]
            state: Updated state tensor for pattern storage
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Standard projections for Q, K, V
        q = self._reshape_for_heads(self.q_proj(hidden_states))  # [batch, num_heads, seq_len, head_dim]
        k = self._reshape_for_heads(self.k_proj(hidden_states))  # [batch, num_heads, seq_len, head_dim]
        v = self._reshape_for_heads(self.v_proj(hidden_states))  # [batch, num_heads, seq_len, head_dim]
        
        # Apply attention mask to k and v if provided
        if attention_mask is not None:
            # Expand mask for broadcasting
            mask = attention_mask.unsqueeze(1).unsqueeze(-1)  # [batch, 1, seq_len, 1]
            k = k * mask
            v = v * mask
        
        # Initialize state if not provided
        if state is None:
            # For Hopfield Network, state is K^T·V
            state = torch.matmul(k.transpose(-2, -1), v)  # [batch, num_heads, head_dim, head_dim]
        else:
            # Update state with new patterns
            current_state = torch.matmul(k.transpose(-2, -1), v)
            state = state + current_state
        
        # Apply pattern retrieval using the energy function
        energy = self._energy_function(q, k)  # [batch, num_heads, seq_len, seq_len]
        
        # Apply causal mask (lower triangular)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=energy.device), 
            diagonal=1
        )
        energy = energy.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for broadcasting
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(batch_size, self.num_heads, seq_len, seq_len)
            energy = energy.masked_fill(attn_mask == 0, float('-inf'))
        
        # Apply softmax to get association probabilities
        attn_probs = F.softmax(energy, dim=-1)
        attn_probs = self.dropout_layer(attn_probs)
        
        # Retrieve patterns
        output = torch.matmul(attn_probs, v)  # [batch, num_heads, seq_len, head_dim]
        
        # Reshape output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Final projection
        output = self.out_proj(output)
        output = self.dropout_layer(output)
        
        return output, state


class HopfieldNetworkLayer(nn.Module):
    """
    A single Hopfield Network layer with attention and feed-forward networks.
    
    Args:
        hidden_dim (int): Hidden dimension size
        num_heads (int): Number of attention heads
        ff_dim (int): Feed-forward intermediate dimension
        temperature (float): Temperature parameter for the Hopfield energy function
        dropout (float): Dropout probability
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        temperature: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = HopfieldAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            temperature=temperature,
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
        """Forward pass for a single Hopfield Network layer."""
        # Pre-LayerNorm architecture
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, new_state = self.attention(hidden_states, attention_mask, state)
        hidden_states = residual + hidden_states
        
        # Feed-forward network
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.ff(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, new_state


class HopfieldNetworkModel(nn.Module):
    """
    Full Hopfield Network model stacking multiple Hopfield Network layers.
    
    Args:
        vocab_size (int): Vocabulary size for embedding layer
        hidden_dim (int): Hidden dimension size
        num_layers (int): Number of Hopfield Network layers
        num_heads (int): Number of attention heads
        ff_dim (int): Feed-forward intermediate dimension
        temperature (float): Temperature parameter for the Hopfield energy function
        dropout (float): Dropout probability
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        temperature: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)
        
        self.layers = nn.ModuleList([
            HopfieldNetworkLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                temperature=temperature,
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
        Forward pass for the full Hopfield Network model.
        
        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
            states: Optional previous states list for each layer
            
        Returns:
            output: Processed tensor of shape [batch_size, seq_len, hidden_dim]
            new_states: Updated states list for each layer
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